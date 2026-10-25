"""GeneralTaskAgent 桌面客户端的本地 JSON-Lines 桥接服务。

Electron 通过标准输入输出调用本进程；API Key 仅通过进程内存传递，
不写入此文件，也不会出现在响应与日志中。

并发模型（2026-10 修复"开启新对话后无法打字"）：
    - stdin 主循环只负责读请求，每个请求派发到独立工作线程处理，
      send_message 这类长任务运行期间，new_session / cancel / upload_file
      等请求仍能被即时处理；
    - asyncio 事件循环跑在独立守护线程，Agent（aiosqlite 检查点）始终
      绑定同一个循环，长任务通过 run_coroutine_threadsafe 投递；
    - send_message 支持手动取消（cancel）与硬超时（run_timeout_seconds），
      不再出现"模型接口挂起 → 客户端输入框永久 disabled"。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sys
import threading
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent.api as api
from agent import token_tracker as tt
from agent.config import BASE_DIR, settings
from agent.core import close_checkpointers
from agent.history_repair import looks_like_orphan_error, repair_history
from agent.models import ModelSpec, _fetch_provider_models, _guess_kind, _guess_vision
from agent.output_guard import StreamGuard
from agent.ui import extract_answer, strip_thinking
from agent.uploads import (build_message_display, build_user_content,
                           cleanup_staging, discard_upload,
                           resolve_attachments, stage_upload,
                           vision_unsupported_message)
from tools.registry import all_tools

# ---------------------------------------------------------------------------
# 输出退化（"复读机"）防护
# ---------------------------------------------------------------------------
# 模型陷入决策瘫痪时会退化成「好。执行。好。执行。」的死循环，实测一次烧掉
# 3.4 万 tokens 且不产生任何结果。流式过程中一旦检出就立口中止生成。
_GUARD_STEP = 200            # 每累积这么多字符做一次退化检测
_STREAM_CLOSE_TIMEOUT = 5.0  # 中止时关闭流的等待上限（秒）
_DEGENERATE_NOTICE = (
    "检测到模型输出陷入重复循环，已自动中止本次生成（省下后续上万 tokens）。"
)
_DEGENERATE_NUDGE = (
    "（系统提醒）你上一次回复陷入了「好。执行。好。执行。」的重复循环，"
    "没有产生任何结果。现在不要再权衡、不要再复述计划：立即调用合适的工具把事情做完。"
    "确实缺少必要信息时，用一句话直接问用户。"
)
_DEGENERATE_FALLBACK = (
    "模型输出陷入了重复循环，本次生成已中止。请重新发送一次指令；"
    "若反复出现，可在 .env 里调高 FREQUENCY_PENALTY（如 0.5）。"
)
# 历史会话自愈时，退化回复最多保留多少字符（前段犹豫独白同样没有价值）
_SANITIZE_KEEP_CHARS = 1200


class OutputDegenerated(RuntimeError):
    """流式过程中检出模型输出退化，用于中止本轮生成。"""

PROVIDERS = {
    "deepseek": {
        "label": "DeepSeek",
        "env": "DEEPSEEK_API_KEY",
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "default": "deepseek-chat",
        "static": ["deepseek-chat", "deepseek-reasoner"],
        "key_url": "https://platform.deepseek.com/api_keys",
    },
    "qwen": {
        "label": "通义千问",
        "env": "QWEN_API_KEY",
        "provider": "openai-compatible",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default": "qwen-plus",
        "static": ["qwen-plus"],
        "key_url": "https://bailian.console.aliyun.com/",
    },
    "kimi": {
        "label": "Kimi",
        "env": "MOONSHOT_API_KEY",
        "provider": "openai-compatible",
        "base_url": "https://api.moonshot.cn/v1",
        "default": "kimi-k2",
        "static": ["kimi-k2"],
        "key_url": "https://platform.moonshot.cn/console/api-keys",
    },
}
SESSION_FILE = Path(os.environ.get("CLIENT_SESSION_FILE", str(BASE_DIR / "data" / "client_sessions.json")))
_initialized = False
_current_provider = "deepseek"
_provider_ids: dict[str, set[str]] = {
    key: set(value["static"]) for key, value in PROVIDERS.items()
}
_session_usage: dict[str, tt.SessionUsage] = {}

# aiosqlite 与 Agent 必须始终使用同一个事件循环，否则会话记忆会失效。
# 循环跑在独立守护线程里，主线程（stdin）与各工作线程都通过
# run_coroutine_threadsafe 把协程投递上去执行。
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, name="naiwa-bridge-loop", daemon=True).start()

# 输出锁：stdout 可能被事件线程（emit）与工作线程（响应）同时写，逐行加锁防交错
_write_lock = threading.Lock()
# 会话文件锁：暂存/归档之外的 sessions JSON 读写全部串行化
_session_lock = threading.RLock()
# 运行中的长任务：session_id -> concurrent.futures.Future（供 cancel / 超时取消）
_active_runs: dict[str, concurrent.futures.Future] = {}
_active_runs_lock = threading.Lock()


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def emit(event: str, data: dict[str, Any]) -> None:
    line = json.dumps({"event": event, "data": data}, ensure_ascii=False, default=str)
    with _write_lock:
        print(line, flush=True)


def respond(request_id: Any, ok: bool, payload: dict[str, Any]) -> None:
    """线程安全的单行 JSON 响应。"""
    line = json.dumps({"id": request_id, "ok": ok, **payload},
                      ensure_ascii=False, default=str)
    with _write_lock:
        print(line, flush=True)


def safe_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    for item in PROVIDERS.values():
        secret = os.environ.get(item["env"], "")
        if secret:
            text = text.replace(secret, "***")
    return text


def run_on_loop(coro, timeout: float | None = None):
    """把协程投递到桥接事件循环执行，阻塞等待结果（可设超时）。"""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=timeout)


def load_sessions() -> list[dict[str, Any]]:
    with _session_lock:
        if not SESSION_FILE.exists():
            return []
        try:
            return json.loads(SESSION_FILE.read_text(encoding="utf-8")).get("sessions", [])
        except (OSError, json.JSONDecodeError):
            return []


def save_sessions(sessions: list[dict[str, Any]]) -> None:
    with _session_lock:
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        temp = SESSION_FILE.with_suffix(".tmp")
        temp.write_text(json.dumps({"sessions": sessions}, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(SESSION_FILE)


def get_session(session_id: str) -> dict[str, Any]:
    with _session_lock:
        session = next((s for s in load_sessions() if s["id"] == session_id), None)
        if not session:
            raise ValueError("会话不存在或已删除")
        return session


def update_session(session_id: str, updater) -> dict[str, Any]:
    with _session_lock:
        sessions = load_sessions()
        session = next((s for s in sessions if s["id"] == session_id), None)
        if not session:
            raise ValueError("会话不存在或已删除")
        updater(session)
        session["updated_at"] = now()
        save_sessions(sessions)
        return session


def public_session(session: dict[str, Any], full: bool = False) -> dict[str, Any]:
    result = {k: v for k, v in session.items() if k != "messages"}
    messages = session.get("messages", [])
    result["message_count"] = len(messages)
    result["preview"] = next((m["content"] for m in reversed(messages) if m.get("content")), "暂无消息")[:90]
    if full:
        result["messages"] = messages
    return result


def discover_provider(key: str) -> dict[str, Any]:
    info = PROVIDERS[key]
    secret = os.environ.get(info["env"], "").strip()
    if not secret:
        raise ValueError(f"请先填写 {info['label']} API Key")
    ids = _fetch_provider_models(info["base_url"], secret, timeout=12.0)
    if not ids:
        raise RuntimeError("接口返回成功，但没有发现可用模型")
    runtime = api.init_runtime(default_model=info["default"])
    added = 0
    for model_id in ids:
        _provider_ids[key].add(model_id)
        if model_id in {m["id"] for m in runtime.list_models()}:
            continue
        runtime.model_manager.register_model(ModelSpec(
            id=model_id,
            label=f"{info['label']} · {model_id}",
            provider=info["provider"],
            model=model_id,
            base_url=info["base_url"],
            api_key_env=info["env"],
            kind=_guess_kind(model_id),
            supports_tools=True,
            supports_vision=_guess_vision(model_id),
            description=f"通过 {info['label']} /models 自动发现",
        ))
        added += 1
    models = [m for m in runtime.list_models() if m["id"] in _provider_ids[key]]
    return {"models": models, "discovered": len(ids), "added": added}


def configure(params: dict[str, Any]) -> dict[str, Any]:
    global _initialized, _current_provider
    key = params.get("provider", "deepseek")
    if key not in PROVIDERS:
        raise ValueError("不支持的模型供应商")
    info = PROVIDERS[key]
    secret = params.get("apiKey", "").strip() or os.environ.get(info["env"], "").strip()
    if not secret:
        raise ValueError(f"请填写 {info['label']} API Key")
    os.environ[info["env"]] = secret
    _current_provider = key
    if not _initialized:
        run_on_loop(_init_runtime_async(info["default"]))
        _initialized = True
    report = discover_provider(key)
    current = api.current_model()
    if not current or current.get("id") not in _provider_ids[key]:
        current = api.switch_model(info["default"])
    return {
        "provider": key,
        "current": current,
        "tools": len(all_tools()),
        **report,
    }


async def _init_runtime_async(default_model: str):
    return api.init_runtime(default_model=default_model)


def create_session(params: dict[str, Any]) -> dict[str, Any]:
    sessions = load_sessions()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session = {
        "id": f"client-{stamp}-{uuid.uuid4().hex[:6]}",
        "title": params.get("title") or "新对话",
        "created_at": now(),
        "updated_at": now(),
        "model_id": api.current_model().get("id", "") if _initialized else "",
        "messages": [],
    }
    sessions.append(session)
    save_sessions(sessions)
    return public_session(session, full=True)


def guess_calls(text: str) -> int:
    keywords = ("文件", "统计", "生成", "创建", "读取", "写入", "修改", "搜索", "查询",
                "运行", "执行", "代码", "目录", "xlsx", "docx", "pptx", "列表", "检查", "保存")
    return 3 if any(k in text for k in keywords) else 1


def _usage_snapshot(usage_session: tt.SessionUsage) -> dict[str, Any]:
    return {
        "turns": usage_session.turns,
        "input_tokens": usage_session.input_tokens,
        "output_tokens": usage_session.output_tokens,
        "total_tokens": usage_session.total_tokens,
        "cost": usage_session.cost,
    }


def _vision_short_circuit(session_id: str, text: str, uploads_list,
                          current: dict[str, Any]) -> dict[str, Any]:
    """
    有照片但当前模型不支持视觉输入：不调用模型，直接落库并返回降级回答。
    """
    model_label = current.get("id") or "未知模型"
    images = [u for u in uploads_list if u.is_image]
    answer = vision_unsupported_message(model_label, images)
    process = [{
        "kind": "tool_result",
        "name": "视觉能力检查",
        "status": "error",
        "text": f"当前模型 {model_label} 不支持图片输入，已跳过模型调用。",
    }]
    display = build_message_display(text, uploads_list)

    def add_user(session):
        session["messages"].append({
            "role": "user",
            "content": display,
            "time": now(),
            "attachments": [u.to_public() for u in uploads_list],
        })
        if session["title"] == "新对话":
            session["title"] = display.replace("\n", " ")[:28]
        session["model_id"] = model_label

    def add_answer(session):
        session["messages"].append({
            "role": "assistant",
            "content": answer,
            "time": now(),
            "process": process,
        })
        session["model_id"] = model_label

    update_session(session_id, add_user)
    emit("session", {"sessionId": session_id})
    session = update_session(session_id, add_answer)
    return {
        "session": public_session(session, full=True),
        "answer": answer,
        "process": process,
        "usage": None,
        "sessionUsage": _usage_snapshot(_session_usage.setdefault(session_id, tt.SessionUsage())),
        "visionUnsupported": True,
    }


async def send_message(params: dict[str, Any]) -> dict[str, Any]:
    if not _initialized:
        raise RuntimeError("请先在“模型与设置”中配置 API Key")
    session_id = params.get("sessionId", "")
    text = (params.get("message") or "").strip()
    attachments = params.get("attachments") or []
    if not text and not attachments:
        raise ValueError("消息不能为空")
    get_session(session_id)

    # 1) 附件归档 + 分类（落盘失败直接报错，不写会话、不调模型）
    uploads_list = resolve_attachments(attachments, session_id)
    current = api.current_model()
    images = [u for u in uploads_list if u.is_image]

    # 2) 降级：有照片但模型不支持视觉 → 输出"模型无法识别"提示，不花 token
    if images and not current.get("supports_vision"):
        return _vision_short_circuit(session_id, text, uploads_list, current)

    display = build_message_display(text, uploads_list)
    content = build_user_content(text, uploads_list)

    def add_user(session):
        session["messages"].append({
            "role": "user",
            "content": display,
            "time": now(),
            "attachments": [u.to_public() for u in uploads_list],
        })
        if session["title"] == "新对话":
            session["title"] = display.replace("\n", " ")[:28]
        session["model_id"] = current.get("id", "")

    update_session(session_id, add_user)
    emit("session", {"sessionId": session_id})
    agent = api.get_agent()
    # 2.5) 会话历史修复：上次任务被手动停止 / 超时取消 / 上下文压缩，
    #   可能留下"孤儿 tool_calls"（assistant 声明了工具调用但没有 tool
    #   回复），带着它请求模型会被接口 400 拒绝，且错误会自我延续。
    #   发送前先修一遍（幂等，历史健康时只读不改）。
    await repair_history(agent, session_id)
    pricing = current.get("pricing") or None
    usage_session = _session_usage.setdefault(session_id, tt.SessionUsage())
    estimate = None
    try:
        estimate = await tt.estimate_run(
            agent, session_id, display, pricing=pricing,
            model_id=current.get("id", ""), avg_output=usage_session.avg_output,
            expected_calls=guess_calls(display), model_kind=current.get("kind", "chat"))
        emit("estimate", {"sessionId": session_id, "estimate": asdict(estimate)})
    except Exception:
        pass

    before_ids = await tt.message_id_snapshot(agent, session_id)
    answer = ""
    streamed: list[str] = []
    # 本轮“思考与行动过程”：与后端 agent/ui.py 的 ProcessBox 设计对齐——
    # 中间文本、推理、工具调用与结果都进过程框，答案单独由 extract_answer 产出。
    process: list[dict[str, Any]] = []

    def push_process(entry: dict[str, Any]) -> None:
        """记录一条过程条目并推给客户端；连续的同类文本合并为一条，避免会话文件膨胀。"""
        kind = entry.get("kind")
        if kind in ("think", "reasoning"):
            # 模型按提示词约定用【思考】标注思考，框内已有"思考"标签，剥掉避免重复
            text = (entry.get("text") or "").lstrip()
            if text.startswith("【思考】"):
                text = text[len("【思考】"):].lstrip()
            entry = {**entry, "text": text}
        if process and process[-1].get("kind") == kind and kind in ("think", "reasoning"):
            merged = f"{process[-1].get('text', '')}{entry.get('text', '')}"
            process[-1]["text"] = merged[-8000:]
        else:
            process.append(entry)
        emit("process", {"sessionId": session_id, **entry})

    config = {"configurable": {"thread_id": session_id}}

    async def consume_stream(guard: StreamGuard | None, user_input: Any) -> None:
        """跑完一轮流式；answer / streamed / process 在闭包内累积。

        guard 不为 None 时同步做退化检测：一旦命中就抛 OutputDegenerated，
        由调用方决定"重试一轮"还是"就截断结果收工"。
        """
        nonlocal answer
        stream_len = 0
        last_check = 0
        agen = agent.astream(
            {"messages": [("user", user_input)]}, config=config,
            stream_mode=["messages", "updates"],
        )
        try:
            async for mode, payload in agen:
                if mode == "messages":
                    chunk, _metadata = payload
                    # 工具回包由下面 updates 分支的 tool_result 呈现；不过滤会把
                    # 同一份输出以"模型文本"身份再灌一次过程框。
                    if getattr(chunk, "type", "") == "tool":
                        continue
                    chunk_content = getattr(chunk, "content", None)
                    if isinstance(chunk_content, str) and chunk_content:
                        streamed.append(chunk_content)
                        push_process({"kind": "think", "text": chunk_content})
                        if guard is not None:
                            stream_len += len(chunk_content)
                            if stream_len - last_check >= _GUARD_STEP:
                                last_check = stream_len
                                if guard.feed("".join(streamed)) is not None:
                                    raise OutputDegenerated()
                    reasoning = getattr(chunk, "reasoning_content", None)
                    if reasoning:
                        push_process({"kind": "reasoning", "text": reasoning})
                elif mode == "updates" and isinstance(payload, dict):
                    for _node, state in payload.items():
                        if not isinstance(state, dict):
                            continue
                        for msg in state.get("messages", []):
                            msg_type = getattr(msg, "type", "")
                            if msg_type == "ai":
                                for call in getattr(msg, "tool_calls", None) or []:
                                    entry = {
                                        "kind": "tool_call",
                                        "name": call.get("name", "未知工具"),
                                        "args": json.dumps(call.get("args", {}), ensure_ascii=False, default=str)[:1200],
                                    }
                                    push_process(entry)
                                answer = extract_answer([msg]) or answer
                            elif msg_type == "tool":
                                push_process({
                                    "kind": "tool_result",
                                    "name": getattr(msg, "name", "工具"),
                                    "status": getattr(msg, "status", "success"),
                                    "text": str(getattr(msg, "content", ""))[:1600],
                                })
        finally:
            # 中止（退化/异常/取消）时显式关闭流，避免异步生成器悬空；
            # 本轮 AI 消息尚未提交检查点，历史不会被这坨垃圾污染。
            # 加时限：取消分支里 aclose 可能卡在等上游请求收尾，不能拖住响应。
            try:
                await asyncio.wait_for(agen.aclose(), timeout=_STREAM_CLOSE_TIMEOUT)
            except (asyncio.TimeoutError, RuntimeError):
                pass
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    def reset_run() -> None:
        """清空本轮累积，准备重跑。"""
        nonlocal answer
        answer = ""
        streamed.clear()
        process.clear()
        emit("process_reset", {"sessionId": session_id})

    # 每一轮：先按退化重试预算跑；每轮内部还有一次"孤儿 tool_calls"自愈重试。
    retries_left = max(0, settings.degenerate_retry) if settings.degenerate_guard else 0
    degenerate_gave_up = False
    pending_input: Any = content
    while True:
        guard = StreamGuard() if settings.degenerate_guard else None
        try:
            for attempt in (1, 2):
                try:
                    await consume_stream(guard, pending_input)
                    break
                except OutputDegenerated:
                    raise
                except asyncio.CancelledError:
                    # 手动停止 / 超时取消：过程框留一句说明，用户消息已落库，不写答案
                    push_process({"kind": "think", "text": "任务已被手动停止或超时取消。"})
                    raise
                except Exception as exc:
                    if attempt == 1 and looks_like_orphan_error(exc):
                        push_process({"kind": "think",
                                      "text": "检测到会话历史记录异常，已自动修复，正在重试…"})
                        await repair_history(agent, session_id)
                        reset_run()
                        guard = StreamGuard() if settings.degenerate_guard else None
                        continue
                    raise
        except OutputDegenerated:
            burn = guard.hit if guard is not None else None
            detail = f"，在第 {burn} 字符处中止" if burn else ""
            if retries_left > 0:
                retries_left -= 1
                reset_run()
                push_process({"kind": "notice",
                              "text": f"{_DEGENERATE_NOTICE}{detail}，正在重试并明确要求立即动手…"})
                pending_input = _DEGENERATE_NUDGE
                continue
            degenerate_gave_up = True
            reset_run()
            push_process({"kind": "notice", "text": f"{_DEGENERATE_NOTICE}{detail}。"})
            break
        break

    if degenerate_gave_up:
        # 退化输出（"我该打开哪个……好。执行。好。执行。"）没有任何保留价值。
        # 已提交的 AI 消息（真的调过工具并给出了结论）才值得留下。
        answer = (f"{answer}\n\n（{_DEGENERATE_FALLBACK}）"
                  if answer.strip() else _DEGENERATE_FALLBACK)
    else:
        answer = answer or strip_thinking("".join(streamed))
    if not answer:
        raise RuntimeError("模型没有返回可显示的内容，请重试")

    usage = await tt.actual_usage(agent, session_id, before_ids,
                                  pricing=pricing, model_id=current.get("id", ""))
    if usage:
        usage_session.add(usage)

    def add_answer(session):
        session["messages"].append({
            "role": "assistant",
            "content": answer,
            "time": now(),
            "process": process,
        })
        session["model_id"] = current.get("id", "")

    session = update_session(session_id, add_answer)
    return {
        "session": public_session(session, full=True),
        "answer": answer,
        "process": process,
        "usage": asdict(usage) if usage else None,
        "sessionUsage": _usage_snapshot(usage_session),
    }


def _schedule_history_repair(session_id: str) -> None:
    """任务被取消/超时后，后台顺手清掉可能残留的孤儿 tool_calls（fire-and-forget）。"""
    async def _job():
        try:
            await repair_history(api.get_agent(), session_id)
        except Exception:
            pass                     # 修复失败不影响取消响应；发送前修复仍会兜底
    try:
        asyncio.run_coroutine_threadsafe(_job(), _loop)
    except RuntimeError:
        pass                         # 事件循环已停（进程退出中），无需修复


def _handle_send_message(request_id: Any, params: dict[str, Any]) -> None:
    """
    send_message 的专属工作线程：注册到 _active_runs（支持取消），
    带硬超时；取消/超时都以 {cancelled: True} 正常响应，让客户端能恢复 UI。
    """
    session_id = params.get("sessionId", "")
    future = asyncio.run_coroutine_threadsafe(send_message(params), _loop)
    with _active_runs_lock:
        # 同一会话已有任务在跑：拒绝第二个（客户端本就会拦，这里双保险）
        if session_id in _active_runs and not _active_runs[session_id].done():
            future.cancel()
            respond(request_id, False, {"error": "该会话已有任务在执行，请先等待完成"})
            return
        _active_runs[session_id] = future
    timeout = settings.run_timeout_seconds or None
    try:
        result = future.result(timeout=timeout)
        respond(request_id, True, {"result": result})
    except concurrent.futures.CancelledError:
        # 任务中断时顺手清掉可能残留的孤儿 tool_calls，避免下一轮 400
        _schedule_history_repair(session_id)
        respond(request_id, True, {"result": {
            "cancelled": True, "reason": "stopped", "sessionId": session_id}})
    except concurrent.futures.TimeoutError:
        future.cancel()
        _schedule_history_repair(session_id)
        respond(request_id, True, {"result": {
            "cancelled": True, "reason": "timeout", "sessionId": session_id}})
    except Exception as exc:
        respond(request_id, False, {"error": safe_error(exc)})
    finally:
        with _active_runs_lock:
            _active_runs.pop(session_id, None)


def upload_file(params: dict[str, Any]) -> dict[str, Any]:
    """把客户端选中的文件暂存到 staging 目录（发送时才归档进会话）。"""
    upload = stage_upload(
        path=params.get("path") or None,
        data_base64=params.get("dataBase64") or "",
        name=params.get("name") or "",
        mime=params.get("mime") or "",
    )
    return upload.to_public()


def discard_uploaded(params: dict[str, Any]) -> dict[str, Any]:
    """用户在发送前移除附件：删除暂存文件。"""
    return {"discarded": discard_upload(params.get("uploadId", ""))}


def cancel_run(params: dict[str, Any]) -> dict[str, Any]:
    """停止当前会话正在执行的任务（无任务在跑时安全返回）。"""
    session_id = params.get("sessionId", "")
    with _active_runs_lock:
        future = _active_runs.get(session_id)
    if future is None or future.done():
        return {"cancelled": False, "reason": "idle"}
    future.cancel()
    return {"cancelled": True, "reason": "stopped"}


def dispatch(method: str, params: dict[str, Any]) -> Any:
    if method == "health":
        return {
            "initialized": _initialized,
            "current": api.current_model() if _initialized else {},
            "tools": len(all_tools()),
            "provider": _current_provider,
            "sessions": len(load_sessions()),
            "run_timeout_seconds": settings.run_timeout_seconds,
            "upload_max_mb": settings.upload_max_mb,
        }
    if method == "configure":
        return configure(params)
    if method == "models":
        key = params.get("provider", _current_provider)
        if not _initialized:
            return {"models": [], "current": {}}
        ids = _provider_ids.get(key, set())
        return {"models": [m for m in api.list_models() if m["id"] in ids],
                "current": api.current_model()}
    if method == "select_model":
        if not _initialized:
            raise RuntimeError("请先配置 API Key")
        return api.switch_model(params["modelId"])
    if method == "new_session":
        return create_session(params)
    if method == "list_sessions":
        sessions = sorted(load_sessions(), key=lambda s: s["updated_at"], reverse=True)
        return {"sessions": [public_session(s) for s in sessions]}
    if method == "get_session":
        return public_session(get_session(params["sessionId"]), full=True)
    if method == "delete_session":
        session_id = params["sessionId"]
        sessions = load_sessions()
        save_sessions([s for s in sessions if s["id"] != session_id])
        return {"deleted": session_id}
    if method == "upload_file":
        return upload_file(params)
    if method == "discard_upload":
        return discard_uploaded(params)
    if method == "cancel":
        return cancel_run(params)
    if method == "tools":
        return {"tools": [{"name": t.name, "description": (t.description or "").split("\n")[0]}
                          for t in all_tools()]}
    if method == "shutdown":
        run_on_loop(close_checkpointers())
        return {"closed": True}
    raise ValueError(f"未知方法: {method}")


def handle_request(request: dict[str, Any]) -> None:
    """单个请求的处理入口（在工作线程中运行）。"""
    request_id = request.get("id")
    method = request.get("method", "")
    params = request.get("params") or {}
    try:
        if method == "send_message":
            _handle_send_message(request_id, params)
            return
        result = dispatch(method, params)
        respond(request_id, True, {"result": result})
    except Exception as exc:
        respond(request_id, False, {"error": safe_error(exc)})


def sanitize_sessions() -> int:
    """
    启动时把历史会话里"复读退化"的回复截断，返回处理条数。

    退化回复（一次上万字符的「好。执行。好。执行。」）是旧版本的产物：它躺在
    client_sessions.json 里，界面上是一坨几十屏的垃圾，还会在下一次发送时被
    检查点还原进模型上下文。这里做一次性自愈：只截断、只改内容，不动其他字段。
    截断后仍超长的（前段的犹豫独白也没价值），只留开头一段。
    """
    from agent.output_guard import trim_degeneration

    try:
        sessions = load_sessions()
    except Exception:
        return 0
    changed = 0
    for session in sessions:
        for msg in session.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content") or ""
            if not isinstance(content, str) or len(content) < 400:
                continue
            kept, hit = trim_degeneration(content)
            if not hit:
                continue
            if len(kept) > _SANITIZE_KEEP_CHARS:
                kept = kept[:_SANITIZE_KEEP_CHARS].rstrip() + "\n\n…（后文为重复内容，已省略）"
            msg["content"] = (kept or "（这一轮模型输出陷入重复循环，没有有效内容）")
            msg["content"] += "\n\n（本次回复出现重复循环，已自动截断）"
            for entry in msg.get("process") or []:
                text = entry.get("text")
                if isinstance(text, str) and len(text) > 8000:
                    entry["text"] = text[:2000] + "\n…（重复内容已截断）"
            changed += 1
    if changed:
        try:
            save_sessions(sessions)
        except Exception:
            return 0
    return changed


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    # 清理上次客户端异常退出遗留的暂存文件
    try:
        removed = cleanup_staging()
        if removed:
            print(f"[uploads] 清理过期暂存文件 {removed} 个", file=sys.stderr, flush=True)
    except Exception:
        pass
    # 清掉旧版本留下的"复读"回复（一次上万字符，纯垃圾）
    try:
        trimmed = sanitize_sessions()
        if trimmed:
            print(f"[sessions] 截断退化的历史回复 {trimmed} 条", file=sys.stderr, flush=True)
    except Exception:
        pass
    emit("ready", {"version": 1, "providers": PROVIDERS})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        # 每个请求一个工作线程：长任务（send_message）不再阻塞 stdin，
        # 期间的新建对话 / 取消 / 上传请求都能被即时处理
        threading.Thread(target=handle_request, args=(request,), daemon=True).start()
    _loop.call_soon_threadsafe(_loop.stop)


if __name__ == "__main__":
    main()
