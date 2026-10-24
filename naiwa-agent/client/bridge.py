"""GeneralTaskAgent 桌面客户端的本地 JSON-Lines 桥接服务。

Electron 通过标准输入输出调用本进程；API Key 仅通过进程内存传递，
不写入此文件，也不会出现在响应与日志中。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
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
from agent.config import BASE_DIR
from agent.core import close_checkpointers
from agent.models import ModelSpec, _fetch_provider_models, _guess_kind
from agent.ui import extract_answer, strip_thinking
from tools.registry import all_tools

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
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def emit(event: str, data: dict[str, Any]) -> None:
    print(json.dumps({"event": event, "data": data}, ensure_ascii=False, default=str), flush=True)


def safe_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    for item in PROVIDERS.values():
        secret = os.environ.get(item["env"], "")
        if secret:
            text = text.replace(secret, "***")
    return text


def load_sessions() -> list[dict[str, Any]]:
    if not SESSION_FILE.exists():
        return []
    try:
        return json.loads(SESSION_FILE.read_text(encoding="utf-8")).get("sessions", [])
    except (OSError, json.JSONDecodeError):
        return []


def save_sessions(sessions: list[dict[str, Any]]) -> None:
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = SESSION_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps({"sessions": sessions}, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(SESSION_FILE)


def get_session(session_id: str) -> dict[str, Any]:
    session = next((s for s in load_sessions() if s["id"] == session_id), None)
    if not session:
        raise ValueError("会话不存在或已删除")
    return session


def update_session(session_id: str, updater) -> dict[str, Any]:
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
        async def initialize():
            return api.init_runtime(default_model=info["default"])
        _loop.run_until_complete(initialize())
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


async def send_message(params: dict[str, Any]) -> dict[str, Any]:
    if not _initialized:
        raise RuntimeError("请先在“模型与设置”中配置 API Key")
    session_id = params.get("sessionId", "")
    text = params.get("message", "").strip()
    if not text:
        raise ValueError("消息不能为空")
    get_session(session_id)

    def add_user(session):
        session["messages"].append({"role": "user", "content": text, "time": now()})
        if session["title"] == "新对话":
            session["title"] = text.replace("\n", " ")[:28]
        session["model_id"] = api.current_model().get("id", "")

    update_session(session_id, add_user)
    emit("session", {"sessionId": session_id})
    agent = api.get_agent()
    current = api.current_model()
    pricing = current.get("pricing") or None
    usage_session = _session_usage.setdefault(session_id, tt.SessionUsage())
    estimate = None
    try:
        estimate = await tt.estimate_run(
            agent, session_id, text, pricing=pricing,
            model_id=current.get("id", ""), avg_output=usage_session.avg_output,
            expected_calls=guess_calls(text), model_kind=current.get("kind", "chat"))
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
    async for mode, payload in agent.astream(
        {"messages": [("user", text)]}, config=config,
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            chunk, _metadata = payload
            # 工具回包由下面 updates 分支的 tool_result 呈现；不过滤会把
            # 同一份输出以"模型文本"身份再灌一次过程框。
            if getattr(chunk, "type", "") == "tool":
                continue
            content = getattr(chunk, "content", None)
            if isinstance(content, str) and content:
                streamed.append(content)
                push_process({"kind": "think", "text": content})
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
        "sessionUsage": {
            "turns": usage_session.turns,
            "input_tokens": usage_session.input_tokens,
            "output_tokens": usage_session.output_tokens,
            "total_tokens": usage_session.total_tokens,
            "cost": usage_session.cost,
        },
    }


def dispatch(method: str, params: dict[str, Any]) -> Any:
    if method == "health":
        return {
            "initialized": _initialized,
            "current": api.current_model() if _initialized else {},
            "tools": len(all_tools()),
            "provider": _current_provider,
            "sessions": len(load_sessions()),
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
    if method == "send_message":
        return _loop.run_until_complete(send_message(params))
    if method == "tools":
        return {"tools": [{"name": t.name, "description": (t.description or "").split("\n")[0]}
                          for t in all_tools()]}
    if method == "shutdown":
        _loop.run_until_complete(close_checkpointers())
        return {"closed": True}
    raise ValueError(f"未知方法: {method}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    emit("ready", {"version": 1, "providers": PROVIDERS})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            request_id = request.get("id")
            result = dispatch(request["method"], request.get("params") or {})
            print(json.dumps({"id": request_id, "ok": True, "result": result},
                             ensure_ascii=False, default=str), flush=True)
        except Exception as exc:
            print(json.dumps({"id": locals().get("request_id"), "ok": False,
                              "error": safe_error(exc)}, ensure_ascii=False), flush=True)
    _loop.close()


if __name__ == "__main__":
    main()
