"""
main.py —— 程序入口（CLI 对话）
================================
启动流程：
    1. 加载配置并校验（缺 API Key 给出明确指引后退出）
    2. 组装 Agent
    3. 打印能力清单
    4. 进入对话循环（流式输出思考与工具调用过程）

命令：
    /tools  列出已注册工具
    /trace  打开完整过程日志
    /new    开启新会话
    /exit   退出
"""

import asyncio
import sys

import agent.api as api
from agent import token_tracker as tt
from agent.config import settings
from agent.core import close_checkpointers
from agent.models import (ModelBuildError, ModelConfigError,
                          ModelNotFoundError, ModelSwitchError)
from agent.observability import log_step, setup_logging, stream_run
from agent.ui import console, open_trace_file
from memory.rag import availability as rag_availability
from tools.registry import all_tools

BANNER = """
╭──────────────────────────────────────────────╮
│          GeneralTaskAgent  v1.0              │
│   DeepSeek · ReAct · MCP Ready               │
╰──────────────────────────────────────────────╯
"""


def _current_trigger() -> int:
    """当前模型的上下文压缩触发点（与 agent/core.py 同一套预算公式）。"""
    cur = api.current_model()
    trigger = tt.compute_summary_trigger(
        cur.get("max_input_tokens"), cur.get("max_output_tokens"),
        static_tokens=tt.static_overhead_tokens(),
        safety_margin=settings.summarize_safety_margin,
    )
    return trigger or settings.summarize_trigger_tokens


def print_startup_info(tool_count: int) -> None:
    """打印启动信息：模型、工具数、安全阀、压缩策略、RAG 状态。"""
    current = api.current_model()
    print(BANNER)
    print(f"  当前模型  : {current.get('id', '?')} ({current.get('label', '')})")
    print(f"  工具数量  : {tool_count}")
    print(f"  安全阀    : 模型调用 ≤ {settings.model_run_limit} 次 / "
          f"工具调用 ≤ {settings.tool_run_limit} 次")
    print(f"  限流      : {settings.calls_per_minute} 次/分钟，"
          f"并发 ≤ {settings.max_concurrency}")
    window = current.get("max_input_tokens")
    window_note = f"，窗口 {window:,}" if isinstance(window, int) else ""
    print(f"  上下文压缩: 历史 ~{_current_trigger():,} tokens 触发{window_note}"
          f"（原会话内凝练为摘要，任务不中断）")
    print(f"  RAG 知识库: {rag_availability()}")
    print("\n  输入问题开始对话；/help 查看命令，/exit 退出\n")


def print_help() -> None:
    print("""
  命令：
    /help          显示本帮助
    /tools         列出所有可用工具
    /models        列出可切换的模型（标注当前 / 可用状态）
    /use <模型id>  切换模型（立即生效，无需重启，记忆保留）
    /reload-models 改完 config/models.json 后热重载模型目录
    /discover      从各已配置 Key 的供应商自动拉取全部可用模型
    /trace         打开完整过程日志（思考/工具调用全量记录）
    /usage         查看本会话累计 token 用量与成本
    /context       查看当前上下文占用与压缩触发点
    /new           开启新会话（清空上下文记忆与用量统计）
    /exit          退出程序

  每轮对话会自动显示：
    开跑前 → 预计 token 消耗（基于系统提示+工具+历史+输入）
    结束后 → 实际消耗（API 真实返回，含 ReAct 中间步）与会话累计

  长任务不怕上下文耗尽：历史接近模型窗口时会自动凝练为任务摘要，
  在原会话中继续执行（会看到 📦 提示），任务目标与进度不丢失。
""")


def print_models() -> None:
    """打印模型列表：当前用 ● 标注，密钥未配置的标注"未配置密钥"。"""
    models = api.list_models()
    if not models:
        print("\n  （模型目录为空，请检查 config/models.json）\n")
        return
    print("\n  可用模型：")
    for m in models:
        mark = "●" if m["current"] else "○"
        status = "" if m["available"] else "  [未配置密钥]"
        kind = "推理" if m["kind"] == "reasoner" else "对话"
        print(f"   {mark} {m['id']:<18} {m['label']}  ({kind}){status}")
        if m.get("description"):
            print(f"       {m['description']}")
    print("\n  用 /use <模型id> 切换，例如：/use deepseek-reasoner\n")


def _guess_expected_calls(user_input: str) -> int:
    """
    粗判本轮任务预计调用模型几次（用于放大 token 预估）。
    命中工具类关键词 → 工具任务通常 2~4 次调用（思考→调工具→观察→总结）；
    纯问答 → 1 次。只是启发式，估多了偏保守（高估比低估好）。
    """
    keywords = ("文件", "统计", "生成", "创建", "读取", "写入", "修改",
                "搜索", "查询", "运行", "执行", "代码", "目录", "xlsx",
                "docx", "pptx", "列表", "计算", "检查", "下载", "保存")
    if any(k in user_input for k in keywords):
        return 3
    return 1


def handle_use(arg: str) -> None:
    """处理 /use 命令：切换模型并给出友好反馈。"""
    model_id = arg.strip()
    if not model_id:
        print("  用法：/use <模型id>。先用 /models 查看可用模型。")
        return
    try:
        info = api.switch_model(model_id)
        print(f"  ✔ 已切换到 {info['id']}（{info['label']}），立即生效，记忆保留。")
    except ModelNotFoundError as exc:
        print(f"  ✖ {exc}")
    except ModelConfigError as exc:
        print(f"  ✖ 配置问题：{exc}")
    except ModelBuildError as exc:
        print(f"  ✖ 构建失败：{exc}")
    except ModelSwitchError as exc:
        print(f"  ✖ 切换失败：{exc}")
    except Exception as exc:  # 兜底，绝不让命令把程序搞崩
        print(f"  ✖ 切换时发生未预期错误：{type(exc).__name__}: {exc}")


async def main() -> int:
    setup_logging()

    # 输出重定向到文件/管道时统一用 UTF-8，避免 GBK 乱码；
    # 交互式控制台不受影响（Windows 控制台走 Unicode API）
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty():
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    # ---- 1. 配置校验 ----
    problems = settings.validate_runtime()
    if problems:
        print("启动失败，请先解决以下问题：\n")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        print("\n提示：配置模板见 .env.example，复制为 .env 后填写。")
        return 1

    # ---- 2. 初始化运行时（含默认模型 + Agent 组装）----
    print("正在初始化 Agent ...")
    try:
        api.init_runtime()
    except Exception as exc:
        print(f"初始化失败: {type(exc).__name__}: {exc}")
        print("排查建议：确认依赖已安装（pip install -r requirements.txt）、"
              "config/models.json 存在、且 .env 已填写 DEEPSEEK_API_KEY。")
        return 1

    tools = all_tools()
    print_startup_info(len(tools))

    # ---- 3. 对话循环 ----
    thread_id = "session-001"
    session_no = 1
    session_usage = tt.SessionUsage()   # 本会话 token 累计（/new 清零）
    compactions = 0                     # 本会话上下文压缩次数（/new 清零）
    while True:
        try:
            user_input = (await asyncio.to_thread(input, "你> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not user_input:
            continue

        # 命令处理
        if user_input.lower() in ("/exit", "/quit"):
            print("再见。")
            break
        if user_input.lower() == "/help":
            print_help()
            continue
        if user_input.lower() == "/trace":
            open_trace_file()
            continue
        if user_input.lower() == "/tools":
            print("\n可用工具：")
            for t in tools:
                desc = (t.description or "").split("\n")[0]
                print(f"  - {t.name:<14} {desc}")
            print()
            continue
        if user_input.lower() == "/models":
            print_models()
            continue
        if user_input.lower().startswith("/use"):
            handle_use(user_input[4:])
            continue
        if user_input.lower() == "/reload-models":
            try:
                n = api.reload_models()
                print(f"  ✔ 模型目录已热重载，共 {n} 个模型。")
            except Exception as exc:
                print(f"  ✖ 重载失败：{type(exc).__name__}: {exc}")
            continue
        if user_input.lower() == "/discover":
            print("  正在从各供应商拉取可用模型列表 ...")
            try:
                report = api.discover_models()
                total_new = 0
                for provider, r in report.items():
                    total_new += r.get("new", 0)
                    if r.get("ok"):
                        print(f"  ✔ {provider}: 发现 {len(r['models'])} 个，"
                              f"新增 {r['new']} 个")
                    else:
                        print(f"  ✖ {provider}: {r.get('error')}")
                print(f"  共新增 {total_new} 个模型。"
                      f"用 /models 查看，/use <id> 切换。")
            except Exception as exc:
                print(f"  ✖ 自动发现失败：{type(exc).__name__}: {exc}")
            continue
        if user_input.lower() == "/new":
            session_no += 1
            thread_id = f"session-{session_no:03d}"
            session_usage.reset()
            compactions = 0
            print(f"已开启新会话（{thread_id}），上下文与 token 统计已清零。")
            continue
        if user_input.lower() == "/usage":
            print("  " + session_usage.describe())
            print(f"  上下文压缩：本会话已触发 {compactions} 次"
                  f"（历史接近模型窗口时自动凝练，任务不中断）")
            print("  （价格按 config/models.json 中 pricing 配置估算，可自行修改）\n")
            continue
        if user_input.lower() == "/context":
            # 当前上下文占用一览：窗口 / 历史 / 静态开销 / 触发点 / 压缩次数
            agent = api.get_agent()
            cur = api.current_model()
            try:
                snapshot = await agent.aget_state(
                    {"configurable": {"thread_id": thread_id}})
                msgs = (snapshot.values or {}).get("messages", []) if snapshot else []
            except Exception:
                msgs = []
            history_tok = tt.messages_tokens(msgs)
            static_tok = tt.static_overhead_tokens()
            trigger = _current_trigger()
            window = cur.get("max_input_tokens")
            window_str = f"{window:,}" if isinstance(window, int) else "未知"
            room = trigger - history_tok
            room_pct = (room / trigger * 100) if trigger > 0 else 0
            print(f"\n  当前模型    : {cur.get('id', '?')}"
                  f"（窗口 {window_str} tokens）")
            print(f"  历史消息    : ~{history_tok:,} tokens（共 {len(msgs)} 条）")
            print(f"  静态开销    : ~{static_tok:,} tokens/次"
                  f"（系统提示 + 工具定义，每次请求都发）")
            print(f"  压缩触发点  : ~{trigger:,} tokens"
                  f"（剩余 ~{max(room, 0):,}，约 {room_pct:.0f}% 空间）")
            print(f"  本会话压缩  : {compactions} 次")
            if history_tok >= trigger:
                print("  状态        : 已接近触发点，下一轮将自动凝练历史\n")
            else:
                print("  状态        : 空间充裕\n")
            continue

        # ---- 正常对话：先预估 → 执行 → 后结算，token 全程可见 ----
        agent = api.get_agent()
        cur = api.current_model()
        pricing = cur.get("pricing") or None

        # ① 事前预估：系统提示 + 工具定义 + 历史对话 + 本次输入 → 预计输入；
        #    工具类任务预计会多轮调用模型，按次数放大；
        #    输出按本会话历史均值预估（首轮用默认值，推理模型加倍）
        est = None
        try:
            est = await tt.estimate_run(
                agent, thread_id, user_input,
                pricing=pricing,
                model_id=cur.get("id", ""),
                avg_output=session_usage.avg_output,
                expected_calls=_guess_expected_calls(user_input),
                model_kind=cur.get("kind", "chat"))
            console.print(f"[dim]⚡ {est.describe()}[/dim]")
        except Exception:
            pass                      # 预估失败绝不阻断对话

        # ② 执行（每次取最新 Agent，切换模型后立即生效；
        #    思考/工具过程进"过程框"，最终答案面板化，详见 agent/ui.py）
        before_ids = await tt.message_id_snapshot(agent, thread_id)
        await stream_run(agent, user_input, thread_id)

        # ②.5 上下文压缩检测：历史被凝练成任务摘要时明确告知用户
        #     （原会话内继续，不是开新对话——任务目标/进度/产物都在摘要里）
        try:
            compressed = await tt.detect_compaction(agent, thread_id, before_ids)
            if compressed:
                compactions += compressed
                console.print(
                    "[yellow]📦 上下文已自动凝练：历史对话压缩为任务摘要"
                    "（目标 / 进度 / 产物已保留），在原会话中继续执行，任务不中断。"
                    f"（本会话第 {compactions} 次）[/yellow]")
                log_step("context_compacted", thread_id=thread_id,
                         session_total=compactions)
        except Exception:
            pass                      # 检测失败绝不阻断对话

        # ③ 事后结算：diff 出本轮新增 AI 消息，汇总真实 usage_metadata，
        #    并对比事前预估给出偏差
        try:
            usage = await tt.actual_usage(agent, thread_id, before_ids,
                                          pricing=pricing,
                                          model_id=cur.get("id", ""))
            if usage:
                session_usage.add(usage)
                console.print(
                    f"[dim]💰 {usage.describe(session_usage)}"
                    f"{usage.deviation(est)}[/dim]")
        except Exception:
            pass                      # 统计失败同样不阻断

    # 关闭 checkpointer 的 aiosqlite 连接，否则后台线程会阻止进程退出
    await close_checkpointers()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n已中断。")
        sys.exit(0)
