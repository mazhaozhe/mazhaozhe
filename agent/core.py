"""
agent/core.py —— Agent 组装（项目心脏）
========================================
把模型、工具、中间件链、会话记忆组装成一个可运行的 Agent。

中间件顺序（重要性递减，越靠前越先拦截）：
    1. ModelCallLimit   —— 安全阀：模型调用次数上限（防无限循环白耗 token）
    2. ToolCallLimit    —— 安全阀：工具调用次数上限
    3. RateLimit        —— 限流：并发 + 频率
    4. ToolError        —— 异常兜底：把异常转成模型可见的错误消息，不崩溃
    5. ToolRetry        —— 自动重试：指数退避（必须在 ToolError 内层）
    6. ModelRetry       —— 模型 API 瞬时故障重试
    7. Summarization    —— 长对话上下文压缩：历史接近窗口时装订为任务摘要，
                           在原会话内继续（用户无感知，任务不中断）
    8. TodoList         —— 复杂任务自动规划（提供 write_todos 工具）
"""

from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain_deepseek import ChatDeepSeek

from agent import token_tracker as tt
from agent.config import settings
from agent.observability import log_step
from agent.prompts import SYSTEM_PROMPT
from memory.cache import enable_llm_cache
from memory.rag import build_rag_tools
from middleware.rate_limit import RateLimitMiddleware
from tools.registry import all_tools, suggest_tools

# 触发工具注册：导入模块时 @register 会执行
import tools.coding      # noqa: F401,E402
import tools.documents   # noqa: F401,E402
import tools.search      # noqa: F401,E402


# ---------------------------------------------------------------------------
# 工具错误处理器：把异常转成"模型能看懂并修正"的提示
# ---------------------------------------------------------------------------
def on_tool_error(exc: Exception, request) -> str | None:
    """
    ToolErrorMiddleware 的回调。

    返回字符串 → 作为 ToolMessage(status="error") 回给模型，让它修正后重试；
    返回 None   → 继续向上抛（不建议，Agent 会中断）。
    """
    # request.tool_call 是本次调用信息（含 name / args），做防御式读取
    tool_call = getattr(request, "tool_call", {}) or {}
    name = tool_call.get("name", "<unknown>")
    args = tool_call.get("args", {})

    hint = ""
    # 参数错误类异常：提示模型核对参数说明
    if isinstance(exc, (TypeError, ValueError)):
        hint = " 请根据工具的参数说明检查参数类型与必填项后重试。"
    elif isinstance(exc, (TimeoutError, ConnectionError)):
        hint = " 这是网络/超时问题，可以稍后重试或换一种方式完成。"
    elif isinstance(exc, PermissionError):
        hint = " 该操作被安全策略拦截，请改用其他安全的实现方式。"
    elif isinstance(exc, FileNotFoundError):
        hint = " 请先用 list_dir 确认路径是否存在，再重试。"

    # 若模型调用了不存在的工具，给出相近名建议（防幻觉）
    if "not found" in str(exc).lower() and "tool" in str(exc).lower():
        close = suggest_tools(name)
        if close:
            hint = " 你是否想调用：" + "、".join(f"`{c}`" for c in close) + "？"

    log_step("tool_error", tool=name, args=str(args)[:200],
             error=f"{type(exc).__name__}: {exc}")

    return (
        f"工具 `{name}` 执行失败：{type(exc).__name__}: {exc}。{hint}"
        f"（参数：{str(args)[:200]}）"
    )


# ---------------------------------------------------------------------------
# 会话记忆：SQLite checkpointer（跨进程持久化，支持中断恢复）
# ---------------------------------------------------------------------------
# 已打开的 checkpointer 连接，程序退出时需要显式关闭，
# 否则 aiosqlite 的后台线程会阻止进程退出（表现为退出时卡死）
_OPEN_CHECKPOINT_CONNS: list = []


async def close_checkpointers() -> None:
    """关闭所有 checkpointer 的 aiosqlite 连接。应在程序退出前调用。"""
    for conn in _OPEN_CHECKPOINT_CONNS:
        try:
            await conn.close()
        except Exception:
            pass
    _OPEN_CHECKPOINT_CONNS.clear()


def build_checkpointer():
    """对外暴露的 checkpointer 构造（供 agent/runtime.py 复用，热切换时保留记忆）。"""
    return _build_checkpointer()


def _build_checkpointer():
    """创建 SQLite checkpointer。失败时降级为 None（无记忆但可运行）。"""
    db = settings.abs_path(settings.checkpoint_db)
    # main.py 通过 asyncio 异步流式调用，必须用异步 checkpointer，
    # 同步的 SqliteSaver 会抛 NotImplementedError
    try:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        # aiosqlite.connect() 是惰性连接：首次使用时才启动后台线程，
        # 因此可以在同步的 build_agent() 里安全构造
        conn = aiosqlite.connect(str(db))
        _OPEN_CHECKPOINT_CONNS.append(conn)
        return AsyncSqliteSaver(conn)
    except Exception as exc:
        print(f"[warn] 会话持久化不可用（本次运行将无记忆）: "
              f"{type(exc).__name__}: {exc}")
        return None


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def build_model(use_reasoner: bool = False) -> ChatDeepSeek:
    """构造 DeepSeek 模型实例。use_reasoner=True 时切深度推理模型。"""
    return ChatDeepSeek(
        model=settings.reasoner_model if use_reasoner else settings.main_model,
        api_key=settings.deepseek_api_key,
        api_base=settings.deepseek_base_url,
        temperature=settings.temperature,
        max_retries=2,          # SDK 层重试（中间件层还有一层）
    )


# ---------------------------------------------------------------------------
# 上下文压缩触发点（预算式：用满窗口，但绝不让请求溢出）
# ---------------------------------------------------------------------------
def _summarize_trigger(model) -> tuple:
    """
    决定 SummarizationMiddleware 的触发阈值（history 消息 token 数）。

    优先按模型窗口预算计算：窗口 - 最大输出 - 静态开销 - 安全余量。
    换更大窗口的模型（/use 热切换会重建 Agent）时触发点自动跟着抬高；
    窗口未知的模型（如自动发现且无 profile）退化为 .env 的绝对阈值。
    """
    profile = getattr(model, "profile", None)
    max_input = profile.get("max_input_tokens") if isinstance(profile, dict) else None
    max_output = getattr(model, "max_tokens", None)

    trigger = tt.compute_summary_trigger(
        max_input, max_output,
        static_tokens=tt.static_overhead_tokens(),
        safety_margin=settings.summarize_safety_margin,
    )
    if trigger is None:
        trigger = settings.summarize_trigger_tokens
        reason = f"模型窗口未知，用绝对阈值兜底 {trigger} tokens"
    else:
        reason = (f"窗口预算：{max_input} - 输出 {max_output or '默认'} "
                  f"- 静态开销 - 安全余量 → {trigger} tokens")
    log_step("summarize_trigger", trigger_tokens=trigger, reason=reason)
    return ("tokens", trigger)


# ---------------------------------------------------------------------------
# Agent 组装
# ---------------------------------------------------------------------------
def build_agent(extra_tools: list | None = None, use_reasoner: bool = False,
                model=None, checkpointer=None, model_label: str | None = None):
    """
    组装并返回可运行的 Agent。

    参数:
        extra_tools: 额外工具（如 MCP 工具），会追加到注册表工具之后
        use_reasoner: 是否使用深度推理模型（仅在未显式传 model 时生效）
        model:        外部传入的已实例化聊天模型（用于多模型热切换）；
                      为 None 时按 .env 用默认 DeepSeek 构造
        checkpointer: 外部传入的会话记忆；为 None 时新建。
                      热切换时传入旧 checkpointer，记忆不丢
        model_label:  日志里显示的模型名（默认按 settings 推断）
    """
    # 1) 开启 LLM 缓存（重复请求零 token）
    enable_llm_cache(settings.abs_path(settings.cache_db))

    # 2) 模型：外部传入优先，否则按 .env 构造默认 DeepSeek
    if model is None:
        model = build_model(use_reasoner)

    # 3) 工具：注册表中的全部工具 + RAG 工具 + 外部工具
    tools = all_tools() + build_rag_tools() + (extra_tools or [])
    if not tools:
        raise RuntimeError("没有可用工具，Agent 无法工作。请检查 tools/ 下的注册代码。")

    # 4) 中间件链（顺序即优先级，见文件头注释）
    middleware = [
        # --- 安全阀：先拦截，未执行就省 token ---
        ModelCallLimitMiddleware(
            run_limit=settings.model_run_limit, exit_behavior="end"),
        ToolCallLimitMiddleware(
            run_limit=settings.tool_run_limit, exit_behavior="continue"),
        # --- 限流：并发 + 频率 ---
        RateLimitMiddleware(
            calls_per_minute=settings.calls_per_minute,
            max_concurrency=settings.max_concurrency),
        # --- 可靠性：Error 在外、Retry 在内（官方要求）---
        ToolErrorMiddleware(on_tool_error),
        ToolRetryMiddleware(
            max_retries=settings.tool_retries,
            backoff_factor=2.0,      # 指数退避：1s → 2s → 4s
            initial_delay=1.0,
            max_delay=30.0,
            jitter=True,             # ±25% 抖动，防惊群
        ),
        ModelRetryMiddleware(max_retries=3, backoff_factor=2.0),
        # --- 长对话上下文压缩（原会话内凝练，任务不中断）---
        # trigger 预算式动态计算；keep 保留最近 N 条原文保证连续性；
        # trim=0 表示不裁剪——待压缩历史的开头是原始任务目标，裁掉它就丢了目标
        SummarizationMiddleware(
            model=model,
            trigger=_summarize_trigger(model),
            keep=("messages", settings.summarize_keep_messages),
            trim_tokens_to_summarize=settings.summarize_trim_tokens or None,
        ),
        # --- 任务规划 ---
        TodoListMiddleware(),
    ]

    # 5) 会话记忆：外部传入则复用（热切换保留记忆），否则新建
    if checkpointer is None:
        checkpointer = _build_checkpointer()

    # 6) 组装
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
        checkpointer=checkpointer,
    )

    # 日志里的模型名：优先用显式标签，其次按 settings 推断
    if model_label is None:
        model_label = settings.reasoner_model if use_reasoner else settings.main_model
    log_step(
        "agent_built",
        model=model_label,
        tools=[t.name for t in tools],
        middleware=[type(m).__name__ for m in middleware],
        memory=checkpointer is not None,
    )
    return agent
