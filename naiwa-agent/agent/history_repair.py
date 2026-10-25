"""
agent/history_repair.py —— 会话历史修复（清理"孤儿 tool_calls"）
================================================================

背景
----
任务执行中被**手动停止 / 超时取消 / 模型报错中断**时，LangGraph 可能
已经把"带 tool_calls 的 assistant 消息"写进 checkpoint，但对应的 tool
消息还没来得及落库。下一轮带着这段历史请求模型，OpenAI 兼容接口
（DeepSeek / 通义 / Kimi）会直接 400：

    An assistant message with 'tool_calls' must be followed by tool
    messages responding to each 'tool_call_id'.
    (insufficient tool messages following tool_calls message)

这个错误会**自我延续**：坏历史不清掉，之后每次发送都 400，整个会话
直接报废，用户只能眼看着它废掉。上下文压缩（SummarizationMiddleware）
把 tool 消息摘要掉时，也会留下同类孤儿。

策略（删除法，而非补占位 tool 消息）
------------------------------------
    - assistant 消息的 tool_calls 没有全部配上 tool 响应 → 整条删除
      （它下面已回来的 tool 消息随之成为孤儿，一并删除）；
    - tool 消息的 tool_call_id 找不到任何幸存 assistant 的背书 → 删除。
只删不增，一遍扫描即收敛，也不依赖各家接口对消息顺序的宽松度。
删除走 LangGraph 官方的 RemoveMessage + aupdate_state，不直接碰
SQLite。少掉的只是"一次未完成的工具调用轮次"，上下文语义不受影响。

入口
----
    repair_history(agent, thread_id)   发送前修复（幂等，可常驻调用）
    looks_like_orphan_error(exc)       astream 报错自愈判定
    find_orphans(messages)             纯函数：算出该删的消息 id 集合
    find_degenerate(messages)          纯函数：算出该截断的退化消息

第二类坏历史：模型"复读"退化
----------------------------
决策瘫痪时模型会输出「好。执行。好。执行。」这类循环，一条消息动辄上万
字符（实测 17,193 字符 / 3.4 万 tokens）。它一旦落进 checkpoint，下一轮又
被完整喂回模型，**自我强化**。这里按 agent/output_guard.py 的判据把它截断，
保留退化开始前的段落，后面接一句说明。
"""

from __future__ import annotations

from typing import Any, Iterable

from langchain_core.messages import RemoveMessage

# log_step 在函数内延迟导入：observability 会引用本模块，顶部直接导入会成环

# 触发自愈的 400 文案特征（各家 OpenAI 兼容接口基本一致，均小写比较）
_ORPHAN_MARKERS = (
    "tool_calls' must be followed by tool messages",
    "insufficient tool messages following tool_calls",
    "must be a response to a preceding message with 'tool_calls'",
    "messages with role 'tool' must be a response",
)


def looks_like_orphan_error(exc: BaseException) -> bool:
    """判断异常是否是"孤儿 tool_calls"导致的接口 400（用于自愈重试）。"""
    text = str(exc).lower()
    return any(marker in text for marker in _ORPHAN_MARKERS)


def _call_ids(message: Any) -> list[str]:
    """取一条消息声明的全部 tool_call_id（兼容 dict 与对象两种形态）。"""
    calls = getattr(message, "tool_calls", None) or []
    ids = []
    for call in calls:
        cid = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
        if cid:
            ids.append(cid)
    return ids


def find_orphans(messages: Iterable[Any]) -> set[str]:
    """
    扫描消息列表，返回应当删除的消息 id 集合。

    删除规则（只删不增，一遍收敛）：
        1. assistant 消息有任一 tool_call 没有对应 tool 响应 → 整条删；
        2. tool 消息的 tool_call_id 不属于任何"幸存"assistant → 删。

    无 id 的消息无法通过 RemoveMessage 删除，跳过（交由接口报错暴露）。
    """
    messages = list(messages)

    # 第一遍：哪些 tool_call 已经收到 tool 响应
    answered: set[str] = set()
    for msg in messages:
        if getattr(msg, "type", "") == "tool":
            cid = getattr(msg, "tool_call_id", None)
            if cid:
                answered.add(cid)

    # 第二遍：有孤儿 tool_call 的 assistant 消息整条删除；
    # 幸存 assistant 声明的 tool_call 才构成"合法背书"
    drop: set[str] = set()
    declared: set[str] = set()
    for msg in messages:
        if getattr(msg, "type", "") != "ai":
            continue
        calls = _call_ids(msg)
        if calls and any(cid not in answered for cid in calls):
            mid = getattr(msg, "id", None)
            if mid:
                drop.add(mid)
            continue                      # 声明作废，其 tool 响应随之成孤儿
        declared.update(calls)

    # 第三遍：没有幸存 assistant 背书的 tool 消息
    for msg in messages:
        if getattr(msg, "type", "") == "tool":
            cid = getattr(msg, "tool_call_id", None)
            if cid not in declared:
                mid = getattr(msg, "id", None)
                if mid:
                    drop.add(mid)
    return drop


_DEGENERATE_SUFFIX = "\n\n（模型输出陷入重复循环，此段已自动截断）"
_DEGENERATE_KEEP = 2000      # 退化回复最多保留多少字符（前段"犹豫独白"也没价值）


def find_degenerate(messages: Iterable[Any], min_chars: int = 400) -> list[Any]:
    """
    找出内容已退化的 assistant 消息，返回"截断后"的替换消息列表。

    只处理纯文本、没有 tool_calls 的 assistant 消息——退化只会发生在这种
    "纯说话"的回复里；带 tool_calls 的消息删掉会连带孤儿，风险大于收益。
    截断按消息 id 原地替换（LangGraph 的 add_messages 支持同 id 覆盖）。
    """
    from agent.output_guard import trim_degeneration

    patches: list[Any] = []
    for msg in messages:
        if getattr(msg, "type", "") != "ai":
            continue
        if getattr(msg, "tool_calls", None):
            continue
        content = getattr(msg, "content", "")
        if not isinstance(content, str) or len(content) < min_chars:
            continue
        if not getattr(msg, "id", None):
            continue                      # 无 id 无法原地替换
        kept, hit = trim_degeneration(content)
        if not hit:
            continue
        if len(kept) > _DEGENERATE_KEEP:
            kept = kept[:_DEGENERATE_KEEP].rstrip() + "\n…（后文为重复内容，已省略）"
        body = kept or "（这一轮只输出了重复循环，没有有效内容）"
        patches.append(msg.model_copy(update={"content": body + _DEGENERATE_SUFFIX}))
    return patches


async def repair_history(agent: Any, thread_id: str) -> int:
    """
    修复单个会话（thread）的历史：截断退化消息 + 删掉孤儿 tool_calls / 孤儿 tool 消息。

    返回改动的消息条数；历史健康时返回 0（只读一次状态，不改写）。
    agent 不可用（如无记忆降级运行）时安全返回 0。
    """
    if agent is None:
        return 0
    from agent.observability import log_step   # 延迟导入，避免与 observability 成环
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = await agent.aget_state(config)
        messages = (snapshot.values or {}).get("messages", []) if snapshot else []
        patches = find_degenerate(messages)
        drop = find_orphans(messages)
        if not patches and not drop:
            return 0
        updates: list[Any] = list(patches)
        updates.extend(RemoveMessage(id=mid) for mid in drop)
        await agent.aupdate_state(config, {"messages": updates})
        log_step("history_repaired", thread_id=thread_id,
                 dropped=len(drop), trimmed=len(patches))
        return len(drop) + len(patches)
    except Exception as exc:
        # 修复失败不阻断主流程：让原始错误自然暴露（上层还有自愈重试兜底）
        log_step("history_repair_failed", thread_id=thread_id,
                 error=f"{type(exc).__name__}: {exc}")
        return 0
