"""
tests/test_history_repair.py —— "孤儿 tool_calls" 历史修复的单元测试
==================================================================
覆盖 agent/history_repair.py：
- find_orphans：各种消息组合下的删除集合（只删不增、一遍收敛）
- repair_history：通过 RemoveMessage + aupdate_state 落地删除
- looks_like_orphan_error：真实 400 文案的识别（含正/负例）

全部用假的 agent/checkpointer 交互对象，不调用真实模型。
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from agent.history_repair import find_orphans, looks_like_orphan_error, repair_history

# 截图里真实的 DeepSeek 400 文案（错误会自我延续的那条）
REAL_400 = (
    "Error invoking remote method 'bridge:request': Error: "
    "OpenAIInvalidRequestError: Error code: 400 - {'error': {'message': "
    "\"An assistant message with 'tool_calls' must be followed by tool "
    "messages responding to each 'tool_call_id'. (insufficient tool "
    "messages following tool_calls message) (request_id: "
    "dfaa62a2-49bd-4c2c-b112-32dbc0506ac1)\", 'type': "
    "'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}"
)


def ai_tool_call(msg_id="ai-1", *call_ids):
    return AIMessage(content="", id=msg_id,
                     tool_calls=[{"id": c, "name": "read_file", "args": {}}
                                 for c in call_ids])


def tool_result(call_id, msg_id=None):
    return ToolMessage(content="结果", id=msg_id or f"tool-{call_id}",
                       tool_call_id=call_id)


# ---------------------------------------------------------------------------
# find_orphans：纯函数
# ---------------------------------------------------------------------------
class TestFindOrphans:
    def test_healthy_history_returns_empty(self):
        messages = [
            HumanMessage(content="读文件", id="h-1"),
            ai_tool_call("ai-1", "c-1"),
            tool_result("c-1"),
            AIMessage(content="读完了", id="ai-2"),
        ]
        assert find_orphans(messages) == set()

    def test_empty_history(self):
        assert find_orphans([]) == set()

    def test_plain_text_history_untouched(self):
        messages = [HumanMessage(content="hi", id="h-1"),
                    AIMessage(content="hello", id="ai-1")]
        assert find_orphans(messages) == set()

    def test_orphan_assistant_tool_calls_dropped(self):
        """手动停止留下的典型坏历史：ai(tool_calls) 后面没有任何 tool 回复。"""
        messages = [
            HumanMessage(content="读文件", id="h-1"),
            ai_tool_call("ai-1", "c-1"),
        ]
        assert find_orphans(messages) == {"ai-1"}

    def test_partially_answered_drops_whole_round(self):
        """两个 tool_call 只回来一个：assistant 与已回来的 tool 消息都删。"""
        messages = [
            ai_tool_call("ai-1", "c-1", "c-2"),
            tool_result("c-1"),
        ]
        assert find_orphans(messages) == {"ai-1", "tool-c-1"}

    def test_orphan_tool_message_dropped(self):
        """tool 消息找不到任何 assistant 背书（如压缩摘要了 tool_calls）。"""
        messages = [
            HumanMessage(content="hi", id="h-1"),
            tool_result("c-9", msg_id="tool-9"),
        ]
        assert find_orphans(messages) == {"tool-9"}

    def test_surviving_round_keeps_tool_message(self):
        """一轮健康 + 一轮孤儿：健康的 tool 消息必须保留。"""
        messages = [
            ai_tool_call("ai-1", "c-1"),
            tool_result("c-1"),
            HumanMessage(content="继续", id="h-2"),
            ai_tool_call("ai-2", "c-2"),          # 孤儿轮
        ]
        assert find_orphans(messages) == {"ai-2"}

    def test_messages_without_id_are_skipped(self):
        """没有 id 的消息无法删除，跳过而不是误删。"""
        messages = [AIMessage(content="", tool_calls=[{"id": "c-1", "name": "t", "args": {}}])]
        assert find_orphans(messages) == set()


# ---------------------------------------------------------------------------
# looks_like_orphan_error：错误识别
# ---------------------------------------------------------------------------
class TestLooksLikeOrphanError:
    def test_real_deepseek_400_detected(self):
        assert looks_like_orphan_error(Exception(REAL_400)) is True

    def test_short_variant_detected(self):
        assert looks_like_orphan_error(ValueError(
            "An assistant message with 'tool_calls' must be followed by "
            "tool messages responding to each 'tool_call_id'.")) is True

    def test_unrelated_error_not_detected(self):
        assert looks_like_orphan_error(Exception("Error code: 401 invalid api key")) is False
        assert looks_like_orphan_error(asyncio.CancelledError()) is False


# ---------------------------------------------------------------------------
# repair_history：落地删除
# ---------------------------------------------------------------------------
class FakeState:
    def __init__(self, messages):
        self.values = {"messages": messages}


class FakeAgent:
    """最小假 agent：记录 aget_state / aupdate_state 的调用。"""

    def __init__(self, messages):
        self.messages = list(messages)
        self.updates = []

    async def aget_state(self, config):
        return FakeState(self.messages)

    async def aupdate_state(self, config, values):
        self.updates.append(values)
        # 模拟 reducer：删掉的移出列表
        drop = {m.id for m in values["messages"] if isinstance(m, RemoveMessage)}
        self.messages = [m for m in self.messages if getattr(m, "id", None) not in drop]


class TestRepairHistory:
    def test_healthy_history_no_write(self):
        agent = FakeAgent([HumanMessage(content="hi", id="h-1"),
                           AIMessage(content="hello", id="ai-1")])
        assert asyncio.run(repair_history(agent, "thread-1")) == 0
        assert agent.updates == []                      # 只读不改

    def test_orphan_history_repaired(self):
        agent = FakeAgent([HumanMessage(content="读文件", id="h-1"),
                           ai_tool_call("ai-1", "c-1")])
        assert asyncio.run(repair_history(agent, "thread-1")) == 1
        assert len(agent.updates) == 1
        removed = agent.updates[0]["messages"]
        assert len(removed) == 1 and isinstance(removed[0], RemoveMessage)
        assert removed[0].id == "ai-1"
        assert [getattr(m, "id", None) for m in agent.messages] == ["h-1"]

    def test_none_agent_is_safe(self):
        assert asyncio.run(repair_history(None, "thread-1")) == 0

    def test_failure_does_not_raise(self):
        class BrokenAgent:
            async def aget_state(self, config):
                raise RuntimeError("db locked")

            async def aupdate_state(self, config, values):
                raise AssertionError("修不了就不该写")

        assert asyncio.run(repair_history(BrokenAgent(), "t")) == 0


# ---------------------------------------------------------------------------
# 真 LangGraph + 真 SQLite checkpointer：验证修复在真实状态机上生效
# ---------------------------------------------------------------------------
class TestRepairWithRealGraph:
    def test_repair_on_real_graph_and_checkpointer(self, tmp_path):
        """
        不用假对象：真 create_agent + AsyncSqliteSaver。
        人为往历史里注入一条孤儿 ai(tool_calls)，repair 之后历史恢复健康，
        且消息内容原样保留（只删坏的，不动好的）。
        """
        import aiosqlite
        from langchain.agents import create_agent
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async def scenario():
            conn = aiosqlite.connect(str(tmp_path / "checkpoint.sqlite"))
            saver = AsyncSqliteSaver(conn)
            agent = create_agent(
                model=FakeListChatModel(responses=["好的"]),
                tools=[],
                checkpointer=saver,
            )
            config = {"configurable": {"thread_id": "t-1"}}

            # 1) 先跑一轮正常对话，把历史写进 SQLite
            await agent.ainvoke({"messages": [("user", "你好")]}, config=config)

            # 2) 人为注入"手动停止"留下的孤儿：ai(tool_calls) 没有 tool 回复
            orphan = AIMessage(content="", id="orphan-1",
                               tool_calls=[{"id": "c-1", "name": "read_file", "args": {}}])
            await agent.aupdate_state(config, {"messages": [orphan]})
            state = await agent.aget_state(config)
            assert "orphan-1" in find_orphans(state.values["messages"])

            # 3) 修复：只删孤儿，不动正常历史
            assert await repair_history(agent, "t-1") == 1
            state = await agent.aget_state(config)
            assert "orphan-1" not in find_orphans(state.values["messages"])
            assert [m.content for m in state.values["messages"]] == ["你好", "好的"]

            # 4) 幂等：再修一次无事发生
            assert await repair_history(agent, "t-1") == 0
            await conn.close()

        asyncio.run(scenario())
