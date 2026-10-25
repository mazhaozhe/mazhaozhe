"""
tests/test_bridge_history_repair.py —— 桥接层历史修复的集成测试
===============================================================
用假 agent 验证 client/bridge.py 的孤儿 tool_calls 防护，不调用真实模型：
- send_message 开跑前：坏历史被修复（aupdate_state 收到 RemoveMessage）
- astream 报孤儿 400：自动修复后重试一轮并成功
- 重试后仍报同一错误：只重试一次，错误上抛不无限循环
- cancel 有任务在跑：后台顺手修复历史
- cancel 空闲：不触发修复
"""

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

CLIENT_DIR = Path(__file__).resolve().parent.parent / "client"
if str(CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_DIR))

import bridge  # noqa: E402
from agent.config import settings  # noqa: E402

ORPHAN_400 = Exception(
    "OpenAIInvalidRequestError: Error code: 400 - {'error': {'message': "
    "\"An assistant message with 'tool_calls' must be followed by tool "
    "messages responding to each 'tool_call_id'. (insufficient tool "
    "messages following tool_calls message)\", 'type': "
    "'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}"
)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """隔离会话文件，避免污染项目 data/。"""
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(bridge, "SESSION_FILE", tmp_path / "sessions.json")
    monkeypatch.setattr(bridge, "_initialized", True)
    yield


class FakeState:
    def __init__(self, messages):
        self.values = {"messages": messages}


class RepairAgent:
    """
    带历史修复能力的假 agent：
    - history 可控（模拟好/坏历史）
    - astream 可配置为：正常吐答案 / 首轮抛孤儿 400 / 永远抛
    - 记录 aupdate_state 收到的删除集合
    """

    def __init__(self, history=None, astream_behavior="ok"):
        self.history = list(history or [])
        self.behavior = astream_behavior
        self.stream_calls = 0
        self.dropped = []

    async def aget_state(self, config):
        return FakeState(self.history)

    async def aupdate_state(self, config, values):
        ids = {m.id for m in values["messages"] if isinstance(m, RemoveMessage)}
        self.dropped.append(ids)
        self.history = [m for m in self.history if getattr(m, "id", None) not in ids]

    def astream(self, input_data, config=None, stream_mode=None):
        self.stream_calls += 1
        first = self.behavior == "fail_once" and self.stream_calls == 1

        async def generator():
            if first:
                raise ORPHAN_400
            if self.behavior == "always_fail":
                raise ORPHAN_400
            yield ("updates", {"agent": {"messages": [
                AIMessage(content="修复后重试成功", id="ai-final",
                          usage_metadata={"input_tokens": 10, "output_tokens": 5,
                                          "total_tokens": 15})]}})

        return generator()


def orphan_history():
    """手动停止后的典型坏历史：ai(tool_calls) 没有 tool 回复。"""
    return [
        HumanMessage(content="这个文件写了什么内容", id="h-1"),
        AIMessage(content="", id="ai-1",
                  tool_calls=[{"id": "call-1", "name": "read_file", "args": {}}]),
    ]


def text_model():
    return {"id": "deepseek-chat", "kind": "chat", "supports_vision": False, "pricing": {}}


def new_session():
    return bridge.dispatch("new_session", {})


def wait_for(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------------------
# 发送前修复
# ---------------------------------------------------------------------------
class TestRepairBeforeSend:
    def test_broken_history_repaired_before_stream(self, monkeypatch):
        monkeypatch.setattr(bridge.api, "current_model", text_model)
        agent = RepairAgent(history=orphan_history())
        monkeypatch.setattr(bridge.api, "get_agent", lambda: agent)

        session = new_session()
        result = asyncio.run(bridge.send_message({
            "sessionId": session["id"], "message": "继续", "attachments": []}))

        assert result["answer"] == "修复后重试成功"
        assert agent.dropped == [{"ai-1"}]            # 开跑前修掉了孤儿 tool_calls
        assert agent.stream_calls == 1                # 一次成功，不需要重试

    def test_healthy_history_not_rewritten(self, monkeypatch):
        monkeypatch.setattr(bridge.api, "current_model", text_model)
        agent = RepairAgent(history=[
            HumanMessage(content="hi", id="h-1"),
            AIMessage(content="hello", id="ai-0"),
        ])
        monkeypatch.setattr(bridge.api, "get_agent", lambda: agent)

        session = new_session()
        asyncio.run(bridge.send_message({
            "sessionId": session["id"], "message": "继续", "attachments": []}))

        assert agent.dropped == []                    # 历史健康：只读不改


# ---------------------------------------------------------------------------
# astream 自愈重试
# ---------------------------------------------------------------------------
class TestSelfHealRetry:
    def test_orphan_400_triggers_repair_and_retry(self, monkeypatch):
        """首轮抛孤儿 400 → 修复 → 第二轮成功（不重试则用户会话报废）。"""
        monkeypatch.setattr(bridge.api, "current_model", text_model)
        agent = RepairAgent(history=[], astream_behavior="fail_once")
        monkeypatch.setattr(bridge.api, "get_agent", lambda: agent)

        session = new_session()
        result = asyncio.run(bridge.send_message({
            "sessionId": session["id"], "message": "继续", "attachments": []}))

        assert agent.stream_calls == 2
        assert result["answer"] == "修复后重试成功"

    def test_retry_only_once(self, monkeypatch):
        """修复后仍报同一错误：只重试一次，错误上抛，绝不无限循环。"""
        monkeypatch.setattr(bridge.api, "current_model", text_model)
        agent = RepairAgent(history=[], astream_behavior="always_fail")
        monkeypatch.setattr(bridge.api, "get_agent", lambda: agent)

        session = new_session()
        with pytest.raises(Exception, match="tool_calls"):
            asyncio.run(bridge.send_message({
                "sessionId": session["id"], "message": "继续", "attachments": []}))

        assert agent.stream_calls == 2                # 1 次原始 + 1 次重试，封顶

    def test_unrelated_error_not_retried(self, monkeypatch):
        """普通 500/超时类错误不走自愈重试（重试也没意义）。"""
        monkeypatch.setattr(bridge.api, "current_model", text_model)

        class FlakyAgent(RepairAgent):
            def astream(self, input_data, config=None, stream_mode=None):
                self.stream_calls += 1

                async def generator():
                    raise ConnectionError("read timeout")
                    yield                  # unreachable：让函数成为 async generator

                return generator()

        agent = FlakyAgent(history=[])
        monkeypatch.setattr(bridge.api, "get_agent", lambda: agent)

        session = new_session()
        with pytest.raises(ConnectionError):
            asyncio.run(bridge.send_message({
                "sessionId": session["id"], "message": "继续", "attachments": []}))

        assert agent.stream_calls == 1


# ---------------------------------------------------------------------------
# cancel 顺手修复
# ---------------------------------------------------------------------------
class HangingAgent(RepairAgent):
    """astream 永远挂起（模拟模型接口卡住的长任务）。"""

    def astream(self, input_data, config=None, stream_mode=None):
        self.stream_calls += 1

        async def generator():
            await asyncio.sleep(30)
            yield                  # pragma: no cover

        return generator()


class TestCancelRepair:
    def test_schedule_repair_runs_in_background(self, monkeypatch):
        """_schedule_history_repair：fire-and-forget 地把修复协程排上事件循环。"""
        monkeypatch.setattr(bridge.api, "current_model", text_model)
        agent = RepairAgent(history=orphan_history())
        monkeypatch.setattr(bridge.api, "get_agent", lambda: agent)

        bridge._schedule_history_repair("sess-1")
        assert wait_for(lambda: agent.dropped and "ai-1" in agent.dropped[0])

    def test_cancel_running_task_repairs_and_responds(self, monkeypatch):
        """
        完整链路：工作线程跑 send_message（挂起中）→ cancel →
        协程被取消、响应正常返回（客户端能恢复 UI）、历史已被修复。
        """
        monkeypatch.setattr(bridge.api, "current_model", text_model)
        agent = HangingAgent(history=orphan_history())
        monkeypatch.setattr(bridge.api, "get_agent", lambda: agent)

        responses = []
        monkeypatch.setattr(bridge, "respond",
                             lambda rid, ok, payload: responses.append((rid, ok, payload)))

        session = new_session()
        params = {"sessionId": session["id"], "message": "开始长任务", "attachments": []}
        worker = threading.Thread(target=bridge._handle_send_message, args=(1, params))
        worker.start()
        try:
            assert wait_for(lambda: session["id"] in bridge._active_runs), \
                "任务应注册进 _active_runs"
            # 必须等任务真正跑到 astream（发送前的修复已执行）再取消：
            # 否则取消可能落在协程启动之前，修复根本没机会跑，断言就成了竞态。
            assert wait_for(lambda: agent.stream_calls > 0), "任务应已开始流式执行"
            assert bridge.dispatch("cancel", {"sessionId": session["id"]})["cancelled"] is True
            worker.join(timeout=5)
            assert not worker.is_alive()                  # 不能挂起
            assert responses and responses[-1][1] is True  # 以成功响应收尾
            assert responses[-1][2]["result"]["cancelled"] is True
            # 开跑前的修复已清掉孤儿 tool_calls（cancel 分支还会再兜一次）
            assert agent.dropped and "ai-1" in agent.dropped[0]
        finally:
            if worker.is_alive():
                with bridge._active_runs_lock:
                    bridge._active_runs.pop(session["id"], None)

    def test_cancel_idle_does_not_repair(self, monkeypatch):
        monkeypatch.setattr(bridge.api, "current_model", text_model)
        agent = RepairAgent(history=orphan_history())
        monkeypatch.setattr(bridge.api, "get_agent", lambda: agent)

        assert bridge.dispatch("cancel", {"sessionId": "idle-sess"})["cancelled"] is False
        time.sleep(0.2)
        assert agent.dropped == []                    # 空闲：不碰历史
