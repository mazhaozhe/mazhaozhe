"""
tests/test_context_management.py —— 上下文压缩机制的单元测试
覆盖：
    - compute_summary_trigger  预算式触发点计算（窗口 - 输出 - 静态 - 余量）
    - is_summary_message       摘要消息识别
    - detect_compaction        一轮对话后的压缩检测（基于 FakeAgent 状态差分）
    - ModelSpec 窗口字段       目录字段解析 + 不透漏进 extra
    - _window_kwargs           窗口/输出上限 → 模型构造参数
    - core._summarize_trigger  真实装配函数的触发点选择（有窗口 / 无窗口兜底）
"""

import asyncio

import pytest
from langchain_core.messages import HumanMessage

from agent import token_tracker as tt
from agent.config import settings
from agent.models import ModelSpec, _window_kwargs


# ---------------------------------------------------------------------------
# 预算式触发点
# ---------------------------------------------------------------------------
class TestComputeSummaryTrigger:
    def test_basic_budget(self):
        # 131072 - 8192(输出) - 5000(静态) - 8192(余量) = 109688
        assert tt.compute_summary_trigger(131072, 8192, 5000, 8192) == 109688

    def test_unknown_output_uses_cap(self):
        # 没声明输出上限 → 按 UNKNOWN_OUTPUT_CAP(8192) 估算
        with_cap = tt.compute_summary_trigger(131072, None, 0, 0)
        assert with_cap == 131072 - tt.UNKNOWN_OUTPUT_CAP

    def test_bigger_output_lowers_trigger(self):
        """输出上限越大，触发点越低（给长输出留位子）。"""
        small = tt.compute_summary_trigger(131072, 8192, 0, 0)
        big = tt.compute_summary_trigger(131072, 32768, 0, 0)
        assert big < small
        assert small - big == 32768 - 8192

    def test_floor_when_budget_exhausted(self):
        """极端小窗口：预算被余量吃光时，保留一个正触发点（不顶破窗口）。"""
        trigger = tt.compute_summary_trigger(8192, 4096, 5000, 8192)
        assert trigger == 1024

    def test_no_window_returns_none(self):
        assert tt.compute_summary_trigger(None) is None
        assert tt.compute_summary_trigger(0) is None
        assert tt.compute_summary_trigger(-1) is None
        assert tt.compute_summary_trigger("128k") is None   # 非 int 不猜

    def test_never_exceeds_window_minus_output(self):
        """任何配置下，触发点 + 输出上限 + 余量都不得越过窗口。"""
        for max_input, max_output in [(131072, 8192), (131072, 32768),
                                      (32768, None), (200000, 16384)]:
            trigger = tt.compute_summary_trigger(max_input, max_output,
                                                 6000, 8192)
            assert trigger + (max_output or tt.UNKNOWN_OUTPUT_CAP) \
                + 6000 + 8192 <= max_input


# ---------------------------------------------------------------------------
# 摘要消息识别 + 压缩检测（FakeAgent 模拟 checkpointer 状态）
# ---------------------------------------------------------------------------
def _summary_msg(msg_id: str = "s1") -> HumanMessage:
    """构造一条与 SummarizationMiddleware 输出同标记的摘要消息。"""
    return HumanMessage(
        content="Here is a summary of the conversation to date:\n\n## SESSION INTENT ...",
        additional_kwargs={"lc_source": "summarization"},
        id=msg_id,
    )


class TestIsSummaryMessage:
    def test_summary_detected(self):
        assert tt.is_summary_message(_summary_msg())

    def test_normal_messages_not(self):
        assert not tt.is_summary_message(HumanMessage(content="你好", id="m1"))
        assert not tt.is_summary_message(
            HumanMessage(content="普通消息",
                         additional_kwargs={"lc_source": "other"}, id="m2"))


class FakeSnapshot:
    def __init__(self, messages):
        self.values = {"messages": messages}


class FakeAgent:
    """只实现 aget_state 的最小 Agent 替身。"""

    def __init__(self, messages=None):
        self.messages = list(messages or [])

    async def aget_state(self, config):
        return FakeSnapshot(self.messages)


class TestDetectCompaction:
    def test_no_compaction(self):
        agent = FakeAgent([HumanMessage(content="hi", id="m0")])
        ids = asyncio.run(tt.message_id_snapshot(agent, "t1"))
        assert asyncio.run(tt.detect_compaction(agent, "t1", ids)) == 0

    def test_compaction_detected(self):
        before = [HumanMessage(content="旧任务", id="m0")]
        agent = FakeAgent(before)
        ids = asyncio.run(tt.message_id_snapshot(agent, "t1"))
        # 运行后：旧历史被 RemoveMessage 清掉，换成摘要 + 保留的原文
        agent.messages = [_summary_msg("s1"),
                          HumanMessage(content="最近的对话", id="m1")]
        assert asyncio.run(tt.detect_compaction(agent, "t1", ids)) == 1

    def test_preexisting_summary_not_counted(self):
        """开跑前就存在的摘要（上一轮压的）不能重复计数。"""
        old_summary = _summary_msg("s0")
        agent = FakeAgent([old_summary, HumanMessage(content="hi", id="m1")])
        ids = asyncio.run(tt.message_id_snapshot(agent, "t1"))
        agent.messages = [old_summary, HumanMessage(content="hi", id="m1"),
                          HumanMessage(content="回答", id="m2")]
        assert asyncio.run(tt.detect_compaction(agent, "t1", ids)) == 0

    def test_broken_agent_returns_zero(self):
        class BrokenAgent:
            async def aget_state(self, config):
                raise RuntimeError("boom")

        assert asyncio.run(
            tt.detect_compaction(BrokenAgent(), "t1", set())) == 0


# ---------------------------------------------------------------------------
# ModelSpec 窗口字段
# ---------------------------------------------------------------------------
class TestModelSpecWindow:
    def test_fields_parsed_and_not_leaked_to_extra(self):
        """窗口字段进专属字段，绝不混进 extra（否则会被当成构造参数传给厂商）。"""
        spec = ModelSpec.from_dict({
            "id": "deepseek-chat", "label": "DS", "provider": "deepseek",
            "model": "deepseek-chat",
            "max_input_tokens": 131072, "max_output_tokens": 8192,
        })
        assert spec.max_input_tokens == 131072
        assert spec.max_output_tokens == 8192
        assert "max_input_tokens" not in spec.extra
        assert "max_output_tokens" not in spec.extra

    def test_defaults_none(self):
        spec = ModelSpec.from_dict({"id": "x", "label": "X",
                                    "provider": "deepseek", "model": "x"})
        assert spec.max_input_tokens is None
        assert spec.max_output_tokens is None

    def test_public_dict_exposes_window(self):
        spec = ModelSpec.from_dict({
            "id": "x", "label": "X", "provider": "deepseek", "model": "x",
            "max_input_tokens": 131072, "max_output_tokens": 8192,
        })
        pub = spec.to_public_dict(available=True, current=True)
        assert pub["max_input_tokens"] == 131072
        assert pub["max_output_tokens"] == 8192


class TestWindowKwargs:
    def test_both_declared(self):
        spec = ModelSpec(id="x", label="X", provider="deepseek", model="x",
                         max_input_tokens=131072, max_output_tokens=8192)
        assert _window_kwargs(spec) == {
            "max_tokens": 8192,
            "profile": {"max_input_tokens": 131072},
        }

    def test_only_window(self):
        spec = ModelSpec(id="x", label="X", provider="deepseek", model="x",
                         max_input_tokens=32768)
        assert _window_kwargs(spec) == {"profile": {"max_input_tokens": 32768}}

    def test_nothing_declared(self):
        spec = ModelSpec(id="x", label="X", provider="deepseek", model="x")
        assert _window_kwargs(spec) == {}

    def test_invalid_values_ignored(self):
        spec = ModelSpec(id="x", label="X", provider="deepseek", model="x",
                         max_input_tokens=0, max_output_tokens=-5)
        assert _window_kwargs(spec) == {}


# ---------------------------------------------------------------------------
# 真实装配函数：core._summarize_trigger
# ---------------------------------------------------------------------------
class _FakeModel:
    def __init__(self, profile=None, max_tokens=None):
        self.profile = profile
        self.max_tokens = max_tokens


class TestSummarizeTriggerWiring:
    def test_with_window_profile(self):
        from agent.core import _summarize_trigger
        model = _FakeModel(profile={"max_input_tokens": 131072},
                           max_tokens=8192)
        kind, value = _summarize_trigger(model)
        assert kind == "tokens"
        # 窗口 - 8192 输出 - 静态开销 - 8192 余量 ≤ value < 窗口 - 输出 - 余量
        assert value <= 131072 - 8192 - 8192
        assert value >= 131072 // 2

    def test_reasoner_bigger_output_lower_trigger(self):
        """reasoner 输出上限 32K → 触发点应显著低于 chat 的 8K。"""
        from agent.core import _summarize_trigger
        chat = _summarize_trigger(
            _FakeModel(profile={"max_input_tokens": 131072}, max_tokens=8192))[1]
        reasoner = _summarize_trigger(
            _FakeModel(profile={"max_input_tokens": 131072},
                       max_tokens=32768))[1]
        assert reasoner < chat
        assert chat - reasoner >= 32768 - 8192 - 100  # 静态开销两者相同

    def test_no_profile_falls_back_to_absolute(self):
        from agent.core import _summarize_trigger
        assert _summarize_trigger(_FakeModel()) == (
            "tokens", settings.summarize_trigger_tokens)

    def test_profile_without_window_falls_back(self):
        from agent.core import _summarize_trigger
        model = _FakeModel(profile={"max_output_tokens": 8192}, max_tokens=8192)
        assert _summarize_trigger(model) == (
            "tokens", settings.summarize_trigger_tokens)
