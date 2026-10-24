"""
tests/test_token_tracker.py —— Token 统计与预估模块的单元测试
覆盖：文本估算 / 消息抽取 / 成本折算 / 事前预估 / 事后差分统计 /
      估算兜底 / 会话累计 / ModelSpec.pricing 解析
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import token_tracker as tt
from agent.models import ModelSpec


# ---------------------------------------------------------------------------
# 文本 / 消息估算
# ---------------------------------------------------------------------------
class TestEstimateTokens:
    def test_empty(self):
        assert tt.estimate_tokens("") == 0

    def test_english(self):
        # 11 个 ASCII 字符 × 0.3 = 3.3 → 3
        assert tt.estimate_tokens("hello world") == 3

    def test_chinese(self):
        # 4 个中文字符 × 0.6 = 2.4 → 2
        assert tt.estimate_tokens("你好世界") == 2

    def test_mixed(self):
        assert tt.estimate_tokens("abc你好") == int(3 * 0.3 + 2 * 0.6)

    def test_monotonic(self):
        assert tt.estimate_tokens("a") <= tt.estimate_tokens("a" * 100)


class TestMessageText:
    def test_plain(self):
        msg = HumanMessage(content="你好")
        assert "你好" in tt.message_text(msg)

    def test_tool_calls_counted(self):
        msg = AIMessage(content="", tool_calls=[
            {"name": "read_file", "args": {"path": "x.py"}, "id": "c1"}
        ])
        text = tt.message_text(msg)
        assert "read_file" in text and "x.py" in text

    def test_block_content(self):
        msg = AIMessage(content=[{"type": "text", "text": "块文本"}])
        assert "块文本" in tt.message_text(msg)


def test_messages_tokens_includes_overhead():
    msgs = [HumanMessage(content="你好"), AIMessage(content="好的")]
    one = tt.messages_tokens(msgs[:1])
    two = tt.messages_tokens(msgs)
    assert one > 0 and two > one
    assert two - one >= tt.PER_MESSAGE_OVERHEAD


# ---------------------------------------------------------------------------
# 成本折算
# ---------------------------------------------------------------------------
class TestCost:
    PRICING = {"input_per_mtok": 2.0, "output_per_mtok": 3.0}

    def test_no_pricing(self):
        assert tt.estimate_cost(None, 100, 100) is None

    def test_zero_tokens(self):
        assert tt.estimate_cost(self.PRICING, 0, 0) is None

    def test_normal(self):
        cost = tt.estimate_cost(self.PRICING, 1_000_000, 1_000_000)
        assert cost == pytest.approx(5.0)

    def test_free_model(self):
        assert tt.estimate_cost({"input_per_mtok": 0, "output_per_mtok": 0},
                                100, 100) == 0.0

    def test_fmt(self):
        assert tt.fmt_cost(None) == "价格未配置"
        assert tt.fmt_cost(0) == "¥0"
        assert "¥" in tt.fmt_cost(0.001234)


# ---------------------------------------------------------------------------
# 事前 / 事后（用假 Agent 模拟 checkpointer 状态）
# ---------------------------------------------------------------------------
class FakeSnapshot:
    def __init__(self, messages):
        self.values = {"messages": messages}


class FakeAgent:
    """只实现 aget_state 的最小 Agent 替身。"""

    def __init__(self, messages=None):
        self.messages = list(messages or [])

    async def aget_state(self, config):
        return FakeSnapshot(self.messages)


def test_estimate_run_first_turn():
    agent = FakeAgent()                     # 无历史
    est = asyncio.run(tt.estimate_run(
        agent, "t1", "帮我写个函数",
        pricing={"input_per_mtok": 2.0, "output_per_mtok": 3.0},
        model_id="deepseek-chat"))
    assert est.static_tokens > 0            # 系统提示+工具定义是大头
    assert est.history_tokens == 0
    assert est.output_tokens == tt.DEFAULT_OUTPUT_ESTIMATE   # 首轮默认值
    assert est.input_tokens > est.static_tokens
    assert est.cost is not None and est.cost > 0
    assert "预计消耗" in est.describe()


def test_estimate_run_with_history_and_avg():
    agent = FakeAgent([HumanMessage(content="问题", id="m1"),
                       AIMessage(content="回答", id="m2")])
    est = asyncio.run(tt.estimate_run(agent, "t1", "继续", avg_output=500))
    assert est.history_tokens > 0
    assert est.output_tokens == 500


def test_estimate_run_tool_task_scales_with_calls():
    """工具任务（expected_calls>1）：输入预估应按调用次数放大。"""
    agent = FakeAgent()
    single = asyncio.run(tt.estimate_run(agent, "t1", "统计文件"))
    multi = asyncio.run(tt.estimate_run(agent, "t1", "统计文件",
                                        expected_calls=3))
    # 3 次调用 = 首次 + 2 × 1.2 倍首次 = 3.4 倍
    assert multi.expected_calls == 3
    assert multi.input_tokens == pytest.approx(
        single.input_tokens * (1 + 2 * tt.EXTRA_CALL_INPUT_FACTOR), rel=1e-3)
    assert multi.input_tokens > single.input_tokens
    assert "3 次模型调用" in multi.describe()


def test_estimate_run_reasoner_output_scaled():
    """推理模型的思考链计入输出，预估应放大。"""
    agent = FakeAgent()
    chat = asyncio.run(tt.estimate_run(agent, "t1", "hi", avg_output=100,
                                       model_kind="chat"))
    reasoner = asyncio.run(tt.estimate_run(agent, "t1", "hi", avg_output=100,
                                           model_kind="reasoner"))
    assert reasoner.output_tokens == pytest.approx(
        int(100 * tt.REASONER_OUTPUT_FACTOR))
    assert reasoner.output_tokens > chat.output_tokens


def test_actual_usage_marks_cached_calls():
    """部分调用命中缓存（无 usage 回传）时：计入缓存计数，不混入实耗。"""
    before = [HumanMessage(content="重复之前的问题", id="m0")]
    agent = FakeAgent(before)
    ids = asyncio.run(tt.message_id_snapshot(agent, "t1"))
    agent.messages = before + [
        # 这条命中缓存：消息存在但没有 usage
        AIMessage(content="缓存的回答", id="m1"),
        # 这条是真实调用
        AIMessage(content="新回答", id="m2",
                  usage_metadata={"input_tokens": 500, "output_tokens": 40,
                                  "total_tokens": 540}),
    ]
    usage = asyncio.run(tt.actual_usage(agent, "t1", ids))
    assert usage.estimated is False
    assert usage.calls == 2
    assert usage.cached_calls == 1
    assert usage.input_tokens == 500       # 缓存那条不计入实耗
    assert "缓存命中 1 次" in usage.describe(tt.SessionUsage())


def test_turn_usage_deviation():
    est = tt.RunEstimate(input_tokens=1000, output_tokens=200,
                         total_tokens=1200, history_tokens=0,
                         static_tokens=800, cost=None)
    over = tt.TurnUsage(input_tokens=1500, output_tokens=300)
    under = tt.TurnUsage(input_tokens=800, output_tokens=100)
    same = tt.TurnUsage(input_tokens=1100, output_tokens=100)
    assert "多 50%" in over.deviation(est)
    assert "少" in under.deviation(est)
    assert "基本一致" in same.deviation(est)
    assert over.deviation(None) == ""


def test_actual_usage_diff():
    # 开跑前：只有一条历史
    before_msgs = [HumanMessage(content="数一下文件", id="m0")]
    agent = FakeAgent(before_msgs)
    before_ids = asyncio.run(tt.message_id_snapshot(agent, "t1"))
    assert before_ids == {"m0"}

    # 跑完后：新增 一次工具循环（AI调用 → 工具结果 → AI答案）
    agent.messages = before_msgs + [
        AIMessage(content="", id="m1", tool_calls=[
            {"name": "run_command", "args": {"cmd": "ls"}, "id": "c1"}],
            usage_metadata={"input_tokens": 800, "output_tokens": 30,
                            "total_tokens": 830}),
        ToolMessage(content="a.py b.py", name="run_command",
                    tool_call_id="c1", id="m2"),
        AIMessage(content="共 2 个文件", id="m3",
                  usage_metadata={"input_tokens": 900, "output_tokens": 50,
                                  "total_tokens": 950}),
    ]
    usage = asyncio.run(tt.actual_usage(
        agent, "t1", before_ids,
        pricing={"input_per_mtok": 2.0, "output_per_mtok": 3.0},
        model_id="deepseek-chat"))

    assert usage is not None
    assert usage.estimated is False
    assert usage.calls == 2                 # ReAct 两步都计入了
    assert usage.input_tokens == 1700
    assert usage.output_tokens == 80
    assert usage.total_tokens == 1780
    assert usage.cost == pytest.approx(1700 / 1e6 * 2 + 80 / 1e6 * 3)


def test_actual_usage_no_new_messages():
    agent = FakeAgent([HumanMessage(content="hi", id="m0")])
    ids = asyncio.run(tt.message_id_snapshot(agent, "t1"))
    usage = asyncio.run(tt.actual_usage(agent, "t1", ids))
    assert usage is None                    # 无新增（如中断）→ 不显示


def test_actual_usage_estimated_fallback():
    """供应商不回传 usage_metadata 时退化为估算并如实标注。"""
    before = [HumanMessage(content="hi", id="m0")]
    agent = FakeAgent(before)
    ids = asyncio.run(tt.message_id_snapshot(agent, "t1"))
    agent.messages = before + [AIMessage(content="这是回答", id="m1")]  # 无 usage
    usage = asyncio.run(tt.actual_usage(agent, "t1", ids))
    assert usage is not None
    assert usage.estimated is True
    assert usage.output_tokens > 0
    assert "估算" in usage.describe(tt.SessionUsage())


def test_agent_state_unreadable():
    class BrokenAgent:
        async def aget_state(self, config):
            raise RuntimeError("boom")

    ids = asyncio.run(tt.message_id_snapshot(BrokenAgent(), "t1"))
    assert ids == set()
    usage = asyncio.run(tt.actual_usage(BrokenAgent(), "t1", set()))
    assert usage is None


# ---------------------------------------------------------------------------
# 会话累计
# ---------------------------------------------------------------------------
class TestSessionUsage:
    def test_add_and_avg(self):
        s = tt.SessionUsage()
        s.add(tt.TurnUsage(input_tokens=1000, output_tokens=100))
        s.add(tt.TurnUsage(input_tokens=2000, output_tokens=300))
        assert s.turns == 2
        assert s.total_tokens == 3400
        assert s.avg_output == 200
        assert "累计" in s.describe()

    def test_cost_accumulates(self):
        s = tt.SessionUsage()
        s.add(tt.TurnUsage(input_tokens=1_000_000, output_tokens=0,
                           cost=2.0))
        s.add(tt.TurnUsage(input_tokens=0, output_tokens=1_000_000,
                           cost=3.0))
        assert s.cost == pytest.approx(5.0)

    def test_reset(self):
        s = tt.SessionUsage()
        s.add(tt.TurnUsage(input_tokens=10, output_tokens=10))
        s.reset()
        assert s.turns == 0 and s.total_tokens == 0
        assert s.avg_output == tt.DEFAULT_OUTPUT_ESTIMATE

    def test_empty_avg_default(self):
        assert tt.SessionUsage().avg_output == tt.DEFAULT_OUTPUT_ESTIMATE

    def test_per_model_breakdown(self):
        """中途切换模型后，累计应按模型分开。"""
        s = tt.SessionUsage()
        s.add(tt.TurnUsage(input_tokens=1000, output_tokens=100,
                           model_id="deepseek-chat", cost=0.001))
        s.add(tt.TurnUsage(input_tokens=2000, output_tokens=200,
                           model_id="deepseek-chat", cost=0.002))
        s.add(tt.TurnUsage(input_tokens=500, output_tokens=50,
                           model_id="deepseek-reasoner", cost=0.003))
        rows = dict(s.per_model())
        assert set(rows) == {"deepseek-chat", "deepseek-reasoner"}
        assert rows["deepseek-chat"]["turns"] == 2
        assert rows["deepseek-chat"]["input"] == 3000
        # 多模型时应展开分模型明细
        assert "分模型" in s.describe()
        assert "deepseek-reasoner" in s.describe()

    def test_single_model_no_breakdown_noise(self):
        s = tt.SessionUsage()
        s.add(tt.TurnUsage(input_tokens=100, output_tokens=10,
                           model_id="m"))
        assert "分模型" not in s.describe()


# ---------------------------------------------------------------------------
# pricing 配置解析
# ---------------------------------------------------------------------------
class TestPricingConfig:
    def test_spec_with_pricing(self):
        spec = ModelSpec.from_dict({
            "id": "x", "label": "X", "provider": "deepseek",
            "model": "x", "pricing": {"input_per_mtok": 0.8,
                                      "output_per_mtok": 2.0},
        })
        assert spec.pricing["input_per_mtok"] == 0.8
        pub = spec.to_public_dict(available=True, current=True)
        assert pub["pricing"]["output_per_mtok"] == 2.0

    def test_spec_without_pricing(self):
        spec = ModelSpec.from_dict({"id": "x", "label": "X",
                                    "provider": "deepseek", "model": "x"})
        assert spec.pricing == {}
        assert spec.to_public_dict(True, True)["pricing"] == {}
