"""
tests/test_output_guard.py —— 输出退化（"复读机"）防护测试
==========================================================

三类用例：
1. 检测器本身：真退化必须命中、正常长文必须不误判；
2. 流式护栏：逐块喂入时能在烧光 token 之前中止；
3. 桥接层全链路：退化 → 中止 → 带去重指令重试一轮 → 拿到正常答案；
   重试预算用尽时返回明确提示，而不是把「好。执行。」倒给用户。
"""

import asyncio
import sys
import threading
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, RemoveMessage

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT / "client"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from agent.output_guard import (StreamGuard, detect_degeneration,  # noqa: E402
                                is_degenerate, trim_degeneration)

# 真实退化的骨架：模型决策瘫痪时的「好。执行。我调用。（执行）」
LOOP_UNIT = "好。\n\n执行。\n\n"
LOOP = LOOP_UNIT * 300

# 退化前那一段「犹豫独白」（取自真实会话，未做删改）
DELIBERATION = (
    "【思考】有两处候选：一是用户自己的课程设计（C 语言 + Q-learning），有 exe 和报告；"
    "二是 Agent 项目自带的示例目录，不是我帮他做的。我不确定用户指哪个，而“打开”是启动"
    "程序的操作。最稳妥的做法是先问清楚，同时给出候选。不过用户意图很明确是“打开游戏”，"
    "我可以先确认哪个。考虑到示例目录是项目自带，而另一个是用户自己的课程设计，两者都"
    "可能。按准则，不确定时应说明情况。但“打开”这个动作本身风险低，我倾向于先问一句"
    "确认，因为打开错误的程序没意义。不过更高效的做法是直接列出候选并询问。"
) * 4

NORMAL_DOC = """# 项目说明

## 安装

```bash
python -m pip install -r requirements.txt
```

## 配置

在项目根目录创建 .env 文件，填入 API Key 与模型名。配置项通过 pydantic-settings
读取，键名与字段一一对应，大小写不敏感。

## 目录结构

- agent/：核心逻辑，包含模型构造、提示词、上传与历史修复
- tools/：工具注册表，每个模块提供一个或多个 LangChain Tool
- client/：Electron 客户端与本地的 JSON-Lines 桥接进程
- tests/：pytest 用例，覆盖工具、桥接、上传与历史修复

## 常见问题

Q1：新对话后输入框打不出字？
A1：模型接口挂起时旧版会把 UI 锁死，现在改为并发桥接 + 硬超时 + 停止按钮。

Q2：出现 400 报错说 tool_calls 没被响应？
A2：取消或超时会在历史里留下孤儿 tool_calls，发送前会自动清理。
""" * 3


class TestDetect:
    def test_tight_loop_is_detected_from_its_start(self):
        text = DELIBERATION + LOOP
        at = detect_degeneration(text)
        assert at is not None
        # 退化起点必须落在犹豫独白之后、循环开始附近（允许检测器滞后一小段）
        assert len(DELIBERATION) - 50 <= at <= len(DELIBERATION) + 900

    def test_pure_loop_detected(self):
        assert detect_degeneration(LOOP) == 0

    def test_normal_prose_not_flagged(self):
        assert detect_degeneration(NORMAL_DOC) is None

    def test_real_project_files_not_flagged(self):
        """真实代码/文档一条都不能误判（代码里有大量重复符号）。"""
        for name in ("README.md", "agent/prompts.py", "agent/core.py",
                     "agent/history_repair.py", "tools/search.py"):
            path = ROOT / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            assert detect_degeneration(text) is None, f"{name} 被误判为退化"

    def test_short_text_skipped(self):
        assert detect_degeneration("好。执行。好。执行。") is None

    def test_dash_separator_not_flagged(self):
        """分隔线那种“单字符重复”不算退化（单元里必须有两个以上不同字符）。"""
        text = NORMAL_DOC + "\n" + "-" * 600 + "\n" + NORMAL_DOC
        assert detect_degeneration(text) is None

    def test_trim_removes_loop_tail(self):
        text = DELIBERATION + LOOP
        kept, hit = trim_degeneration(text)
        assert hit is True
        assert len(kept) < len(text)
        assert "执行" not in kept[-200:] or kept.count("执行") < LOOP.count("执行")

    def test_trim_keeps_clean_text(self):
        kept, hit = trim_degeneration(NORMAL_DOC)
        assert hit is False
        assert kept == NORMAL_DOC

    def test_is_degenerate_ignores_short(self):
        assert is_degenerate("好。", min_chars=400) is False


class TestStreamGuard:
    def test_aborts_before_stream_ends(self):
        """逐块喂入：必须在文本喂完之前就判定，否则等于没省 token。"""
        guard = StreamGuard()
        fed = 0
        for i in range(0, len(LOOP), 40):
            fed = i + 40
            if guard.feed(LOOP[:fed]) is not None:
                break
        assert fed < len(LOOP) // 2, "中止得太晚，token 已经烧掉大半"

    def test_stream_guard_keeps_full_text_when_clean(self):
        guard = StreamGuard()
        for i in range(0, len(NORMAL_DOC), 200):
            assert guard.feed(NORMAL_DOC[:i + 200]) is None
        assert guard.text() == NORMAL_DOC

    def test_hit_is_sticky(self):
        guard = StreamGuard()
        assert guard.feed(LOOP[:2000]) is not None
        assert guard.feed(LOOP[:2400]) == guard.hit


# ---------------------------------------------------------------------------
# 桥接层全链路
# ---------------------------------------------------------------------------
import bridge  # noqa: E402
from agent.config import settings  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "SESSION_FILE", tmp_path / "sessions.json")
    monkeypatch.setattr(bridge, "_initialized", True)
    monkeypatch.setattr(settings, "degenerate_guard", True)
    monkeypatch.setattr(settings, "degenerate_retry", 1)
    yield


class FakeState:
    def __init__(self, messages):
        self.values = {"messages": messages}


class DegenerateAgent:
    """
    第一轮吐退化循环（永远吐不完），第二轮吐正常答案。
    用来验证"中止 → 重试"这条链路。
    """

    def __init__(self, degenerate_rounds: int = 1):
        self.degenerate_rounds = degenerate_rounds
        self.stream_calls = 0
        self.inputs = []

    async def aget_state(self, config):
        return FakeState([])

    async def aupdate_state(self, config, values):
        return None

    def astream(self, input_data, config=None, stream_mode=None):
        self.stream_calls += 1
        self.inputs.append(input_data)
        degenerate = self.stream_calls <= self.degenerate_rounds

        async def generator():
            if degenerate:
                for i in range(0, 4000, 60):
                    yield ("messages", (AIMessageChunk(content=LOOP[i:i + 60]), {}))
                    await asyncio.sleep(0)       # 让取消/中断有机会插进来
                return
            yield ("messages", (AIMessageChunk(content="已打开贪吃蛇。"), {}))
            yield ("updates", {"agent": {"messages": [AIMessage(
                content="已打开贪吃蛇。", id="ai-ok",
                usage_metadata={"input_tokens": 10, "output_tokens": 5,
                                "total_tokens": 15})]}})

        return generator()


def test_degenerate_run_aborts_and_retries(monkeypatch):
    """退化 → 中止 → 自动重试一轮 → 拿到正常答案，且过程框只有干净的条目。"""
    agent = DegenerateAgent(degenerate_rounds=1)
    monkeypatch.setattr(bridge.api, "get_agent", lambda: agent)
    monkeypatch.setattr(bridge.api, "current_model", lambda: {
        "id": "deepseek-chat", "kind": "chat", "supports_vision": False, "pricing": {}})
    session = bridge.dispatch("new_session", {})

    result = asyncio.run(
        bridge.send_message({"sessionId": session["id"], "message": "打开贪吃蛇游戏"}))
    assert "已打开贪吃蛇" in result["answer"]
    assert result["answer"].count("执行") == 0
    assert agent.stream_calls == 2, "应重试恰好一次"
    # 重试时带上了纠正指令，而不是原样重发
    assert "不要再权衡" in str(agent.inputs[1])
    kinds = [entry["kind"] for entry in result["process"]]
    assert "notice" in kinds
    assert all(len(entry.get("text", "")) < 400 for entry in result["process"]), \
        "退化产生的大段垃圾不应写进会话"


def test_degenerate_without_retry_returns_clear_message(monkeypatch):
    """重试预算为 0：返回明确提示，不把「好。执行。」当答案给用户。"""
    monkeypatch.setattr(settings, "degenerate_retry", 0)
    agent = DegenerateAgent(degenerate_rounds=99)
    monkeypatch.setattr(bridge.api, "get_agent", lambda: agent)
    monkeypatch.setattr(bridge.api, "current_model", lambda: {
        "id": "deepseek-chat", "kind": "chat", "supports_vision": False, "pricing": {}})
    session = bridge.dispatch("new_session", {})

    result = asyncio.run(
        bridge.send_message({"sessionId": session["id"], "message": "打开贪吃蛇游戏"}))
    assert agent.stream_calls == 1
    assert "重复循环" in result["answer"]
    assert result["answer"].count("执行") == 0


def test_degenerate_guard_can_be_disabled(monkeypatch):
    """关闭护栏：行为回到"整段吐完"，用于排查问题或对比效果。"""
    monkeypatch.setattr(settings, "degenerate_guard", False)
    agent = DegenerateAgent(degenerate_rounds=1)
    monkeypatch.setattr(bridge.api, "get_agent", lambda: agent)
    monkeypatch.setattr(bridge.api, "current_model", lambda: {
        "id": "deepseek-chat", "kind": "chat", "supports_vision": False, "pricing": {}})
    session = bridge.dispatch("new_session", {})

    result = asyncio.run(
        bridge.send_message({"sessionId": session["id"], "message": "打开贪吃蛇游戏"}))
    assert agent.stream_calls == 1
    assert result["answer"].count("执行") > 100


# ---------------------------------------------------------------------------
# 历史清洗：已膨胀的退化消息不能一轮轮喂回模型
# ---------------------------------------------------------------------------
from agent.history_repair import find_degenerate, repair_history  # noqa: E402


class RecordingAgent:
    def __init__(self, history):
        self.history = list(history)
        self.updates = []

    async def aget_state(self, config):
        return FakeState(self.history)

    async def aupdate_state(self, config, values):
        self.updates.append(values["messages"])


def test_find_degenerate_truncates_ai_message():
    msg = AIMessage(content=DELIBERATION + LOOP, id="ai-bad")
    patches = find_degenerate([HumanMessage(content="打开游戏", id="h-1"), msg])
    assert len(patches) == 1
    assert patches[0].id == "ai-bad", "必须原地替换，不能改变消息顺序"
    assert len(patches[0].content) < len(msg.content) / 3


def test_find_degenerate_skips_tool_call_messages():
    """带 tool_calls 的消息不能动，否则会制造孤儿。"""
    msg = AIMessage(content=DELIBERATION + LOOP, id="ai-tools",
                    tool_calls=[{"id": "c-1", "name": "read_file", "args": {}}])
    assert find_degenerate([msg]) == []


def test_find_degenerate_skips_clean_history():
    assert find_degenerate([AIMessage(content=NORMAL_DOC, id="ai-ok")]) == []


def test_repair_history_applies_truncation():
    agent = RecordingAgent([AIMessage(content=DELIBERATION + LOOP, id="ai-bad")])
    changed = asyncio.run(repair_history(agent, "sess-x"))
    assert changed == 1
    assert agent.updates and isinstance(agent.updates[0][0], AIMessage)
    assert len(agent.updates[0][0].content) < 3000


def test_sanitize_sessions_trims_degenerate_bubble(monkeypatch):
    """旧会话里一次上万字符的泡泡：启动自愈只留开头一段 + 说明。"""
    monkeypatch.setattr(bridge.api, "current_model", lambda: {
        "id": "deepseek-chat", "kind": "chat", "supports_vision": False, "pricing": {}})
    session = bridge.dispatch("new_session", {})
    good = bridge.dispatch("new_session", {})
    bridge.update_session(session["id"], lambda s: s["messages"].append({
        "role": "assistant", "content": DELIBERATION + LOOP, "process": [],
    }))
    bridge.update_session(good["id"], lambda s: s["messages"].append({
        "role": "assistant", "content": NORMAL_DOC, "process": [],
    }))

    assert bridge.sanitize_sessions() == 1
    trimmed = bridge.get_session(session["id"])["messages"][-1]["content"]
    assert len(trimmed) < 1600
    assert "重复循环" in trimmed
    # 正常回复一个字都不许动
    assert bridge.get_session(good["id"])["messages"][-1]["content"] == NORMAL_DOC
