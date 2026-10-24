"""
tests/test_ui.py —— 过程框 UI 的单元测试
=========================================
覆盖 agent/ui.py 的纯逻辑部分：
- shorten：框内单行截断
- extract_answer：从消息里提取最终答案
- LineBuffer：固定容量过程行缓冲（滚出旧行）
- TraceLog：完整过程流水落盘
- RunUI：一轮对话的门面流程（非 TTY 回退 + 兜底答案）
"""

from pathlib import Path

from agent.ui import (LineBuffer, RunUI, TraceLog, extract_answer,
                      shorten, strip_thinking)


# ---------------------------------------------------------------------------
# 测试辅助：构造最小消息对象
# ---------------------------------------------------------------------------
def make_msg(msg_type: str, content: str, tool_calls=None, name=None):
    return type("Msg", (), {"type": msg_type, "content": content,
                           "tool_calls": tool_calls, "name": name})()


# ---------------------------------------------------------------------------
# shorten
# ---------------------------------------------------------------------------
class TestShorten:
    def test_short_text_unchanged(self):
        assert shorten("你好", 10) == "你好"

    def test_long_text_truncated_with_ellipsis(self):
        out = shorten("x" * 50, 10)
        assert len(out) == 10
        assert out.endswith("…")

    def test_newline_replaced(self):
        assert "⏎" in shorten("第一行\n第二行", 100)

    def test_tiny_limit_does_not_crash(self):
        out = shorten("abcdefgh", 1)
        assert len(out) >= 1  # 至少保留 1 个可读字符


# ---------------------------------------------------------------------------
# extract_answer
# ---------------------------------------------------------------------------
class TestExtractAnswer:
    def test_final_answer_wins(self):
        msgs = [
            make_msg("ai", "让我先查一下", tool_calls=[{"name": "search"}]),
            make_msg("tool", "找到 3 条结果", name="search"),
            make_msg("ai", "最终答案是 42"),
        ]
        assert extract_answer(msgs) == "最终答案是 42"

    def test_skips_non_ai_messages(self):
        msgs = [make_msg("human", "你好"), make_msg("tool", "ok")]
        assert extract_answer(msgs) == ""

    def test_strips_inner_monologue_before_result_mark(self):
        msgs = [
            make_msg("ai", "让我先查一下", tool_calls=[{"name": "search"}]),
            make_msg("tool", "找到 3 条结果", name="search"),
            make_msg("ai", "【思考】用户要最终答案了。\n\n现在总结交付。【结果】最终答案是 42"),
        ]
        assert extract_answer(msgs) == "最终答案是 42"

    def test_keeps_last_result_mark_only(self):
        msgs = [make_msg("ai", "【结果】第一版\n\n【思考】再想一下\n\n【结果】第二版")]
        assert extract_answer(msgs) == "第二版"

    def test_plain_answer_without_mark_unchanged(self):
        msgs = [make_msg("ai", "  你好，随时帮你  ")]
        assert extract_answer(msgs) == "你好，随时帮你"


# ---------------------------------------------------------------------------
# strip_thinking
# ---------------------------------------------------------------------------
class TestStripThinking:
    def test_removes_thinking_prefix(self):
        assert strip_thinking("【思考】想很多【结果】正式答案") == "正式答案"

    def test_no_mark_returns_trimmed(self):
        assert strip_thinking("  普通答案  ") == "普通答案"

    def test_empty_tail_returns_empty(self):
        assert strip_thinking("【思考】只有思考【结果】") == ""

    def test_non_string_passthrough(self):
        assert strip_thinking(None) is None

    def test_all_have_tool_calls_falls_back_to_last(self):
        msgs = [
            make_msg("ai", "第一步", tool_calls=[{"name": "a"}]),
            make_msg("ai", "第二步", tool_calls=[{"name": "b"}]),
        ]
        assert extract_answer(msgs) == "第二步"

    def test_empty_input(self):
        assert extract_answer([]) == ""
        assert extract_answer(None) == ""

    def test_blank_content_ignored(self):
        msgs = [make_msg("ai", "   "), make_msg("ai", "真答案")]
        assert extract_answer(msgs) == "真答案"


# ---------------------------------------------------------------------------
# LineBuffer
# ---------------------------------------------------------------------------
class TestLineBuffer:
    def test_keeps_only_last_n_lines(self):
        buf = LineBuffer(5)
        for i in range(8):
            buf.push(f"line-{i}")
        assert buf.snapshot() == [f"line-{i}" for i in range(3, 8)]

    def test_total_counts_rolled_out_lines(self):
        buf = LineBuffer(3)
        for i in range(10):
            buf.push(f"line-{i}")
        assert buf.total == 10
        assert buf.overflowed is True

    def test_not_overflowed_when_fits(self):
        buf = LineBuffer(5)
        buf.push("a")
        buf.push("b")
        assert buf.overflowed is False
        assert buf.total == 2

    def test_empty_and_blank_lines_ignored(self):
        buf = LineBuffer(5)
        buf.push("")
        buf.push("   ")
        assert buf.total == 0
        assert buf.snapshot() == []

    def test_min_lines_floor(self):
        assert LineBuffer(1).max_lines == 3


# ---------------------------------------------------------------------------
# TraceLog
# ---------------------------------------------------------------------------
class TestTraceLog:
    def test_full_turn_written(self, tmp_path: Path):
        trace = TraceLog(tmp_path / "trace.log")
        trace.turn("session-001", "你好")
        trace.line("[思考] 让我查一下")
        trace.tool_call("search", {"query": "test"})
        trace.tool_result("search", True, "找到 3 条")
        trace.finish("最终答案")

        text = (tmp_path / "trace.log").read_text(encoding="utf-8")
        assert "session-001" in text
        assert "[用户] 你好" in text
        assert "[思考] 让我查一下" in text
        assert "⚙ 调用工具 search" in text
        assert '"query": "test"' in text
        assert "✔ 工具 search 返回" in text
        assert "[结果] 最终答案" in text

    def test_failed_tool_marked(self, tmp_path: Path):
        trace = TraceLog(tmp_path / "trace.log")
        trace.tool_result("run_code", False, "SyntaxError")
        text = (tmp_path / "trace.log").read_text(encoding="utf-8")
        assert "✖ 工具 run_code 失败" in text

    def test_unwritable_path_warns_once_no_crash(self, tmp_path: Path):
        # 指向一个"目录当文件写"的非法路径，必须只警告一次且不抛异常
        bad = TraceLog(tmp_path)  # tmp_path 是目录，open 会失败
        bad.line("第一条")   # 触发警告
        bad.line("第二条")   # 不应再次警告/抛异常

    def test_non_json_args_fallback(self, tmp_path: Path):
        trace = TraceLog(tmp_path / "trace.log")
        trace.tool_call("t", {"不可序列化": object()})
        text = (tmp_path / "trace.log").read_text(encoding="utf-8")
        assert "⚙ 调用工具 t" in text


# ---------------------------------------------------------------------------
# RunUI（非 TTY 回退路径 + 兜底答案 + trace 联动）
# ---------------------------------------------------------------------------
class TestRunUI:
    def _run(self, tmp_path: Path, monkeypatch, answer: str) -> str:
        from agent.config import settings
        trace_path = tmp_path / "trace.log"
        monkeypatch.setattr(settings, "trace_file", str(trace_path))

        ui = RunUI("session-001", "1+1=?", max_lines=3)
        ui.model_text("让我算一下\n")
        ui.tool_call("calc", {"expr": "1+1"})
        ui.tool_result("calc", True, "2")
        ui.model_text("答案是 2")
        ui.finish(answer)
        ui.close()   # 幂等：重复 close 不能报错
        return trace_path.read_text(encoding="utf-8")

    def test_explicit_answer_used(self, tmp_path, monkeypatch, capsys):
        text = self._run(tmp_path, monkeypatch, answer="标准答案：2")
        assert "[结果] 标准答案：2" in text
        # 终端输出里有答案面板的标题
        assert "结果" in capsys.readouterr().out

    def test_fallback_answer_when_empty(self, tmp_path, monkeypatch):
        text = self._run(tmp_path, monkeypatch, answer="")
        # 没有显式答案 → 用工具返回之后的最后一段模型文本
        assert "[结果] 答案是 2" in text

    def test_tool_events_in_trace(self, tmp_path, monkeypatch):
        text = self._run(tmp_path, monkeypatch, answer="x")
        assert "⚙ 调用工具 calc" in text
        assert "✔ 工具 calc 返回" in text

    def test_fail_writes_error(self, tmp_path, monkeypatch):
        from agent.config import settings
        trace_path = tmp_path / "trace.log"
        monkeypatch.setattr(settings, "trace_file", str(trace_path))
        ui = RunUI("s", "问题")
        ui.fail("[中断] 测试")
        ui.close()
        assert "[异常] [中断] 测试" in trace_path.read_text(encoding="utf-8")
