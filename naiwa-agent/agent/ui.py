"""
agent/ui.py —— 过程框 UI
=========================
解决"过程淹没结果"的输出问题：

1. ProcessBox：固定高度的边框，Rich Live 原地刷新。
   思考/工具调用过程全部进框，框大小恒定（默认 10 行内容），
   旧行从框顶"滚出"，框里始终能看到最新进展；
2. TraceLog：完整过程逐行写入 data/trace.log（不丢任何细节），
   /trace 命令可用记事本打开滚动查看；
3. 结果面板：一轮结束后，最终答案以高亮 Panel 打印在框的正下方，
   用户不需要翻屏就能看到答案。

非 TTY（输出重定向到文件/管道）时自动退回纯文本输出，保证日志可读。
"""

import json
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

console = Console()

# ---------------------------------------------------------------------------
# 纯函数（可测）
# ---------------------------------------------------------------------------
def shorten(text: str, limit: int) -> str:
    """
    把单行文本压到 limit 字符以内；超长截断加省略号，换行替换为 ⏎。
    用于框内展示（完整内容不截断，写进 trace.log）。
    """
    text = str(text).replace("\n", " ⏎ ")
    if len(text) <= limit:
        return text
    keep = max(limit - 1, 1)
    return text[:keep] + "…"


RESULT_MARK = "【结果】"


def strip_thinking(text: str) -> str:
    """
    剥掉最终答案里混入的内心独白。

    系统提示词（agent/prompts.py）要求模型用【结果】标注最终答案，
    但模型的最后一条消息常常把【思考】…内心独白…和【结果】正式答案
    写在同一个 content 里。这里只保留最后一个【结果】之后的内容；
    没有标记的文本原样返回（简单对话不强制标标记）。
    """
    if not isinstance(text, str):
        return text
    if RESULT_MARK in text:
        return text.rsplit(RESULT_MARK, 1)[1].strip()
    return text.strip()


def extract_answer(messages) -> str:
    """
    从一批消息里提取"最终答案"：
    最后一条【不带工具调用】的 AI 消息（LangGraph 的模型节点要么返回
    工具调用继续循环，要么返回纯文本结束——后者就是答案）。
    答案文本会经过 strip_thinking 剥掉【结果】之前的内心独白。
    若所有 AI 消息都带工具调用（异常中断），退而取最后一条 AI 消息的文本。
    """
    fallback = ""
    for msg in messages or []:
        if getattr(msg, "type", "") != "ai":
            continue
        content = getattr(msg, "content", "")
        if not (isinstance(content, str) and content.strip()):
            continue
        if getattr(msg, "tool_calls", None):
            fallback = content.strip()      # 只是中间思考，先记着兜底
        else:
            return strip_thinking(content)  # 无工具调用 → 这就是答案
    return strip_thinking(fallback)


class LineBuffer:
    """
    固定容量的过程行缓冲：只保留最近 max_lines 行，其余"滚出"。
    ProcessBox 的纯逻辑内核（不碰终端，可单测）。
    """

    def __init__(self, max_lines: int = 10):
        self.max_lines = max(max_lines, 3)
        self._lines: deque[str] = deque(maxlen=self.max_lines)
        self.total = 0          # 累计 push 过的行数（含滚出的）

    def push(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        self._lines.append(line)
        self.total += 1

    @property
    def overflowed(self) -> bool:
        """是否有内容滚出了框（用于提示用户 /trace 看完整过程）。"""
        return self.total > self.max_lines

    def snapshot(self) -> list[str]:
        return list(self._lines)


# ---------------------------------------------------------------------------
# 过程框
# ---------------------------------------------------------------------------
class ProcessBox:
    """固定高度的边框面板；交互式终端用 Live 原地刷新，非 TTY 退回打印。"""

    def __init__(self, max_lines: int = 10, title: str = "Agent 思考与行动过程"):
        self.buf = LineBuffer(max_lines)
        self.title = title
        self.interactive = sys.stdout.isatty()
        self._live: Live | None = None
        self._pending = ""      # 流式文本中尚未换行的尾巴

    # ---- 生命周期 ----
    def start(self) -> None:
        if self.interactive:
            self._live = Live(console=console, refresh_per_second=8,
                              transient=False)
            self._live.start()
            self._refresh()

    def stop(self) -> None:
        self.flush_pending()
        if self._live is not None:
            self._live.stop()
            self._live = None

    # ---- 内容 ----
    def push(self, line: str) -> None:
        """完整一行过程（工具调用/返回等）。"""
        if not self.interactive:
            print(line, flush=True)
            return
        self.buf.push(line)
        self._refresh()

    def push_text(self, delta: str, style: str = "dim") -> None:
        """
        流式文本片段：拼接到未换行尾巴上，出现换行符才落成完整行。
        框内实时可见（含未换行的尾巴），不刷屏。
        """
        if not self.interactive:
            sys.stdout.write(delta)
            sys.stdout.flush()
            return
        self._pending += delta
        while "\n" in self._pending:
            line, _, self._pending = self._pending.partition("\n")
            self.buf.push(line)
        self._refresh(style=style)

    def flush_pending(self) -> None:
        if not self.interactive:
            return
        if self._pending.strip():
            self.buf.push(self._pending)
        self._pending = ""
        self._refresh()

    # ---- 渲染 ----
    def _render(self, style: str = "dim") -> Panel:
        lines = self.buf.snapshot()
        if self._pending:
            lines = lines + [self._pending]
        body = "\n".join(lines) if lines else "…"
        text = Text(body, style=style, no_wrap=True, overflow="crop")
        return Panel(text, height=self.buf.max_lines + 2, title=self.title,
                     title_align="left", border_style="dim")

    def _refresh(self, style: str = "dim") -> None:
        if self._live is not None:
            self._live.update(self._render(style))


# ---------------------------------------------------------------------------
# 完整过程流水（trace.log）
# ---------------------------------------------------------------------------
class TraceLog:
    """
    把一轮对话的完整过程（思考/推理/工具调用/结果）逐行写入文本文件。
    框里只能看到最近几行，这里是全量记录，/trace 命令打开滚动查看。
    写文件失败不阻断对话（提示一次后静默跳过）。
    """

    def __init__(self, path):
        self.path = Path(path)
        self._warned = False

    def _write(self, text: str) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(text)
        except OSError as exc:
            if not self._warned:
                self._warned = True
                print(f"[warn] 过程日志写入失败（{exc}），本次运行不再记录。")

    def turn(self, thread_id: str, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"\n{'=' * 60}\n"
                    f"时间: {ts}  会话: {thread_id}\n"
                    f"[用户] {message}\n")

    def line(self, text: str) -> None:
        self._write(text + "\n")

    def tool_call(self, name: str, args) -> None:
        try:
            args_str = json.dumps(args, ensure_ascii=False)
        except (TypeError, ValueError):
            args_str = str(args)
        self.line(f"⚙ 调用工具 {name}  参数: {shorten(args_str, 500)}")

    def tool_result(self, name: str, ok: bool, content: str) -> None:
        mark = "✔" if ok else "✖"
        self.line(f"{mark} 工具 {name} "
                  f"{'返回' if ok else '失败'}: {shorten(content, 2000)}")

    def finish(self, answer: str) -> None:
        self.line(f"\n[结果] {answer}\n")

    def error(self, text: str) -> None:
        self.line(f"[异常] {text}\n")


# ---------------------------------------------------------------------------
# RunUI：一轮对话的 UI 门面
# ---------------------------------------------------------------------------
class _Pending:
    """trace 落盘用的文本缓冲：按换行切成行，前缀标记类型（思考/推理）。"""

    def __init__(self, kind: str, trace: TraceLog):
        self.kind = kind
        self.trace = trace
        self.buf = ""

    def add(self, delta: str) -> None:
        self.buf += delta
        while "\n" in self.buf:
            line, _, self.buf = self.buf.partition("\n")
            if line.strip():
                self.trace.line(f"[{self.kind}] {line}")

    def flush(self) -> None:
        if self.buf.strip():
            self.trace.line(f"[{self.kind}] {self.buf}")
        self.buf = ""


class RunUI:
    """
    stream_run 的 UI 门面：过程进框 + trace 落盘 + 结束后打印答案面板。
    用法：
        ui = RunUI(thread_id, message)
        try:
            ui.model_text(chunk) / ui.reasoning_text(chunk)
            ui.tool_call(...) / ui.tool_result(...)
            ui.finish(answer)          # answer 为空时自动用兜底文本
        finally:
            ui.close()
    """

    def __init__(self, thread_id: str, message: str, max_lines: int | None = None):
        from agent.config import settings
        if max_lines is None:
            max_lines = settings.process_box_lines
        self.box = ProcessBox(max_lines)
        self.trace = TraceLog(settings.abs_path(settings.trace_file))
        self.box.start()
        self.trace.turn(thread_id, message)
        self._pend: _Pending | None = None
        self._answer_fallback = ""     # 最近一次工具环节之后的模型文本
        self._closed = False

    # ---- 过程事件 ----
    def model_text(self, delta: str) -> None:
        self._switch_pending("思考")
        self._pend.add(delta)
        self.box.push_text(delta)
        self._answer_fallback += delta

    def reasoning_text(self, delta: str) -> None:
        self._switch_pending("推理")
        self._pend.add(delta)
        self.box.push_text(delta)

    def tool_call(self, name: str, args) -> None:
        self._flush_pending()
        self.box.push(f"⚙ 调用工具 {name} {shorten(str(args), 60)}")
        self.trace.tool_call(name, args)
        self._answer_fallback = ""      # 进入工具环节，之前文本不算答案

    def tool_result(self, name: str, ok: bool, content: str) -> None:
        self._flush_pending()
        mark = "✔" if ok else "✖"
        self.box.push(f"{mark} {name}: {shorten(content, 80)}")
        self.trace.tool_result(name, ok, content)
        self._answer_fallback = ""

    # ---- 结束 ----
    def finish(self, answer: str = "") -> None:
        self._flush_pending()
        self.box.stop()
        answer = (answer or self._answer_fallback or "").strip()
        if self.box.buf.overflowed:
            console.print(f"[dim]（过程共 {self.box.buf.total} 行，"
                          f"框内为最近 {self.box.buf.max_lines} 行）[/dim]")
        if answer:
            console.print(Panel(Text(answer), title="结果",
                                title_align="left", border_style="cyan",
                                padding=(0, 1)))
        else:
            console.print("[yellow]（模型没有返回文本结果，"
                          "可输入 /trace 查看过程细节）[/yellow]")
        console.print("[dim]完整过程已存 data/trace.log，输入 /trace 可滚动查看。[/dim]")
        self.trace.finish(answer)
        self._closed = True

    def fail(self, error_text: str) -> None:
        """异常/中断路径：关掉框，把错误平铺打印并记入 trace。"""
        self._flush_pending()
        self.box.stop()
        console.print(f"[red]{error_text}[/red]")
        self.trace.error(error_text)
        self._closed = True

    def close(self) -> None:
        """finally 兜底：无论正常/异常，框都必须停掉（Live 挂着会卡住终端）。"""
        if not self._closed:
            try:
                self.box.stop()
            except Exception:
                pass
            self._closed = True

    # ---- 内部 ----
    def _switch_pending(self, kind: str) -> None:
        if self._pend is None or self._pend.kind != kind:
            self._flush_pending()
            self._pend = _Pending(kind, self.trace)

    def _flush_pending(self) -> None:
        if self._pend is not None:
            self._pend.flush()
            self._pend = None


# ---------------------------------------------------------------------------
# /trace 命令支持
# ---------------------------------------------------------------------------
def open_trace_file() -> Path:
    """
    打开过程日志：Windows 用默认程序（记事本，可滚动），
    其他平台打印路径。返回日志路径。
    """
    from agent.config import settings
    path = settings.abs_path(settings.trace_file)
    if not path.exists():
        print("  还没有过程记录。先问一个问题，再来 /trace。")
        return path
    print(f"  过程日志: {path}")
    try:
        import os
        import platform
        if platform.system() == "Windows":
            os.startfile(path)  # Windows 专用 API
        print("  已调用系统默认程序打开，可直接滚动查看。")
    except Exception as exc:
        print(f"  自动打开失败（{exc}），请手动打开上面的路径。")
    return path
