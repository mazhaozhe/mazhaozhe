"""
tools/coding.py —— 代码开发类工具
==================================
文件读写 / 精确编辑 / 目录浏览 / Shell 执行。
错误处理策略：可重试的异常（文件不存在、命令超时等）直接 raise，
交给 ToolRetryMiddleware / ToolErrorMiddleware 处理；
不可恢复的情况（文件过大）返回说明文本让模型换方案。

run_shell 必须用异步子进程实现（2026-10 修复「到这个工具就卡住」）：
同步的 subprocess.run 会被 LangGraph 丢进线程池执行，用户点「停止」
时协程取消杀不掉线程里的子进程——pip install 之类会顶着 timeout
继续在后台跑满数分钟，Windows 上 shell=True 的管道孙进程还会让
communicate() 干等管道 EOF，表现为"卡住"。异步实现下：
    - 事件循环不被阻塞，等待期间新对话/取消/上传都能即时处理；
    - 取消（CancelledError）与超时都会杀掉整棵进程树（含 cmd/bash
      的子孙进程），不留孤儿；
    - Windows 上先 chcp 65001 + PYTHONUTF8=1，解决中文输出乱码。
"""

import asyncio
import os
import re
import signal
import sys
from pathlib import Path

from langchain.tools import tool
from pydantic import BaseModel, Field

from tools.registry import register

# 最大可读取文件大小（2MB）
MAX_READ_SIZE = 2 * 1024 * 1024

# 危险命令黑名单（安全阀：高危操作必须拦下）
_DANGEROUS_PATTERNS = (
    "rm -rf /",
    "format c:",
    "del /f /s /q c:\\",
    "rd /s /q c:\\",
    "shutdown",
    "mkfs",
    ":(){:|:&};:",
)

# 凭据赋值掩码：字段名含 key/secret/token/password 等，且值带引号、长度 ≥4。
# 模型看得到字段名、看不到值——这是防泄露兜底，配合 prompt 里的凭据卫生准则。
_SECRET_RE = re.compile(
    r"(?i)([A-Za-z_]*(?:api[_-]?key|secret|token|password|passwd|credential)"
    r"[A-Za-z_]*)(\s*[:=]\s*)([\"'])([^\"']{4,})\3"
)


def _mask_secrets(text: str) -> str:
    """把 key = "value" 形式的凭据值替换为 ***。"""
    return _SECRET_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}***{m.group(3)}", text
    )


# ---------------------------------------------------------------------------
# 参数 Schema（Pydantic 强校验：参数不对在执行前就会被拦截）
# ---------------------------------------------------------------------------
class ReadFileArgs(BaseModel):
    path: str = Field(..., description="文件的绝对路径")
    encoding: str = Field("utf-8", description="文件编码，默认 utf-8")
    start_line: int = Field(1, ge=1, description="起始行（含），用于分段精读大文件")
    end_line: int = Field(0, ge=0, description="结束行（含），0 表示读到文件末尾")
    pattern: str = Field(
        "",
        description=(
            "正则表达式；提供时只返回命中的行（带行号），"
            "适合在大文件里定位/计数，不要把大文件整段读进上下文"
        ),
    )
    max_hits: int = Field(200, ge=1, le=1000, description="pattern 模式最多返回的命中行数")


class WriteFileArgs(BaseModel):
    path: str = Field(..., description="目标文件绝对路径（父目录会自动创建）")
    content: str = Field(..., description="要写入的完整内容（覆盖式）")


class EditFileArgs(BaseModel):
    path: str = Field(..., description="文件绝对路径")
    old_text: str = Field(..., description="要被替换的原文（必须精确匹配）")
    new_text: str = Field(..., description="替换后的新文本")


class ListDirArgs(BaseModel):
    path: str = Field(".", description="目录路径，默认当前目录")
    pattern: str = Field("*", description="通配符过滤，如 *.py")


class RunShellArgs(BaseModel):
    command: str = Field(..., description="要执行的命令，如 python -m pytest -q")
    timeout: int = Field(60, ge=1, le=600, description="超时秒数，默认 60")


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------
@register
@tool(args_schema=ReadFileArgs)
def read_file(
    path: str,
    encoding: str = "utf-8",
    start_line: int = 1,
    end_line: int = 0,
    pattern: str = "",
    max_hits: int = 200,
) -> str:
    """读取文本文件。大文件请按下面两种方式按需读取，不要整段读进上下文：
    - 定位/计数：传 pattern 正则，只返回命中行（带行号），如 pattern="ERROR"；
    - 分段精读：传 start_line / end_line 取行区间。
    小文件直接整读（默认行为）。含 key/secret/token/password 的赋值会被自动掩码。
    """
    p = Path(path).expanduser()
    if not p.exists():
        # raise → 中间件捕获并反馈，模型可修正路径后重试
        raise FileNotFoundError(f"文件不存在: {p}")
    if not p.is_file():
        raise IsADirectoryError(f"路径是目录而非文件: {p}")
    size = p.stat().st_size
    if size > MAX_READ_SIZE and not pattern:
        # 返回说明而不是 raise：换读法是模型自己能做的决定
        return (
            f"文件过大（{size // 1024} KB），不要整读。"
            "请先用 pattern 参数定位（如 ERROR、Traceback、def ），"
            "再用 start_line/end_line 精读需要的行区间。"
        )
    try:
        text = p.read_text(encoding=encoding)
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"编码错误：{p.name} 无法用 {encoding} 解码"
            "（可能是二进制文件或非 UTF-8 编码，可换 encoding 参数重试）"
        ) from exc
    if not text.strip():
        return "(空文件)"

    lines = text.splitlines()

    # 模式一：正则定位（grep 思路，大文件首选）
    if pattern:
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"正则表达式无效: {exc}") from exc
        hits = [(i, ln) for i, ln in enumerate(lines, 1) if rx.search(ln)]
        head = f"共 {len(lines)} 行，命中 {len(hits)} 行（pattern: {pattern}）"
        if len(hits) > max_hits:
            head += f"，仅显示前 {max_hits} 行"
        body = "\n".join(f"{i}: {_mask_secrets(ln)}" for i, ln in hits[:max_hits])
        return head + ("\n" + body if body else "")

    # 模式二：整读 / 行区间
    total = len(lines)
    if end_line == 0 or end_line > total:
        end_line = total
    if start_line > total:
        return f"文件共 {total} 行，起始行 {start_line} 超出范围。"
    seg = lines[start_line - 1 : end_line]
    if start_line == 1 and end_line == total:
        return _mask_secrets("\n".join(seg))
    return _mask_secrets(
        f"（第 {start_line}-{end_line} 行 / 共 {total} 行）\n" + "\n".join(seg)
    )


@register
@tool(args_schema=WriteFileArgs)
def write_file(path: str, content: str) -> str:
    """把内容写入指定文件（覆盖式），父目录不存在会自动创建。"""
    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise IOError(f"写入失败: {exc}") from exc
    return f"已写入 {p}（{len(content)} 字符）"


@register
@tool(args_schema=EditFileArgs)
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """精确替换文件中的一段文本（old_text 必须与文件内容完全一致）。"""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    text = p.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count == 0:
        raise ValueError(
            f"未在 {p.name} 中找到要替换的原文。"
            "请先用 read_file 核对原文内容（注意缩进与空格）。"
        )
    if count > 1:
        return (
            f"原文在 {p.name} 中出现 {count} 次，为避免误改已中止。"
            "请提供更长、更唯一的原文片段。"
        )
    p.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    return f"已替换 {p.name} 中的 1 处文本。"


@register
@tool(args_schema=ListDirArgs)
def list_dir(path: str = ".", pattern: str = "*") -> str:
    """列出目录下匹配通配符的文件/子目录（浏览项目结构用）。"""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"目录不存在: {p}")
    entries = sorted(p.glob(pattern))
    if not entries:
        return f"{p} 下没有匹配 `{pattern}` 的条目。"
    lines = []
    for e in entries[:200]:  # 上限 200 条，防止刷屏
        tag = "目录" if e.is_dir() else f"{e.stat().st_size} B"
        lines.append(f"{e.name}  ({tag})")
    return "\n".join(lines)


# 项目根目录（tools/coding.py → 上两级）：用于定位项目虚拟环境
_BASE_DIR = Path(__file__).resolve().parent.parent


def _shell_env() -> dict[str, str]:
    """
    构造 Shell 命令的执行环境。

    核心是**把项目 .venv 放到 PATH 最前**：否则 `python` / `pip` 会解析到
    系统 Python（甚至 Microsoft Store 的存根，执行时会弹窗挂住），模型
    `pip install` 装进别的环境——Agent 自己（跑在 .venv 里）永远用不上，
    于是它反复装、反复失败，用户看到的就是"每次到这个工具就卡住"。
    """
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    bin_dir = _BASE_DIR / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    if bin_dir.is_dir():
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(bin_dir.parent)
        env.pop("PYTHONHOME", None)   # 别处继承来的 PYTHONHOME 会弄乱 .venv
    return env


@register
@tool(args_schema=RunShellArgs)
async def run_shell(command: str, timeout: int = 60) -> str:
    """执行 Shell 命令（跑测试/装依赖/看版本等），超时或取消时连同子进程树一并终止。"""
    lowered = command.lower()
    for pattern in _DANGEROUS_PATTERNS:
        if pattern in lowered:
            # 高危命令：不执行，直接报错让模型放弃该方案
            raise PermissionError(f"命令被安全策略拦截：包含危险操作 `{pattern}`")

    # Windows 控制台默认 GBK 代码页：切 UTF-8 + 强制 Python 输出 UTF-8，
    # 否则中文 stderr（如「'tail' 不是内部或外部命令」）会变乱码，模型读不懂
    env = _shell_env()
    if sys.platform == "win32":
        command = f"chcp 65001 >nul && {command}"

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,   # POSIX：独立进程组，取消时整组杀
        )
    except Exception as exc:
        raise RuntimeError(f"命令启动失败：{type(exc).__name__}: {exc}") from exc

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        await _kill_process_tree(proc)
        # 关键：超时**不能抛异常**。抛 TimeoutError 会被 ToolRetryMiddleware
        # 当成瞬时故障重试 3 次（默认策略重试所有异常），再叠加模型看到失败
        # 后的再次尝试——一条 timeout=240 的命令会被放大成几十分钟的等待
        # （实测单次 1 秒的命令能跑出 16 次执行、40 秒总耗时）。
        # 这里返回结构化结果，让模型看到"超时"并主动换方案。
        return (f"exit_code=timeout\n[stdout]\n\n[stderr]\n"
                f"命令执行超过 {timeout} 秒，已被强制终止（连同全部子进程）。\n"
                f"这不是命令语法错误，而是时间上限。请换更快的做法：\n"
                f"- 缩小处理范围，或拆成多条小命令分步执行；\n"
                f"- 下载/安装类命令确认网络可用（可加 --quiet 减少输出）；\n"
                f"- 若确实需要很久，先汇报当前进度，让用户决定是否继续。")
    except asyncio.CancelledError:
        # 用户点「停止」/任务取消：杀掉整棵进程树，绝不留孤儿子进程
        await _kill_process_tree(proc)
        raise

    output = (stdout or b"")[:4000]
    err = (stderr or b"")[:2000]
    return (f"exit_code={proc.returncode}\n[stdout]\n"
            f"{output.decode('utf-8', errors='replace')}\n"
            f"[stderr]\n{err.decode('utf-8', errors='replace')}")


async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """
    杀掉子进程**整棵树**并收尸。

    shell=True 启动的是 cmd.exe / bash，真正干活的命令（pip、python 等）
    是它的子进程甚至孙进程——proc.kill() 只杀外壳，会留下孤儿进程继续
    占用管道与资源。Windows 用 taskkill /F /T 杀树；POSIX 杀进程组
    （create_subprocess_shell 时已开 start_new_session）。
    """
    if proc.returncode is not None:
        return
    try:
        if sys.platform == "win32":
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/F", "/T", "/PID", str(proc.pid),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            try:
                await asyncio.wait_for(killer.wait(), timeout=5)
            except asyncio.TimeoutError:
                killer.kill()
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    # 收尸：等进程真正退出，避免僵尸（有上限，别为收尸再卡住调用方）
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except Exception:
        pass
