"""
tools/coding.py —— 代码开发类工具
==================================
文件读写 / 精确编辑 / 目录浏览 / Shell 执行。
错误处理策略：可重试的异常（文件不存在、命令超时等）直接 raise，
交给 ToolRetryMiddleware / ToolErrorMiddleware 处理；
不可恢复的情况（文件过大）返回说明文本让模型换方案。
"""

import re
import subprocess
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


@register
@tool(args_schema=RunShellArgs)
def run_shell(command: str, timeout: int = 60) -> str:
    """执行 Shell 命令（跑测试/装依赖/看版本等），超时自动终止。"""
    lowered = command.lower()
    for pattern in _DANGEROUS_PATTERNS:
        if pattern in lowered:
            # 高危命令：不执行，直接报错让模型放弃该方案
            raise PermissionError(f"命令被安全策略拦截：包含危险操作 `{pattern}`")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"命令超时（>{timeout}s）: {command[:120]}") from exc
    output = (result.stdout or "")[:4000]
    err = (result.stderr or "")[:2000]
    return f"exit_code={result.returncode}\n[stdout]\n{output}\n[stderr]\n{err}"
