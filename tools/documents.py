"""
tools/documents.py —— 文档生成工具
====================================
Word / Excel / PPT 三类文档生成，对应"生成各种类型的文档"的能力要求。
所有工具都对路径、参数做校验，异常统一向上抛（由中间件捕获反馈给模型）。
"""

import json
import re
from pathlib import Path

from langchain.tools import tool
from pydantic import BaseModel, Field

from tools.registry import register


# ---------------------------------------------------------------------------
# 参数 Schema
# ---------------------------------------------------------------------------
class GenDocxArgs(BaseModel):
    markdown_path: str = Field(..., description="Markdown 源文件绝对路径")
    output_path: str = Field(..., description="输出的 .docx 绝对路径")
    title: str = Field("", description="文档标题（可选，会作为一级标题插入）")


class GenXlsxArgs(BaseModel):
    output_path: str = Field(..., description="输出的 .xlsx 绝对路径")
    rows: str = Field(
        ...,
        description='二维数组的 JSON 字符串，第一行为表头。'
                    '示例：[["姓名","分数"],["张三",90]]',
    )
    sheet_name: str = Field("Sheet1", description="工作表名称")


class GenPptxArgs(BaseModel):
    output_path: str = Field(..., description="输出的 .pptx 绝对路径")
    title: str = Field(..., description="PPT 封面标题")
    slides: str = Field(
        ...,
        description='幻灯片内容的 JSON 字符串数组，每项 {"title": "...", "content": "..."}。'
                    'content 内用 \\n 分条。示例：[{"title":"背景","content":"第一点\\n第二点"}]',
    )


# ---------------------------------------------------------------------------
# 内部辅助：Markdown 行 → Word 段落
# ---------------------------------------------------------------------------
def _add_runs(paragraph, text: str) -> None:
    """解析 **粗体** 语法：按 ** 切分，奇数段加粗。"""
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        run = paragraph.add_run(part)
        run.bold = (i % 2 == 1)


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------
@register
@tool(args_schema=GenDocxArgs)
def gen_docx(markdown_path: str, output_path: str, title: str = "") -> str:
    """把 Markdown 文件转换成排版好的 Word 文档（支持标题/列表/粗体）。"""
    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError as exc:
        raise ImportError("缺少 python-docx，请执行: pip install python-docx") from exc

    src = Path(markdown_path).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Markdown 源文件不存在: {src}")

    doc = Document()
    # 全局字号
    doc.styles["Normal"].font.size = Pt(11)

    if title:
        doc.add_heading(title, level=0)

    for raw_line in src.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=3)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=1)
        elif line.startswith(("- ", "* ")):
            _add_runs(doc.add_paragraph(style="List Bullet"), line[2:].strip())
        elif re.match(r"^\d+[.)]\s+", line):
            _add_runs(doc.add_paragraph(style="List Number"),
                      re.sub(r"^\d+[.)]\s+", "", line).strip())
        elif line.strip() in ("---", "***"):
            continue  # 分隔线不处理
        else:
            _add_runs(doc.add_paragraph(), line.strip())

    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        doc.save(str(out))
    except PermissionError as exc:
        raise PermissionError(
            f"无法写入 {out}，文件可能正被 Word/WPS 打开，请关闭后重试。"
        ) from exc
    return f"已生成 {out}（{len(doc.paragraphs)} 个段落）"


@register
@tool(args_schema=GenXlsxArgs)
def gen_xlsx(output_path: str, rows: str, sheet_name: str = "Sheet1") -> str:
    """把二维数组（JSON）写成 Excel 表格，第一行作为表头。"""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError as exc:
        raise ImportError("缺少 openpyxl，请执行: pip install openpyxl") from exc

    try:
        data = json.loads(rows)
    except json.JSONDecodeError as exc:
        raise ValueError(
            'rows 必须是合法 JSON 二维数组。示例：[["姓名","分数"],["张三",90]]'
        ) from exc
    if not isinstance(data, list) or not data or not isinstance(data[0], list):
        raise ValueError("rows 必须是非空二维数组，第一行为表头。")

    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name or "Sheet1"

    header_fill = PatternFill("solid", fgColor="4472C4")
    for r, row in enumerate(data, start=1):
        for c, value in enumerate(row, start=1):
            cell = ws.cell(row=r, column=c, value=value)
            if r == 1:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center")

    # 自适应列宽（简单估算）
    for column_cells in ws.columns:
        width = max(len(str(cell.value or "")) for cell in column_cells) + 4
        ws.column_dimensions[column_cells[0].column_letter].width = min(width, 40)

    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        wb.save(str(out))
    except PermissionError as exc:
        raise PermissionError(
            f"无法写入 {out}，文件可能正被 Excel/WPS 打开，请关闭后重试。"
        ) from exc
    return f"已生成 {out}（{len(data)} 行 × {len(data[0])} 列）"


@register
@tool(args_schema=GenPptxArgs)
def gen_pptx(output_path: str, title: str, slides: str) -> str:
    """生成 PPT：1 页封面 + 每项内容 1 页（标题 + 要点）。"""
    try:
        from pptx import Presentation
        from pptx.util import Pt
    except ImportError as exc:
        raise ImportError("缺少 python-pptx，请执行: pip install python-pptx") from exc

    try:
        items = json.loads(slides)
    except json.JSONDecodeError as exc:
        raise ValueError(
            'slides 必须是合法 JSON 数组。示例：[{"title":"背景","content":"第一点\\n第二点"}]'
        ) from exc
    if not isinstance(items, list) or not items:
        raise ValueError("slides 必须是非空数组。")

    prs = Presentation()  # 默认 16:9 尺寸

    # --- 封面 ---
    cover = prs.slides.add_slide(prs.slide_layouts[0])
    cover.shapes.title.text = title
    if len(cover.placeholders) > 1:
        cover.placeholders[1].text = "GeneralTaskAgent 自动生成"

    # --- 内容页 ---
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("slides 数组的每一项都必须是对象。")
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = str(item.get("title", "未命名"))
        body = slide.placeholders[1].text_frame
        body.text = ""  # 清空占位
        lines = str(item.get("content", "")).split("\n")
        for i, line in enumerate([ln.strip() for ln in lines if ln.strip()]):
            para = body.paragraphs[0] if i == 0 else body.add_paragraph()
            para.text = line
            para.font.size = Pt(20)

    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        prs.save(str(out))
    except PermissionError as exc:
        raise PermissionError(
            f"无法写入 {out}，文件可能正被 PowerPoint/WPS 打开，请关闭后重试。"
        ) from exc
    return f"已生成 {out}（共 {len(prs.slides._sldIdLst)} 页）"
