"""
tests/test_registry.py —— 工具注册表测试（防幻觉核心逻辑）
==========================================================
验证：严格匹配 / 模糊匹配建议 / 未知工具提示。
"""

import pytest
from langchain.tools import tool

from tools.registry import TOOL_REGISTRY, all_tools, register, resolve_tool, suggest_tools


def test_builtin_tools_registered():
    """导入 tools 包后，内置工具已全部注册。"""
    import tools  # noqa: F401  触发 @register

    names = set(TOOL_REGISTRY)
    for expected in ("read_file", "write_file", "edit_file",
                     "list_dir", "run_shell", "gen_docx",
                     "gen_xlsx", "gen_pptx", "search_web"):
        assert expected in names, f"工具 {expected} 未注册"


def test_resolve_exact_match():
    """精确匹配成功，返回工具对象且无提示。"""
    import tools  # noqa: F401

    t, hint = resolve_tool("read_file")
    assert t is not None and hint is None
    assert t.name == "read_file"


def test_resolve_fuzzy_suggestion():
    """模型幻觉工具名（read_files）→ 给出 read_file 建议。"""
    import tools  # noqa: F401

    t, hint = resolve_tool("read_files")
    assert t is None
    assert "read_file" in hint and "你是否想调用" in hint


def test_resolve_unknown_no_close_match():
    """完全无关的名字 → 列出全部可用工具，而不是瞎猜。"""
    import tools  # noqa: F401

    t, hint = resolve_tool("zzzz_not_a_tool")
    assert t is None
    assert "可用工具" in hint


def test_duplicate_register_raises():
    """重复注册同名工具 → ValueError（防止覆盖）。"""
    import tools  # noqa: F401

    @tool
    def read_file(path: str) -> str:  # noqa: F811  故意重名
        """与内置工具重名，用于测试。"""
        return path

    with pytest.raises(ValueError, match="重复注册"):
        register(read_file)


def test_all_tools_unique():
    """注册表内工具名不重复。"""
    import tools  # noqa: F401

    names = [t.name for t in all_tools()]
    assert len(names) == len(set(names))
