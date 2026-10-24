"""
tools/registry.py —— 工具注册表（防幻觉的第一道防线）
======================================================
设计目标（对应需求"要对工具名进行匹配，如果不匹配就要提示，
并筛选出意思较近的工具名来方便修改与匹配"）：

1. 所有工具必须先注册才能被使用（@register 装饰器）；
2. resolve_tool() 做严格匹配：名字不存在直接拒绝；
3. 匹配失败时用编辑距离（difflib）给出相近工具名建议，
   例如模型想调 read_files → 提示"你想调 read_file 吗？"。
"""

from difflib import get_close_matches

# 注册表：工具名 -> BaseTool 对象
TOOL_REGISTRY: dict = {}


def register(tool):
    """装饰器：把 @tool 装饰后的工具登记进注册表（重复注册直接报错）。"""
    if tool.name in TOOL_REGISTRY:
        raise ValueError(f"工具名重复注册: {tool.name}")
    TOOL_REGISTRY[tool.name] = tool
    return tool


def all_tools() -> list:
    """返回全部已注册工具（给 create_agent 用）。"""
    return list(TOOL_REGISTRY.values())


def suggest_tools(name: str, n: int = 3, cutoff: float = 0.5) -> list[str]:
    """编辑距离模糊匹配：返回与 name 最相近的 n 个已注册工具名。"""
    return get_close_matches(name, TOOL_REGISTRY.keys(), n=n, cutoff=cutoff)


def resolve_tool(name: str):
    """
    严格匹配工具名。

    返回:
        (tool, None)          匹配成功
        (None, 提示文本)      匹配失败，提示文本包含相近工具名建议
    """
    if name in TOOL_REGISTRY:
        return TOOL_REGISTRY[name], None
    close = suggest_tools(name)
    if close:
        suggestions = "、".join(f"`{c}`" for c in close)
        return None, (
            f"未找到工具 `{name}`。你是否想调用：{suggestions}？"
            "请修正工具名后重试。"
        )
    return None, (
        f"未找到工具 `{name}`。当前可用工具："
        + ", ".join(TOOL_REGISTRY.keys())
    )
