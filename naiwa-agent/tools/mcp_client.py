"""
tools/mcp_client.py —— MCP 工具接入（可选扩展）
================================================
连接外部 MCP Server（Model Context Protocol），
把其暴露的工具转换为 LangChain 工具注入 Agent。

依赖（可选安装）：
    pip install mcp langchain-mcp-adapters

未安装时 import_mcp() 返回 False，Agent 主流程完全不受影响。
"""

from agent.observability import log_step

# 探测可选依赖是否可用
try:
    from mcp import ClientSession, StdioServerParameters  # noqa: F401

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


async def load_mcp_tools(server_command: list[str]) -> list:
    """
    启动一个 MCP Server（stdio 模式）并加载其全部工具。

    参数:
        server_command: 启动命令，如 ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
    返回:
        LangChain 工具列表；失败时抛出带指引的异常。
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP 依赖未安装。请执行: pip install mcp langchain-mcp-adapters"
        )
    from langchain_mcp_adapters.tools import load_mcp_tools as _load
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=server_command[0], args=server_command[1:])
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await _load(session)
                log_step("mcp_loaded", server=server_command[0],
                         tools=[t.name for t in tools])
                # 注意：返回的工具在 session 关闭后不可用，
                # 长期使用需要保持 session 打开（生产环境建议常驻连接管理器）
                return tools
    except Exception as exc:
        raise ConnectionError(
            f"连接 MCP Server 失败（{server_command}）: {exc}。"
            "请确认命令正确、依赖已安装。"
        ) from exc
