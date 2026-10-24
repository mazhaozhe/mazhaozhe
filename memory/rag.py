"""
memory/rag.py —— RAG 长期记忆（可选模块）
===========================================
知识库入库 + 相似度检索。依赖 langchain-chroma（requirements-rag.txt），
未安装或未开启时自动降级（build_rag_tools 返回空列表），
Agent 其余功能完全不受影响 —— 这是"错误处理"要求在模块级的体现。

用法：
    # 1) 把文档入库（Python 侧调用）
    from memory.rag import ingest
    ingest(["第一篇文档内容...", "第二篇..."])
    # 2) Agent 对话中自动使用 rag_search 工具检索
"""

from pathlib import Path

from agent.config import settings
from agent.observability import log_step

# ---------------------------------------------------------------------------
# 依赖探测：chroma 未安装时优雅降级
# ---------------------------------------------------------------------------
try:
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

_SPLITTER = None
_STORE = None


def _get_store() -> "Chroma":
    """懒加载向量库单例（使用 Chroma 默认 onnx 嵌入模型，无需 torch）。"""
    global _STORE, _SPLITTER
    if _STORE is None:
        persist_dir = settings.abs_path(settings.chroma_dir)
        _STORE = Chroma(persist_directory=str(persist_dir))
        _SPLITTER = RecursiveCharacterTextSplitter(
            chunk_size=500,      # 每块约 500 字
            chunk_overlap=50,    # 块间重叠 50 字，保留上下文连续性
        )
    return _STORE


def availability() -> str:
    """返回 RAG 可用性说明（给启动日志 / README 排障用）。"""
    if not settings.rag_enabled:
        return "未开启（.env 中 RAG_ENABLED=false）"
    if not CHROMA_AVAILABLE:
        return "已开启但依赖缺失（pip install -r requirements-rag.txt）"
    return "就绪"


def ingest(texts: list, metadatas: list | None = None) -> int:
    """文档入库：切分 → 向量化 → 写入 Chroma。返回切分后的块数。"""
    if not CHROMA_AVAILABLE:
        raise ImportError("RAG 依赖未安装: pip install -r requirements-rag.txt")
    store = _get_store()
    chunks = _SPLITTER.create_documents(
        [t if isinstance(t, str) else str(t) for t in texts],
        metadatas=metadatas,
    )
    store.add_documents(chunks)
    log_step("rag_ingest", docs=len(texts), chunks=len(chunks))
    return len(chunks)


def retrieve(query: str, k: int | None = None) -> list:
    """相似度检索：返回最相关的 k 个文本块。"""
    if not CHROMA_AVAILABLE:
        raise ImportError("RAG 依赖未安装: pip install -r requirements-rag.txt")
    store = _get_store()
    return [d.page_content for d in store.similarity_search(
        query, k=k or settings.rag_top_k
    )]


def build_rag_tools() -> list:
    """
    构造暴露给 Agent 的 rag_search 工具。
    未开启 / 依赖缺失时返回 []，并在日志说明原因。
    """
    if not settings.rag_enabled:
        return []
    if not CHROMA_AVAILABLE:
        print("[rag] RAG_ENABLED=true 但 chromadb 未安装，已跳过 RAG 工具。"
              "（安装：pip install -r requirements-rag.txt）")
        return []

    from langchain.tools import tool
    from pydantic import BaseModel, Field

    class RagSearchArgs(BaseModel):
        query: str = Field(..., description="检索关键词或问题")

    @tool(args_schema=RagSearchArgs)
    def rag_search(query: str) -> str:
        """检索本地知识库，返回与问题最相关的资料片段。回答资料相关问题时先用本工具。"""
        try:
            results = retrieve(query)
        except Exception as exc:
            raise RuntimeError(f"知识库检索失败: {exc}") from exc
        if not results:
            return (
                f"知识库中没有检索到与「{query}」相关的内容。"
                "如需基于资料回答，请先补充资料；不要编造内容。"
            )
        return "\n---\n".join(f"[片段 {i + 1}] {r}" for i, r in enumerate(results))

    return [rag_search]
