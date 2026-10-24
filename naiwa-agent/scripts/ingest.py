"""
scripts/ingest.py —— RAG 知识库入库脚本
========================================
把文档批量写入向量库，供 Agent 对话时检索。

用法：
    python scripts/ingest.py 文件1.txt 文件2.md ...
    python scripts/ingest.py ./docs/          （整个目录下的 .txt/.md）

前置条件：
    1) pip install -r requirements-rag.txt
    2) .env 中设置 RAG_ENABLED=true
"""

import sys
from pathlib import Path

# 支持从项目根目录直接运行：python scripts/ingest.py ...
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import settings  # noqa: E402
from agent.observability import setup_logging  # noqa: E402
from memory.rag import CHROMA_AVAILABLE, ingest  # noqa: E402

TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".py", ".java", ".c", ".cpp", ".json", ".csv"}


def collect_files(args: list[str]) -> list[Path]:
    """把命令行参数（文件或目录）展开成文本文件列表。"""
    files: list[Path] = []
    for arg in args:
        p = Path(arg).expanduser()
        if p.is_dir():
            files.extend(f for f in p.rglob("*") if f.suffix.lower() in TEXT_EXTS)
        elif p.is_file():
            if p.suffix.lower() in TEXT_EXTS:
                files.append(p)
            else:
                print(f"[跳过] 不支持的格式: {p}（支持 {sorted(TEXT_EXTS)}）")
        else:
            print(f"[跳过] 路径不存在: {p}")
    return files


def main() -> int:
    if not settings.rag_enabled:
        print("[错误] 请先在 .env 中设置 RAG_ENABLED=true")
        return 1
    if not CHROMA_AVAILABLE:
        print("[错误] RAG 依赖未安装：pip install -r requirements-rag.txt")
        return 1
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    setup_logging()
    files = collect_files(sys.argv[1:])
    if not files:
        print("[错误] 没有找到可入库的文本文件")
        return 1

    texts, names = [], []
    for f in files:
        try:
            texts.append(f.read_text(encoding="utf-8"))
            names.append(str(f))
        except UnicodeDecodeError:
            print(f"[跳过] 非 UTF-8 文本文件: {f}")

    chunks = ingest(texts, metadatas=[{"source": n} for n in names])
    print(f"[完成] {len(texts)} 个文件 → {chunks} 个知识块已入库")
    print(f"向量库位置: {settings.chroma_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
