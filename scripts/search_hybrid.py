import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.embedder import SentenceTransformerEmbedder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import ChromaVectorStore


def _preview(text: str, limit: int = 250) -> str:
    return " ".join(text.split())[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Search chunks with hybrid RRF retrieval.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--bm25-k", type=int, default=20)
    parser.add_argument("--vector-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--bm25-path", default=str(config.BM25_PATH))
    parser.add_argument("--persist-dir", default=str(config.CHROMA_DIR))
    args = parser.parse_args()

    bm25_path = Path(args.bm25_path)
    if not bm25_path.exists():
        print(f"BM25 index not found: {bm25_path}")
        print("Run python scripts/build_bm25.py first.")
        return 1

    persist_dir = Path(args.persist_dir)
    if not persist_dir.exists():
        print(f"Chroma index directory not found: {persist_dir}")
        print("Run python scripts/build_chroma.py first.")
        return 1

    bm25_retriever = BM25Retriever.load(bm25_path)
    vector_store = ChromaVectorStore(persist_dir=persist_dir)
    embedder = SentenceTransformerEmbedder(model_name=args.model)
    retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_store=vector_store,
        embedder=embedder,
        rrf_k=args.rrf_k,
    )

    results = retriever.search(
        args.query,
        top_k=args.top_k,
        bm25_k=args.bm25_k,
        vector_k=args.vector_k,
    )

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print("=" * 80)
        print(f"rank: {rank}")
        print(f"rrf_score: {result['rrf_score']:.6f}")
        print(f"bm25_rank: {result['bm25_rank']}")
        print(f"vector_rank: {result['vector_rank']}")
        print(f"page: {metadata.get('page_start')} ~ {metadata.get('page_end')}")
        print(f"section: {metadata.get('section')}")
        print(f"codes: {metadata.get('codes')}")
        print(f"text preview: {_preview(result['text'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
