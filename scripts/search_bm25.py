import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.retrieval.bm25 import BM25Retriever


def _preview(text: str, limit: int = 200) -> str:
    return " ".join(text.split())[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Search chunks with BM25.")
    parser.add_argument("--query", required=True, help="Search query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results.")
    args = parser.parse_args()

    if not config.BM25_PATH.exists():
        print(f"BM25 index not found: {config.BM25_PATH}")
        print("Run python scripts/build_bm25.py first.")
        return 1

    retriever = BM25Retriever.load(config.BM25_PATH)
    results = retriever.search(args.query, top_k=args.top_k)

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print("=" * 80)
        print(f"rank: {rank}")
        print(f"score: {result['score']:.4f}")
        print(f"page: {metadata.get('page_start')}")
        print(f"section: {metadata.get('section')}")
        print(f"text preview: {_preview(result['text'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
