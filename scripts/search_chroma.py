import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.retrieval.embedder import SentenceTransformerEmbedder
from src.retrieval.vector_store import ChromaVectorStore


def _preview(text: str, limit: int = 200) -> str:
    return " ".join(text.split())[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Search chunks with Chroma.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--persist-dir", default=str(config.CHROMA_DIR))
    parser.add_argument("--model", default="BAAI/bge-m3")
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder(model_name=args.model)
    query_embedding = embedder.embed_query(args.query)
    store = ChromaVectorStore(persist_dir=args.persist_dir)
    results = store.search(query_embedding, top_k=args.top_k)

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print("=" * 80)
        print(f"rank: {rank}")
        print(f"similarity score: {result['score']:.4f}")
        print(f"page: {metadata.get('page_start')} ~ {metadata.get('page_end')}")
        print(f"section: {metadata.get('section')}")
        print(f"codes: {metadata.get('codes')}")
        print(f"text preview: {_preview(result['text'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
