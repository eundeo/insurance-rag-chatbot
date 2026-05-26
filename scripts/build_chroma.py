import argparse
import json
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.retrieval.embedder import SentenceTransformerEmbedder
from src.retrieval.vector_store import DEFAULT_COLLECTION_NAME, ChromaVectorStore


def load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build Chroma vector index.")
    parser.add_argument("--chunks", default=str(config.PROCESSED_DIR / "chunks.jsonl"))
    parser.add_argument("--persist-dir", default=str(config.CHROMA_DIR))
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"chunks.jsonl not found: {chunks_path}")
        print("Run python scripts/ingest.py --stage chunks --pdf <path> first.")
        return 1

    chunks = load_chunks(chunks_path)
    embedder = SentenceTransformerEmbedder(model_name=args.model)
    embeddings = embedder.embed_documents([chunk.get("text", "") for chunk in chunks])

    store = ChromaVectorStore(
        persist_dir=args.persist_dir,
        collection_name=DEFAULT_COLLECTION_NAME,
    )
    store.reset_collection()
    store.add_chunks(chunks, embeddings, batch_size=args.batch_size)

    print(f"Total chunks: {len(chunks)}")
    print(f"Model: {args.model}")
    print(f"Persist dir: {Path(args.persist_dir)}")
    print(f"Collection: {DEFAULT_COLLECTION_NAME}")
    print("Chroma index build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
