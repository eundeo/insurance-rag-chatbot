import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.retrieval.bm25 import BM25Retriever


def load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> int:
    chunks_path = config.PROCESSED_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"chunks.jsonl not found: {chunks_path}")
        print("Run chunks ingest before building BM25.")
        return 1

    chunks = load_chunks(chunks_path)
    retriever = BM25Retriever(chunks).build()
    retriever.save(config.BM25_PATH)

    token_lengths = [len(tokens) for tokens in retriever.tokenized_corpus]
    avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0

    print(f"Total chunks: {len(chunks)}")
    print(f"Average token length: {avg_tokens:.1f}")
    print(f"Saved BM25 index: {config.BM25_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
