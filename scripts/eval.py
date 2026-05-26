import argparse
import json
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


def load_eval_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def page_matches(result: dict, expected_pages: list[int], tolerance: int = 0) -> bool:
    metadata = result.get("metadata", {})
    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    if page_start is None or page_end is None:
        return False

    retrieved_pages = range(int(page_start), int(page_end) + 1)
    return any(
        abs(retrieved_page - expected_page) <= tolerance
        for retrieved_page in retrieved_pages
        for expected_page in expected_pages
    )


def keywords_match(results: list[dict], expected_keywords: list[str]) -> bool:
    combined_text = "\n".join(result.get("text", "") for result in results).lower()
    return all(keyword.lower() in combined_text for keyword in expected_keywords)


def build_retriever(model_name: str) -> HybridRetriever:
    bm25_retriever = BM25Retriever.load(config.BM25_PATH)
    vector_store = ChromaVectorStore(config.CHROMA_DIR)
    embedder = SentenceTransformerEmbedder(model_name=model_name)
    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_store=vector_store,
        embedder=embedder,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate hybrid retrieval smoke set.")
    parser.add_argument("--dataset", default="eval/smoke_qa.jsonl")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--model", default="BAAI/bge-m3")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Evaluation dataset not found: {dataset_path}")
        return 1
    if not config.BM25_PATH.exists() or not config.CHROMA_DIR.exists():
        print("Required indexes not found. Run:")
        print("python scripts/build_bm25.py")
        print("python scripts/build_chroma.py")
        return 1

    dataset = load_eval_dataset(dataset_path)
    retriever = build_retriever(args.model)

    recall_passes = 0
    page_accuracy_passes = 0
    keyword_passes = 0

    for item in dataset:
        query = item["query"]
        expected_pages = item["expected_pages"]
        expected_keywords = item["expected_keywords"]
        retrieved = retriever.search(query, top_k=args.top_k)

        recall_pass = any(page_matches(result, expected_pages) for result in retrieved)
        top1 = retrieved[0] if retrieved else {}
        top1_page = top1.get("metadata", {}).get("page_start")
        page_accuracy_pass = bool(top1) and page_matches(
            top1,
            expected_pages,
            tolerance=1,
        )
        keyword_pass = keywords_match(retrieved, expected_keywords)

        recall_passes += int(recall_pass)
        page_accuracy_passes += int(page_accuracy_pass)
        keyword_passes += int(keyword_pass)

        print("=" * 40)
        print(f"Query: {query}")
        print(f"Recall@{args.top_k}: {'PASS' if recall_pass else 'FAIL'}")
        print(f"Top1 Page: {top1_page}")
        print(f"Expected: {expected_pages}")
        print(f"Page Accuracy: {'PASS' if page_accuracy_pass else 'FAIL'}")
        print(f"Keyword Match: {'PASS' if keyword_pass else 'FAIL'}")

    total = len(dataset)
    recall = recall_passes / total if total else 0
    page_accuracy = page_accuracy_passes / total if total else 0
    keyword_match = keyword_passes / total if total else 0

    print("=" * 40)
    print(f"Total Queries: {total}")
    print(f"Recall@{args.top_k}: {recall:.2f}")
    print(f"Page Accuracy: {page_accuracy:.2f}")
    print(f"Keyword Match: {keyword_match:.2f}")
    print("=" * 40)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
