import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.llm.ollama_client import OllamaClient
from src.rag.pipeline import RAGPipeline
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.embedder import SentenceTransformerEmbedder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import ChromaVectorStore


EXIT_COMMANDS = {"exit", "quit", "q"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local insurance RAG CLI.")
    parser.add_argument("--model", default=config.OLLAMA_MODEL)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.020)
    args = parser.parse_args()

    if not config.BM25_PATH.exists():
        print(f"BM25 index not found: {config.BM25_PATH}")
        print("Run python scripts/build_bm25.py first.")
        return 1

    if not config.CHROMA_DIR.exists():
        print(f"Chroma index directory not found: {config.CHROMA_DIR}")
        print("Run python scripts/build_chroma.py first.")
        return 1

    bm25_retriever = BM25Retriever.load(config.BM25_PATH)
    vector_store = ChromaVectorStore(config.CHROMA_DIR)
    embedder = SentenceTransformerEmbedder()
    retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_store=vector_store,
        embedder=embedder,
    )
    llm_client = OllamaClient(
        base_url=config.OLLAMA_BASE_URL,
        model=args.model,
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
        top_k=args.top_k,
        relevance_threshold=args.threshold,
    )

    print("Insurance RAG CLI. Type exit, quit, or q to stop.")
    while True:
        try:
            query = input("\n질문> ").strip()
        except EOFError:
            print()
            break

        if not query:
            continue
        if query.lower() in EXIT_COMMANDS:
            break

        result = pipeline.answer(query, temperature=args.temperature)
        print("\n답변")
        print(result["answer"])

        print("\n출처")
        if not result["sources"]:
            print("- 없음")
            continue

        for source in result["sources"]:
            print(
                "- "
                f"page {source['page_start']}~{source['page_end']} | "
                f"section: {source['section']} | "
                f"codes: {source['codes']} | "
                f"rrf_score: {float(source['rrf_score']):.6f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
