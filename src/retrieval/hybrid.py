from typing import Any


class HybridRetriever:
    def __init__(
        self,
        bm25_retriever,
        vector_store,
        embedder,
        rrf_k: int = 60,
    ):
        self.bm25_retriever = bm25_retriever
        self.vector_store = vector_store
        self.embedder = embedder
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 8,
        bm25_k: int = 20,
        vector_k: int = 20,
    ) -> list[dict]:
        bm25_results = self.bm25_retriever.search(query, top_k=bm25_k)
        query_embedding = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(query_embedding, top_k=vector_k)

        fused: dict[str, dict[str, Any]] = {}
        self._merge_ranked_results(fused, bm25_results, source="bm25")
        self._merge_ranked_results(fused, vector_results, source="vector")

        return sorted(
            fused.values(),
            key=lambda result: result["rrf_score"],
            reverse=True,
        )[:top_k]

    def _merge_ranked_results(
        self,
        fused: dict[str, dict[str, Any]],
        results: list[dict],
        source: str,
    ) -> None:
        for rank, result in enumerate(results, start=1):
            chunk_id = result["id"]
            if chunk_id not in fused:
                fused[chunk_id] = {
                    "id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "vector_rank": None,
                    "bm25_score": None,
                    "vector_score": None,
                }

            fused_result = fused[chunk_id]
            fused_result["rrf_score"] += 1 / (self.rrf_k + rank)
            fused_result[f"{source}_rank"] = rank
            fused_result[f"{source}_score"] = float(result["score"])
