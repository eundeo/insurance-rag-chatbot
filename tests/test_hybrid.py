from src.retrieval.hybrid import HybridRetriever


class DummyBM25Retriever:
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return [
            {
                "id": "ch_001",
                "score": 10.0,
                "text": "재진 진찰료",
                "metadata": {"page_start": 1, "section": "나. 재진 진찰료"},
            },
            {
                "id": "ch_002",
                "score": 8.0,
                "text": "야간 가산",
                "metadata": {"page_start": 2, "section": "제1절 기본진료료"},
            },
        ][:top_k]


class DummyVectorStore:
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        return [
            {
                "id": "ch_002",
                "score": 0.9,
                "text": "야간 가산",
                "metadata": {"page_start": 2, "section": "제1절 기본진료료"},
            },
            {
                "id": "ch_003",
                "score": 0.7,
                "text": "치과의원",
                "metadata": {"page_start": 3, "section": "가. 초진 진찰료"},
            },
        ][:top_k]


class DummyEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


def make_retriever(rrf_k: int = 60) -> HybridRetriever:
    return HybridRetriever(
        bm25_retriever=DummyBM25Retriever(),
        vector_store=DummyVectorStore(),
        embedder=DummyEmbedder(),
        rrf_k=rrf_k,
    )


def test_same_chunk_id_is_merged():
    results = make_retriever().search("재진 진찰료", top_k=5)

    ids = [result["id"] for result in results]

    assert ids.count("ch_002") == 1
    assert len(ids) == 3


def test_rrf_score_is_calculated():
    result = next(result for result in make_retriever(rrf_k=60).search("query") if result["id"] == "ch_002")

    assert result["rrf_score"] == (1 / 62) + (1 / 61)


def test_results_are_sorted_by_rrf_score_descending():
    results = make_retriever().search("query", top_k=5)
    scores = [result["rrf_score"] for result in results]

    assert scores == sorted(scores, reverse=True)


def test_source_ranks_are_preserved():
    result = next(result for result in make_retriever().search("query") if result["id"] == "ch_002")

    assert result["bm25_rank"] == 2
    assert result["vector_rank"] == 1
    assert result["bm25_score"] == 8.0
    assert result["vector_score"] == 0.9


def test_top_k_limits_results():
    results = make_retriever().search("query", top_k=2)

    assert len(results) == 2
