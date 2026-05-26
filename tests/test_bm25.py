from src.retrieval.bm25 import BM25Retriever


def sample_chunks() -> list[dict]:
    return [
        {
            "id": "ch_000001",
            "text": "초진 진찰료 산정 기준 AA157",
            "metadata": {"page_start": 1, "section": "제1절 기본진료료"},
        },
        {
            "id": "ch_000002",
            "text": "재진 진찰료는 외래 진료 기준에 따라 산정한다.",
            "metadata": {"page_start": 2, "section": "제1절 기본진료료"},
        },
        {
            "id": "ch_000003",
            "text": "영상진단료와 방사선 치료료 항목",
            "metadata": {"page_start": 3, "section": "제2절 영상진단료"},
        },
    ]


def test_bm25_retriever_creation():
    retriever = BM25Retriever(sample_chunks())

    assert retriever.chunks[0]["id"] == "ch_000001"


def test_tokenize_returns_list():
    retriever = BM25Retriever(sample_chunks())

    assert isinstance(retriever.tokenize("재진 진찰료 AA157"), list)


def test_search_returns_top_k_results():
    retriever = BM25Retriever(sample_chunks()).build()

    results = retriever.search("재진 진찰료", top_k=2)

    assert len(results) == 2


def test_search_results_sorted_by_score_descending():
    retriever = BM25Retriever(sample_chunks()).build()

    results = retriever.search("재진 진찰료", top_k=3)
    scores = [result["score"] for result in results]

    assert scores == sorted(scores, reverse=True)


def test_search_preserves_metadata():
    retriever = BM25Retriever(sample_chunks()).build()

    result = retriever.search("재진", top_k=1)[0]

    assert result["metadata"]["page_start"] == 2
    assert result["metadata"]["section"] == "제1절 기본진료료"
