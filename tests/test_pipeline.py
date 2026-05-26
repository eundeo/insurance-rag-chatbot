from src.rag.pipeline import (
    INSUFFICIENT_CONTEXT_ANSWER,
    OUT_OF_SCOPE_ANSWER,
    RAGPipeline,
    is_relevant_query,
)


class MockRetriever:
    def __init__(self, contexts):
        self.contexts = contexts

    def search(self, query: str, top_k: int = 8):
        return self.contexts[:top_k]


class MockLLMClient:
    def __init__(self, response: str = "문서 기반 답변입니다.\n\n[출처]\n- ch_000001"):
        self.calls = []
        self.response = response

    def generate(self, prompt: str, temperature: float = 0.2):
        self.calls.append({"prompt": prompt, "temperature": temperature})
        return self.response


def sample_context(
    score: float = 0.0327,
    bm25_rank: int | None = 1,
    vector_rank: int | None = 1,
) -> dict:
    return {
        "id": "ch_000001",
        "text": "나. 재진 진찰료 Established Patient",
        "metadata": {
            "page_start": 101,
            "page_end": 102,
            "section": "나. 재진 진찰료",
            "codes": ["AA222"],
        },
        "rrf_score": score,
        "bm25_rank": bm25_rank,
        "vector_rank": vector_rank,
    }


def test_irrelevant_query_does_not_call_llm():
    llm = MockLLMClient()
    pipeline = RAGPipeline(MockRetriever([sample_context()]), llm)

    result = pipeline.answer("오늘 날씨 어때?")

    assert result["answer"] == OUT_OF_SCOPE_ANSWER
    assert result["is_relevant"] is False
    assert result["sources"] == []
    assert llm.calls == []


def test_empty_contexts_return_blocked_answer():
    llm = MockLLMClient()
    pipeline = RAGPipeline(MockRetriever([]), llm)

    result = pipeline.answer("재진 진찰료")

    assert result["answer"] == OUT_OF_SCOPE_ANSWER
    assert result["sources"] == []
    assert llm.calls == []


def test_relevant_query_calls_llm():
    llm = MockLLMClient()
    pipeline = RAGPipeline(MockRetriever([sample_context()]), llm)

    result = pipeline.answer("재진 진찰료 야간 가산", temperature=0.1)

    assert result["is_relevant"] is True
    assert "문서 기반 답변" in result["answer"]
    assert len(llm.calls) == 1
    assert llm.calls[0]["temperature"] == 0.1


def test_sources_are_returned():
    pipeline = RAGPipeline(MockRetriever([sample_context()]), MockLLMClient())

    result = pipeline.answer("재진 진찰료")

    assert result["sources"][0]["id"] == "ch_000001"
    assert result["sources"][0]["page_start"] == 101
    assert result["sources"][0]["codes"] == ["AA222"]


def test_threshold_below_minimum_blocks():
    llm = MockLLMClient()
    pipeline = RAGPipeline(
        MockRetriever([sample_context(score=0.01, bm25_rank=None, vector_rank=None)]),
        llm,
        relevance_threshold=0.020,
    )

    result = pipeline.answer("재진 진찰료")

    assert result["is_relevant"] is False
    assert llm.calls == []


def test_code_pattern_query_relaxes_threshold():
    assert is_relevant_query(
        "AA222는 어떤 항목이야?",
        [sample_context(score=0.012, bm25_rank=None, vector_rank=None)],
    )


def test_top_bm25_rank_signal_allows_low_rrf_score():
    assert is_relevant_query(
        "식도조루술의 코드를 알려줘",
        [sample_context(score=0.016, bm25_rank=1, vector_rank=None)],
    )


def test_small_talk_blocks_even_with_top_rank_signal():
    assert not is_relevant_query(
        "오늘 날씨 어때?",
        [sample_context(score=0.016, bm25_rank=1, vector_rank=None)],
    )


def test_chinese_llm_response_is_replaced_with_korean_fallback():
    llm = MockLLMClient(response="根据提供的文件，无法确认该内容。")
    pipeline = RAGPipeline(MockRetriever([sample_context()]), llm)

    result = pipeline.answer("재진 진찰료")

    assert result["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    assert result["is_relevant"] is True
    assert len(llm.calls) == 1
