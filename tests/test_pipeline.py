from src.rag.pipeline import OUT_OF_SCOPE_ANSWER, RAGPipeline, is_relevant_query


class MockRetriever:
    def __init__(self, contexts):
        self.contexts = contexts

    def search(self, query: str, top_k: int = 8):
        return self.contexts[:top_k]


class MockLLMClient:
    def __init__(self):
        self.calls = []

    def generate(self, prompt: str, temperature: float = 0.2):
        self.calls.append({"prompt": prompt, "temperature": temperature})
        return "문서 기반 답변입니다.\n\n[출처]\n- ch_000001"


def sample_context(score: float = 0.0327) -> dict:
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
        "bm25_rank": 1,
        "vector_rank": 1,
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
        MockRetriever([sample_context(score=0.01)]),
        llm,
        relevance_threshold=0.020,
    )

    result = pipeline.answer("재진 진찰료")

    assert result["is_relevant"] is False
    assert llm.calls == []


def test_code_pattern_query_relaxes_threshold():
    assert is_relevant_query("AA222는 어떤 항목이야?", [sample_context(score=0.012)])
