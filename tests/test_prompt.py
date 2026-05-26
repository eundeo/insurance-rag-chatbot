from src.llm.prompt import build_rag_prompt
from src.rag.pipeline import is_relevant_query


def sample_contexts(score: float = 0.0327) -> list[dict]:
    return [
        {
            "id": "ch_000001",
            "text": "나. 재진 진찰료 Established Patient",
            "metadata": {
                "page_start": 101,
                "page_end": 102,
                "section": "나. 재진 진찰료",
                "codes": ["AA222"],
            },
            "rrf_score": score,
        }
    ]


def test_build_rag_prompt_contains_context():
    prompt = build_rag_prompt("재진 진찰료", sample_contexts())

    assert "[문서 1]" in prompt
    assert "id: ch_000001" in prompt
    assert "section: 나. 재진 진찰료" in prompt
    assert "codes: AA222" in prompt


def test_build_rag_prompt_contains_out_of_scope_instruction():
    prompt = build_rag_prompt("오늘 날씨 어때?", sample_contexts())

    assert "이 질문은 제공된 보험 고시 문서와 직접 관련이 없어 답변할 수 없습니다." in prompt


def test_build_rag_prompt_requires_sources_section():
    prompt = build_rag_prompt("재진 진찰료", sample_contexts())

    assert "[출처]" in prompt


def test_build_rag_prompt_requires_korean_answer():
    prompt = build_rag_prompt("재진 진찰료", sample_contexts())

    assert "중국어" in prompt
    assert "한국어가 아닌 언어로 답변하지 않는다" in prompt


def test_is_relevant_query_false_for_empty_contexts():
    assert is_relevant_query("재진 진찰료", []) is False


def test_is_relevant_query_false_for_small_talk():
    assert is_relevant_query("오늘 날씨 어때?", sample_contexts()) is False


def test_is_relevant_query_false_below_threshold():
    assert is_relevant_query("재진 진찰료", sample_contexts(score=0.01)) is False


def test_is_relevant_query_relaxes_threshold_for_code_query():
    assert is_relevant_query("AA222는 뭐야?", sample_contexts(score=0.012)) is True
