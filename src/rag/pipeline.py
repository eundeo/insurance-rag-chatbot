from __future__ import annotations

import re

from src.llm.prompt import build_rag_prompt


OUT_OF_SCOPE_ANSWER = "이 질문은 제공된 보험 고시 문서와 직접 관련이 없어 답변할 수 없습니다."
INSUFFICIENT_CONTEXT_ANSWER = "제공된 문서 범위에서는 확인되지 않습니다."
DEFAULT_RELEVANCE_THRESHOLD = 0.020
CODE_PATTERN_RE = re.compile(r"[A-Z]{2}\d+|[가-힣]-\d+|응-\d+")
SMALL_TALK_PATTERNS = (
    "오늘 날씨",
    "날씨 어때",
    "너 누구",
    "점심 뭐",
    "연애 상담",
    "파이썬 코드",
    "코드 짜줘",
)


def is_relevant_query(
    query: str,
    retrieved_contexts: list[dict],
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> bool:
    if not retrieved_contexts:
        return False

    normalized_query = query.strip().lower()
    if any(pattern in normalized_query for pattern in SMALL_TALK_PATTERNS):
        return False

    top_score = max(float(context.get("rrf_score", 0.0)) for context in retrieved_contexts)
    effective_threshold = threshold * 0.5 if CODE_PATTERN_RE.search(query) else threshold
    return top_score >= effective_threshold


class RAGPipeline:
    def __init__(
        self,
        retriever,
        llm_client,
        top_k: int = 8,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

    def answer(
        self,
        query: str,
        temperature: float = 0.2,
    ) -> dict:
        contexts = self.retriever.search(query, top_k=self.top_k)
        is_relevant = is_relevant_query(
            query,
            contexts,
            threshold=self.relevance_threshold,
        )

        sources = [_source_from_context(context) for context in contexts]
        if not is_relevant:
            return {
                "answer": OUT_OF_SCOPE_ANSWER,
                "sources": [],
                "is_relevant": False,
            }

        prompt = build_rag_prompt(query, contexts)
        answer = self.llm_client.generate(prompt, temperature=temperature)
        return {
            "answer": answer,
            "sources": sources,
            "is_relevant": True,
        }


def _source_from_context(context: dict) -> dict:
    metadata = context.get("metadata", {})
    return {
        "id": context.get("id"),
        "page_start": metadata.get("page_start"),
        "page_end": metadata.get("page_end"),
        "section": metadata.get("section"),
        "codes": metadata.get("codes", []),
        "text": context.get("text", ""),
        "rrf_score": context.get("rrf_score", 0.0),
    }
