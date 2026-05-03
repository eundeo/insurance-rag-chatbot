from __future__ import annotations


def build_rag_prompt(
    query: str,
    contexts: list[dict],
) -> str:
    context_text = "\n\n".join(
        _format_context(index, context) for index, context in enumerate(contexts, start=1)
    )

    return f"""너는 보험 고시 문서 검색 보조자다.

규칙:
1. 반드시 제공된 CONTEXT 안의 내용만 근거로 답한다.
2. CONTEXT에 없는 내용은 추측하지 않는다.
3. 문서와 관련 없는 질문에는 정확히 다음 문장으로 답한다.
   "이 질문은 제공된 보험 고시 문서와 직접 관련이 없어 답변할 수 없습니다."
4. 근거가 부족하면 정확히 다음 문장으로 답한다.
   "제공된 문서 범위에서는 확인되지 않습니다."
5. 답변에는 가능한 경우 관련 조항명, 코드, 페이지, 핵심 산정 규정을 포함한다.
6. 마지막에는 반드시 [출처] 섹션을 포함한다.
7. 법률적/의학적 최종 판단처럼 단정하지 말고 문서 검색 보조 답변으로 작성한다.
8. 한국어로 답변한다.

QUESTION:
{query}

CONTEXT:
{context_text}

ANSWER:
"""


def _format_context(index: int, context: dict) -> str:
    metadata = context.get("metadata", {})
    page_start = metadata.get("page_start", "")
    page_end = metadata.get("page_end", "")
    section = metadata.get("section", "")
    codes = metadata.get("codes", [])
    if isinstance(codes, list):
        codes_text = ", ".join(str(code) for code in codes)
    else:
        codes_text = str(codes)

    return f"""[문서 {index}]
id: {context.get("id", "")}
page: {page_start}-{page_end}
section: {section}
codes: {codes_text}
rrf_score: {context.get("rrf_score", "")}
text:
{context.get("text", "")}"""
