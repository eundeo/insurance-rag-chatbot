from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.llm.ollama_client import OLLAMA_MODEL_ERROR, OLLAMA_SERVER_ERROR, OllamaClient
from src.rag.pipeline import RAGPipeline


INDEX_HELP = """먼저 인덱스를 생성하세요.

```bash
python scripts/build_bm25.py
python scripts/build_chroma.py
```"""


def indexes_ready() -> bool:
    return config.BM25_PATH.exists() and config.CHROMA_DIR.exists()


def create_pipeline(
    model_name: str,
    top_k: int,
    relevance_threshold: float,
) -> RAGPipeline:
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.embedder import SentenceTransformerEmbedder
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.vector_store import ChromaVectorStore

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
        model=model_name,
    )
    return RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
        top_k=top_k,
        relevance_threshold=relevance_threshold,
    )


def get_pipeline(
    model_name: str,
    top_k: int,
    relevance_threshold: float,
) -> RAGPipeline:
    pipeline_config = (model_name, top_k, relevance_threshold)
    if (
        "pipeline" not in st.session_state
        or st.session_state.get("pipeline_config") != pipeline_config
    ):
        st.session_state.pipeline = create_pipeline(
            model_name=model_name,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
        )
        st.session_state.pipeline_config = pipeline_config
    return st.session_state.pipeline


def ask_pipeline(
    pipeline: RAGPipeline,
    query: str,
    temperature: float,
) -> dict:
    return pipeline.answer(query, temperature=temperature)


def normalize_answer_for_ui(answer: str) -> str:
    if answer == OLLAMA_SERVER_ERROR:
        return "Ollama 서버가 실행 중이 아닙니다. ollama serve를 실행하세요."
    if answer == OLLAMA_MODEL_ERROR:
        return "모델을 찾을 수 없습니다. ollama pull qwen2.5:7b-instruct"
    return answer


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander("출처 보기"):
        for index, source in enumerate(sources, start=1):
            codes = source.get("codes", [])
            if isinstance(codes, list):
                codes_text = ", ".join(str(code) for code in codes) or "-"
            else:
                codes_text = str(codes) if codes else "-"

            text_preview = " ".join(source.get("text", "").split())[:300]
            st.markdown(
                f"""[문서 {index}]

페이지: {source.get("page_start")}~{source.get("page_end")}

섹션: {source.get("section") or "-"}

코드: {codes_text}

유사도: {float(source.get("rrf_score", 0.0)):.4f}

내용:

{text_preview}
"""
            )


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("is_relevant"):
            render_sources(message.get("sources", []))


def main() -> None:
    st.set_page_config(page_title="보험 고시 RAG 챗봇", page_icon="📄")
    st.title("보험 고시 RAG 챗봇")
    st.caption("건강보험 고시 문서를 기반으로 답변합니다.")

    with st.sidebar:
        model_name = st.text_input("모델명", value=config.OLLAMA_MODEL)
        top_k = st.slider("top_k", min_value=3, max_value=15, value=8)
        temperature = st.slider("temperature", min_value=0.0, max_value=1.0, value=0.2)
        relevance_threshold = st.slider(
            "relevance_threshold",
            min_value=0.0,
            max_value=0.05,
            value=0.02,
            step=0.001,
            format="%.3f",
        )
        if st.button("대화 초기화"):
            st.session_state.messages = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        render_message(message)

    user_query = st.chat_input("질문을 입력하세요")
    if not user_query:
        return

    user_message = {"role": "user", "content": user_query}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    if not indexes_ready():
        assistant_message = {
            "role": "assistant",
            "content": INDEX_HELP,
            "sources": [],
            "is_relevant": False,
        }
        st.session_state.messages.append(assistant_message)
        render_message(assistant_message)
        return

    try:
        with st.spinner("답변 생성 중..."):
            pipeline = get_pipeline(
                model_name=model_name,
                top_k=top_k,
                relevance_threshold=relevance_threshold,
            )
            result = ask_pipeline(
                pipeline=pipeline,
                query=user_query,
                temperature=temperature,
            )
    except Exception as exc:
        result = {
            "answer": f"오류가 발생했습니다: {exc}",
            "sources": [],
            "is_relevant": False,
        }

    assistant_message = {
        "role": "assistant",
        "content": normalize_answer_for_ui(result.get("answer", "")),
        "sources": result.get("sources", []),
        "is_relevant": result.get("is_relevant", False),
    }
    st.session_state.messages.append(assistant_message)
    render_message(assistant_message)


if __name__ == "__main__":
    main()
