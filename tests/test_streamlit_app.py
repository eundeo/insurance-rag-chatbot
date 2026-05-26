import importlib


def test_streamlit_app_importable():
    module = importlib.import_module("src.ui.streamlit_app")

    assert hasattr(module, "main")
    assert hasattr(module, "create_pipeline")


def test_ask_pipeline_calls_rag_pipeline():
    from src.ui.streamlit_app import ask_pipeline

    class MockPipeline:
        def __init__(self):
            self.calls = []

        def answer(self, query: str, temperature: float = 0.2):
            self.calls.append((query, temperature))
            return {
                "answer": "답변",
                "sources": [],
                "is_relevant": True,
            }

    pipeline = MockPipeline()
    result = ask_pipeline(pipeline, "재진 진찰료", temperature=0.1)

    assert result["answer"] == "답변"
    assert pipeline.calls == [("재진 진찰료", 0.1)]


def test_ollama_error_message_is_normalized():
    from src.llm.ollama_client import OLLAMA_MODEL_ERROR, OLLAMA_SERVER_ERROR
    from src.ui.streamlit_app import normalize_answer_for_ui

    assert normalize_answer_for_ui(OLLAMA_SERVER_ERROR) == (
        "Ollama 서버가 실행 중이 아닙니다. ollama serve를 실행하세요."
    )
    assert normalize_answer_for_ui(OLLAMA_MODEL_ERROR) == (
        "모델을 찾을 수 없습니다. ollama pull qwen2.5:7b-instruct"
    )
