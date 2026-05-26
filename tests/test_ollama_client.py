import requests

from src.llm.ollama_client import (
    OLLAMA_MODEL_ERROR,
    OLLAMA_SERVER_ERROR,
    OllamaClient,
)


class FakeResponse:
    def __init__(self, ok=True, payload=None):
        self.ok = ok
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_generate_returns_response_text(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse(payload={"response": "문서 기반 답변"})

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient(base_url="http://localhost:11434", timeout=30)

    result = client.generate("prompt", temperature=0.1, max_tokens=256)

    assert result == "문서 기반 답변"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["json"]["stream"] is False
    assert captured["json"]["options"]["temperature"] == 0.1
    assert captured["json"]["options"]["num_predict"] == 256
    assert captured["timeout"] == 30


def test_generate_returns_server_message_on_connection_failure(monkeypatch):
    def fake_post(url, json, timeout):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient()

    assert client.generate("prompt") == OLLAMA_SERVER_ERROR


def test_generate_returns_model_message_on_error_response(monkeypatch):
    def fake_post(url, json, timeout):
        return FakeResponse(ok=False)

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient()

    assert client.generate("prompt") == OLLAMA_MODEL_ERROR
