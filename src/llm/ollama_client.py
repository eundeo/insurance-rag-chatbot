from __future__ import annotations

import requests


OLLAMA_SERVER_ERROR = "Ollama 서버가 실행 중이 아닙니다. ollama serve를 먼저 실행하세요."
OLLAMA_MODEL_ERROR = "Ollama 모델을 사용할 수 없습니다. ollama pull qwen2.5:7b-instruct를 확인하세요."


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b-instruct",
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError:
            return OLLAMA_SERVER_ERROR
        except requests.exceptions.Timeout:
            return OLLAMA_SERVER_ERROR
        except requests.exceptions.RequestException:
            return OLLAMA_MODEL_ERROR

        if not response.ok:
            return OLLAMA_MODEL_ERROR

        try:
            data = response.json()
        except ValueError:
            return OLLAMA_MODEL_ERROR

        return str(data.get("response", ""))
