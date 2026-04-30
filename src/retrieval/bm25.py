import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


CODE_RE = re.compile(r"[A-Za-z]{2}\d+|[가-힣]-\d+")
FALLBACK_TOKEN_RE = re.compile(r"[A-Za-z]{2}\d+|[가-힣A-Za-z0-9]+")


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self._kiwi: Any | None = None
        self._kiwi_failed = False

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []

        surface_tokens = self._fallback_tokenize(text)
        try:
            kiwi_tokens = self._tokenize_with_kiwi(text)
        except Exception:
            self._kiwi_failed = True
            kiwi_tokens = []

        return surface_tokens + kiwi_tokens

    def build(self):
        self.tokenized_corpus = [
            self.tokenize(chunk.get("text", "")) for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        return self

    def save(self, path: str | Path):
        if self.bm25 is None:
            raise ValueError("BM25 index has not been built. Call build() before save().")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "tokenized_corpus": self.tokenized_corpus,
                    "bm25": self.bm25,
                },
                file,
            )

    @classmethod
    def load(cls, path: str | Path):
        input_path = Path(path)
        with input_path.open("rb") as file:
            payload = pickle.load(file)

        retriever = cls(payload["chunks"])
        retriever.tokenized_corpus = payload["tokenized_corpus"]
        retriever.bm25 = payload["bm25"]
        return retriever

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.bm25 is None:
            raise ValueError("BM25 index has not been built or loaded.")

        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:top_k]

        results = []
        for index in ranked_indexes:
            chunk = self.chunks[index]
            results.append(
                {
                    "id": chunk["id"],
                    "score": float(scores[index]),
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                }
            )

        return results

    def _tokenize_with_kiwi(self, text: str) -> list[str]:
        if self._kiwi_failed:
            return self._fallback_tokenize(text)

        if self._kiwi is None:
            from kiwipiepy import Kiwi

            self._kiwi = Kiwi()

        tokens = []
        for token in self._kiwi.tokenize(text):
            form = token.form.strip().lower()
            if not form:
                continue
            if CODE_RE.fullmatch(form):
                tokens.append(form)
            elif token.tag.startswith("N") or token.tag in {"SL", "SN"}:
                tokens.append(form)

        return tokens

    def _fallback_tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in FALLBACK_TOKEN_RE.findall(text)]
