from pathlib import Path
from typing import Any

import chromadb


DEFAULT_COLLECTION_NAME = "insurance_notice_chunks"


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def reset_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        batch_size: int = 64,
    ):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length.")

        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]
            self.collection.add(
                ids=[chunk["id"] for chunk in batch_chunks],
                documents=[chunk.get("text", "") for chunk in batch_chunks],
                metadatas=[
                    _metadata_to_chroma(chunk.get("metadata", {}))
                    for chunk in batch_chunks
                ],
                embeddings=batch_embeddings,
            )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        results = []
        for item_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            results.append(
                {
                    "id": item_id,
                    "score": 1 / (1 + float(distance)),
                    "text": document,
                    "metadata": _metadata_from_chroma(metadata or {}),
                }
            )
        return results


def _metadata_to_chroma(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    chroma_metadata: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            chroma_metadata[key] = ""
        elif key == "codes" and isinstance(value, list):
            chroma_metadata[key] = ",".join(str(item) for item in value)
        elif isinstance(value, (str, int, float, bool)):
            chroma_metadata[key] = value
        elif isinstance(value, list):
            chroma_metadata[key] = ",".join(str(item) for item in value)
        else:
            chroma_metadata[key] = str(value)
    return chroma_metadata


def _metadata_from_chroma(metadata: dict[str, Any]) -> dict[str, Any]:
    restored = dict(metadata)
    if "codes" in restored:
        codes = restored["codes"]
        if isinstance(codes, str):
            restored["codes"] = [code for code in codes.split(",") if code]
    return restored
