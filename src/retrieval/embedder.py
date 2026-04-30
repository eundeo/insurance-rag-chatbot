import logging
import os

from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "true")
        logger.info("Loading sentence-transformers model: %s", model_name)
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            logger.warning(
                "Local model load failed. Retrying from remote: %s",
                model_name,
            )
            self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        normalized_texts = [text if text else " " for text in texts]
        embeddings = self.model.encode(
            normalized_texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        text = query if query else " "
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()
