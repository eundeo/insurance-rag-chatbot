import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")


def _path_from_env(name: str, default: str) -> Path:
    value = os.getenv(name, default)
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


PDF_PATH = _path_from_env("PDF_PATH", "data/raw/BZ202603053039374.pdf")
PROCESSED_DIR = _path_from_env("PROCESSED_DIR", "data/processed")
INDEX_DIR = _path_from_env("INDEX_DIR", "data/index")
CHROMA_DIR = _path_from_env("CHROMA_DIR", "data/index/chroma")
BM25_PATH = _path_from_env("BM25_PATH", "data/index/bm25.pkl")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
