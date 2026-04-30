from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config


def _print_setting(name: str, value: object) -> None:
    print(f"{name}: {value}")


def main() -> None:
    print("Insurance RAG chatbot setup check")
    print("=" * 40)

    _print_setting("PDF_PATH", config.PDF_PATH)
    _print_setting("PROCESSED_DIR", config.PROCESSED_DIR)
    _print_setting("INDEX_DIR", config.INDEX_DIR)
    _print_setting("CHROMA_DIR", config.CHROMA_DIR)
    _print_setting("BM25_PATH", config.BM25_PATH)
    _print_setting("OLLAMA_BASE_URL", config.OLLAMA_BASE_URL)
    _print_setting("OLLAMA_MODEL", config.OLLAMA_MODEL)

    print("\nCreating required directories...")
    config.ensure_dirs()

    print("\nDirectory check result")
    for directory in (config.PROCESSED_DIR, config.INDEX_DIR, config.CHROMA_DIR):
        status = "OK" if directory.exists() and directory.is_dir() else "MISSING"
        print(f"- {directory}: {status}")

    print("\nPDF file check")
    if config.PDF_PATH.exists():
        print(f"- PDF file exists: {config.PDF_PATH}")
    else:
        print(f"- PDF file not found: {config.PDF_PATH}")
        print("  Put the source PDF in data/raw/ before running parser steps.")


if __name__ == "__main__":
    main()
