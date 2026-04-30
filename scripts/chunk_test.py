import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.chunker import chunk_pages
from src.parser.pdf_parser import parse_pdf


def _preview(text: str, limit: int = 200) -> str:
    return " ".join(text.split())[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Test hierarchical chunking.")
    parser.add_argument("--pdf", required=True, help="Path to the source PDF.")
    parser.add_argument("--limit", type=int, default=10, help="Number of chunks to print.")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)

    try:
        pages = parse_pdf(pdf_path)
    except FileNotFoundError:
        print(f"PDF file not found: {pdf_path}")
        print("Put the source PDF in data/raw/ or pass a valid path with --pdf.")
        return 0

    chunks = chunk_pages(pages)

    for chunk in chunks[: args.limit]:
        metadata = chunk["metadata"]
        print("=" * 80)
        print(f"id: {chunk['id']}")
        print(f"page range: {metadata['page_start']}-{metadata['page_end']}")
        print(f"chapter: {metadata['chapter']}")
        print(f"section: {metadata['section']}")
        print(f"codes: {metadata['codes']}")
        print(f"text preview: {_preview(chunk['text'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
