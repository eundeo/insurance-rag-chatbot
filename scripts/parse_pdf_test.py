import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.pdf_parser import parse_pdf


def _preview(text: str, limit: int = 300) -> str:
    return " ".join(text.split())[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Test PDF text extraction.")
    parser.add_argument("--pdf", required=True, help="Path to the source PDF.")
    parser.add_argument("--limit", type=int, default=5, help="Number of pages to print.")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)

    try:
        pages = parse_pdf(pdf_path)
    except FileNotFoundError:
        print(f"PDF file not found: {pdf_path}")
        print("Put the source PDF in data/raw/ or pass a valid path with --pdf.")
        return 0

    for page in pages[: args.limit]:
        text = page["text"]
        print("=" * 80)
        print(f"page_no: {page['page_no']}")
        print(f"text_length: {len(text)}")
        print(f"text_preview: {_preview(text)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
