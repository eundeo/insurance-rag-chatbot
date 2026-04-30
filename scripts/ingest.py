import argparse
import json
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.parser.chunker import chunk_pages
from src.parser.pdf_parser import parse_pdf


logger = logging.getLogger(__name__)


def run_chunks_stage(pdf_path: Path) -> Path:
    pages = parse_pdf(pdf_path)
    chunks = chunk_pages(pages)

    config.ensure_dirs()
    output_path = config.PROCESSED_DIR / "chunks.jsonl"
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    total_chunks = len(chunks)
    avg_length = (
        sum(chunk["metadata"]["char_count"] for chunk in chunks) / total_chunks
        if total_chunks
        else 0
    )
    chunks_with_codes = sum(1 for chunk in chunks if chunk["metadata"]["codes"])

    logger.info("Total pages: %s", len(pages))
    logger.info("Total chunks: %s", total_chunks)
    logger.info("Average length: %.1f", avg_length)
    logger.info("Chunks with codes: %s", chunks_with_codes)
    logger.info("Wrote chunks to %s", output_path)

    return output_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run ingest stages.")
    parser.add_argument("--stage", required=True, choices=["chunks"], help="Ingest stage to run.")
    parser.add_argument("--pdf", required=True, help="Path to the source PDF.")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    try:
        run_chunks_stage(pdf_path)
    except FileNotFoundError:
        print(f"PDF file not found: {pdf_path}")
        print("Put the source PDF in data/raw/ or pass a valid path with --pdf.")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
