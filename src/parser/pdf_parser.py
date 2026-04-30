import importlib
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def _extract_with_pymupdf(pdf_path: Path, page_index: int) -> str:
    try:
        fitz = importlib.import_module("fitz")
        with fitz.open(pdf_path) as document:
            page = document.load_page(page_index)
            return page.get_text("text") or ""
    except Exception:
        logger.exception("PyMuPDF fallback failed on page %s", page_index + 1)
        return ""


def parse_pdf(pdf_path: str | Path) -> list[dict]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    pdfplumber = importlib.import_module("pdfplumber")
    pages: list[dict] = []

    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)

        for page_index, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if len(text.strip()) < 20:
                text = _extract_with_pymupdf(path, page_index)

            pages.append(
                {
                    "page_no": page_index + 1,
                    "text": text,
                }
            )

    success_pages = sum(1 for page in pages if page["text"].strip())
    empty_pages = total_pages - success_pages

    logger.info("Total pages: %s", total_pages)
    logger.info("Extracted pages: %s", success_pages)
    logger.info("Empty pages: %s", empty_pages)

    return pages
