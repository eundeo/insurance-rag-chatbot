import sys
from types import SimpleNamespace

import pytest

from src.parser.pdf_parser import parse_pdf


class FakePage:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self) -> str:
        return self._text


class FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeFitzPage:
    def get_text(self, option: str) -> str:
        assert option == "text"
        return "Fallback text extracted by PyMuPDF."


class FakeFitzDocument:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def load_page(self, page_index: int) -> FakeFitzPage:
        assert page_index == 0
        return FakeFitzPage()


def test_parse_pdf_raises_file_not_found_for_missing_path(tmp_path):
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        parse_pdf(missing_pdf)


def test_parse_pdf_returns_page_items(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")

    fake_pdfplumber = SimpleNamespace(
        open=lambda path: FakePdf(
            [
                FakePage("This is page one text with enough characters."),
                FakePage("This is page two text with enough characters."),
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_pdfplumber)

    result = parse_pdf(pdf_path)

    assert isinstance(result, list)
    assert result[0]["page_no"] == 1
    for page in result:
        assert "page_no" in page
        assert "text" in page


def test_parse_pdf_uses_pymupdf_fallback_for_short_text(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")

    fake_pdfplumber = SimpleNamespace(open=lambda path: FakePdf([FakePage("")]))
    fake_fitz = SimpleNamespace(open=lambda path: FakeFitzDocument())
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_pdfplumber)
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    result = parse_pdf(pdf_path)

    assert result == [{"page_no": 1, "text": "Fallback text extracted by PyMuPDF."}]
