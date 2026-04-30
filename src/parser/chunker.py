import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)

VOLUME_RE = re.compile(r"제\d+편[^\n]*")
PART_RE = re.compile(r"제\d+부[^\n]*")
CHAPTER_RE = re.compile(r"제\d+장[^\n]*")
CODE_RE = re.compile(r"\b[A-Z]{2}\d+\b|[가-힣]-\d+")
SECTION_MAX_CHARS = 80

TARGET_CHARS = 1000
MIN_CHARS = 600
MAX_CHARS = 1200
OVERLAP_CHARS = 150


@dataclass
class Segment:
    text: str
    page_no: int
    metadata: dict[str, Any]


def extract_codes(text: str) -> list[str]:
    codes = []
    seen = set()
    for match in CODE_RE.findall(text):
        if match not in seen:
            codes.append(match)
            seen.add(match)
    return codes


def detect_headers(text: str, state: dict[str, str | None]) -> dict[str, str | None]:
    next_state = dict(state)

    if match := VOLUME_RE.search(text):
        next_state["volume"] = match.group(0).strip()
        next_state["part"] = None
        next_state["chapter"] = None
        next_state["section"] = None

    if match := PART_RE.search(text):
        next_state["part"] = match.group(0).strip()
        next_state["chapter"] = None
        next_state["section"] = None

    if match := CHAPTER_RE.search(text):
        next_state["chapter"] = match.group(0).strip()
        next_state["section"] = None

    if section := _extract_section_header(text):
        next_state["section"] = section

    return next_state


def _extract_section_header(text: str) -> str | None:
    candidate = " ".join(text.strip().split())
    if not candidate or len(candidate) > SECTION_MAX_CHARS:
        return None

    if _looks_like_body_sentence(candidate):
        return None

    if re.fullmatch(r"\[산정지침\]", candidate):
        return candidate

    if re.fullmatch(r"제\d+절\s+.+", candidate):
        return candidate

    if re.fullmatch(r"[가-힣]-\d+[가-힣A-Za-z0-9\s･ㆍ·(),/-]*", candidate):
        return _strip_english_tail(candidate)

    if re.fullmatch(r"[가-힣]\.\s*[가-힣A-Za-z0-9\s･ㆍ·()/-]{1,40}", candidate):
        return _strip_english_tail(candidate)

    if _has_short_title_code(candidate):
        return candidate

    return None


def _looks_like_body_sentence(text: str) -> bool:
    if text.count(",") + text.count("，") >= 2:
        return True
    if text.count("(") + text.count(")") >= 6:
        return True
    if re.search(r"(한다|한다\.|하며|하고|하되|경우|따라|의하여|제외|산정하지)", text):
        return True
    return False


def _strip_english_tail(text: str) -> str:
    return re.sub(r"\s+[A-Za-z][A-Za-z\s/()-]*$", "", text).strip()


def _has_short_title_code(text: str) -> bool:
    has_code = bool(CODE_RE.search(text) or re.search(r"\b\d{5}\b", text))
    has_code_heading = bool(re.search(r"분류번호|코\s*드|분\s*류|점\s*수", text))
    if not has_code or not has_code_heading:
        return False
    return not re.search(r"[.!?。]", text)


def chunk_pages(
    pages: list[dict],
    target_chars: int = TARGET_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[dict]:
    segments = _build_segments(pages)
    chunks = _segments_to_chunks(segments, target_chars, overlap_chars)

    avg_length = (
        sum(chunk["metadata"]["char_count"] for chunk in chunks) / len(chunks)
        if chunks
        else 0
    )
    logger.info("Total pages: %s", len(pages))
    logger.info("Total chunks: %s", len(chunks))
    logger.info("Average chunk length: %.1f", avg_length)
    logger.info(
        "Chunks with codes: %s",
        sum(1 for chunk in chunks if chunk["metadata"]["codes"]),
    )

    return chunks


def _build_segments(pages: list[dict]) -> list[Segment]:
    state: dict[str, str | None] = {
        "volume": None,
        "part": None,
        "chapter": None,
        "section": None,
    }
    segments: list[Segment] = []

    for page in pages:
        page_no = int(page["page_no"])
        text = page.get("text", "")
        for line in _iter_text_units(text):
            state = detect_headers(line, state)
            segments.append(
                Segment(
                    text=line,
                    page_no=page_no,
                    metadata=deepcopy(state),
                )
            )

    return segments


def _iter_text_units(text: str) -> list[str]:
    units = []
    buffer = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if buffer:
                units.append(" ".join(buffer))
                buffer = []
            continue

        if _looks_like_header(line) and buffer:
            units.append(" ".join(buffer))
            buffer = []

        buffer.append(line)

        if _looks_like_header(line):
            units.append(" ".join(buffer))
            buffer = []

    if buffer:
        units.append(" ".join(buffer))

    return [unit for unit in units if unit.strip()]


def _looks_like_header(text: str) -> bool:
    return bool(
        VOLUME_RE.match(text)
        or PART_RE.match(text)
        or CHAPTER_RE.match(text)
        or _extract_section_header(text)
    )


def _segments_to_chunks(
    segments: list[Segment],
    target_chars: int,
    overlap_chars: int,
) -> list[dict]:
    chunks: list[dict] = []
    current_text = ""
    current_meta: dict[str, Any] | None = None
    page_start: int | None = None
    page_end: int | None = None

    def flush() -> None:
        nonlocal current_text, current_meta, page_start, page_end
        text = current_text.strip()
        if not text or current_meta is None or page_start is None or page_end is None:
            return

        chunks.append(
            _make_chunk(
                chunk_no=len(chunks) + 1,
                text=text,
                metadata=current_meta,
                page_start=page_start,
                page_end=page_end,
            )
        )
        current_text = ""
        current_meta = None
        page_start = None
        page_end = None

    for segment in segments:
        segment_text = segment.text.strip()
        if not segment_text:
            continue

        if current_meta is None:
            current_meta = deepcopy(segment.metadata)
            page_start = segment.page_no
            page_end = segment.page_no

        projected_length = len(_join_text(current_text, segment_text))
        metadata_changed = _header_metadata(segment.metadata) != _header_metadata(current_meta)

        if metadata_changed and len(current_text) >= MIN_CHARS:
            flush()
            current_meta = deepcopy(segment.metadata)
            page_start = segment.page_no
            page_end = segment.page_no
        elif (
            metadata_changed
            and segment.metadata.get("section")
            and not current_meta.get("section")
        ):
            current_meta = deepcopy(segment.metadata)
        elif metadata_changed and len(current_text) < 200:
            current_meta = deepcopy(segment.metadata)
        elif projected_length > target_chars and len(current_text) >= MIN_CHARS:
            overlap = _tail_overlap(current_text, overlap_chars)
            flush()
            current_meta = deepcopy(segment.metadata)
            current_text = overlap
            page_start = segment.page_no
            page_end = segment.page_no

        current_text = _join_text(current_text, segment_text)
        page_end = segment.page_no

        while len(current_text) > MAX_CHARS:
            split_text = current_text[:target_chars].strip()
            remainder = current_text[target_chars - overlap_chars :].strip()
            chunks.append(
                _make_chunk(
                    chunk_no=len(chunks) + 1,
                    text=split_text,
                    metadata=current_meta,
                    page_start=page_start or segment.page_no,
                    page_end=page_end or segment.page_no,
                )
            )
            current_text = remainder
            page_start = segment.page_no

    flush()
    return chunks


def _join_text(left: str, right: str) -> str:
    if not left:
        return right
    return f"{left}\n\n{right}"


def _tail_overlap(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0:
        return ""
    return text.strip()[-overlap_chars:]


def _header_metadata(metadata: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        metadata.get("volume"),
        metadata.get("part"),
        metadata.get("chapter"),
        metadata.get("section"),
    )


def _make_chunk(
    chunk_no: int,
    text: str,
    metadata: dict[str, Any],
    page_start: int,
    page_end: int,
) -> dict:
    codes = extract_codes(text)
    return {
        "id": f"ch_{chunk_no:06d}",
        "text": text,
        "metadata": {
            "page_start": page_start,
            "page_end": page_end,
            "volume": metadata.get("volume"),
            "part": metadata.get("part"),
            "chapter": metadata.get("chapter"),
            "section": metadata.get("section"),
            "codes": codes,
            "char_count": len(text),
        },
    }
