from src.parser.chunker import chunk_pages, detect_headers, extract_codes


def test_detect_headers_updates_hierarchy_state():
    state = {"volume": None, "part": None, "chapter": None, "section": None}

    state = detect_headers("제1편 행위 급여 일반원칙", state)
    state = detect_headers("제2부 검사료", state)
    state = detect_headers("제3장 영상진단 및 방사선치료료", state)
    state = detect_headers("제1절 방사선 단순영상진단료", state)

    assert state["volume"] == "제1편 행위 급여 일반원칙"
    assert state["part"] == "제2부 검사료"
    assert state["chapter"] == "제3장 영상진단 및 방사선치료료"
    assert state["section"] == "제1절 방사선 단순영상진단료"


def test_extract_codes_deduplicates_english_and_korean_codes():
    text = "AA157 검사료와 AB123, AA157 및 가-1 항목을 확인한다. 가-1 중복."

    assert extract_codes(text) == ["AA157", "AB123", "가-1"]


def test_chunk_pages_splits_long_text_with_overlap():
    long_text = "제1장 기본\n" + ("AA157 청구 기준 설명입니다. " * 120)
    pages = [{"page_no": 1, "text": long_text}]

    chunks = chunk_pages(pages, target_chars=500, overlap_chars=50)

    assert len(chunks) > 1
    assert chunks[0]["id"] == "ch_000001"
    assert chunks[1]["id"] == "ch_000002"
    assert chunks[0]["metadata"]["char_count"] <= 1200


def test_chunk_pages_preserves_metadata():
    pages = [
        {
            "page_no": 88,
            "text": (
                "제1편 행위 급여\n"
                "제2부 검사료\n"
                "제3장 검체 검사료\n"
                "[산정지침]\n"
                "AA157 " + ("산정 기준 본문입니다. " * 40)
            ),
        }
    ]

    chunks = chunk_pages(pages)

    first = chunks[0]
    assert first["metadata"]["page_start"] == 88
    assert first["metadata"]["page_end"] == 88
    assert first["metadata"]["volume"] == "제1편 행위 급여"
    assert first["metadata"]["part"] == "제2부 검사료"
    assert first["metadata"]["chapter"] == "제3장 검체 검사료"
    assert first["metadata"]["section"] == "[산정지침]"
    assert first["metadata"]["codes"] == ["AA157"]


def test_section_metadata_ignores_long_body_sentence():
    state = {"volume": None, "part": None, "chapter": None, "section": "나. 재진 진찰료"}
    long_body = (
        "요-52) (라) 중환자실 입원료(요-53), 다만, AJ002, 19002는 제외 "
        "(마) 격리실 입원료(요-54) (바) 요양병원 임종실 입원 정액(요-30) "
        "전문병원 관리료 및 의료질평가 지원금을 산정하지 아니한다."
    )

    next_state = detect_headers(long_body, state)

    assert next_state["section"] == "나. 재진 진찰료"


def test_section_metadata_accepts_short_item_titles():
    state = {"volume": None, "part": None, "chapter": None, "section": None}

    assert (
        detect_headers("나. 재진 진찰료 Established Patient", state)["section"]
        == "나. 재진 진찰료"
    )
    assert (
        detect_headers("가-1 외래환자 진찰료 Outpatient Care", state)["section"]
        == "가-1 외래환자 진찰료"
    )


def test_section_metadata_ignores_short_code_body_fragment():
    state = {"volume": None, "part": None, "chapter": None, "section": "제1절 기본진료료"}

    next_state = detect_headers("VA800 사용, 소아전문 VA600 사용]", state)

    assert next_state["section"] == "제1절 기본진료료"
