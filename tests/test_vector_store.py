from src.retrieval.vector_store import ChromaVectorStore, _metadata_to_chroma


def sample_chunks() -> list[dict]:
    return [
        {
            "id": "ch_000001",
            "text": "치과의원 재진 진찰료 야간 가산",
            "metadata": {
                "page_start": 1,
                "page_end": 1,
                "section": "나. 재진 진찰료",
                "codes": ["AA157", "AA100"],
                "char_count": 20,
            },
        },
        {
            "id": "ch_000002",
            "text": "영상진단료 방사선 치료료",
            "metadata": {
                "page_start": 2,
                "page_end": 2,
                "section": "제3장 영상진단",
                "codes": [],
                "char_count": 14,
            },
        },
    ]


def test_chroma_vector_store_creation(tmp_path):
    store = ChromaVectorStore(tmp_path)

    assert store.collection_name == "insurance_notice_chunks"


def test_metadata_conversion_for_chroma():
    metadata = {
        "page_start": 1,
        "section": None,
        "codes": ["AA157", "가-1"],
        "nested": {"x": 1},
    }

    converted = _metadata_to_chroma(metadata)

    assert converted["page_start"] == 1
    assert converted["section"] == ""
    assert converted["codes"] == "AA157,가-1"
    assert converted["nested"] == "{'x': 1}"


def test_add_chunks_and_search_returns_results(tmp_path):
    store = ChromaVectorStore(tmp_path)
    store.reset_collection()
    store.add_chunks(sample_chunks(), [[1.0, 0.0], [0.0, 1.0]], batch_size=1)

    results = store.search([1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == "ch_000001"


def test_search_result_contains_required_fields(tmp_path):
    store = ChromaVectorStore(tmp_path)
    store.reset_collection()
    store.add_chunks(sample_chunks(), [[1.0, 0.0], [0.0, 1.0]])

    result = store.search([1.0, 0.0], top_k=1)[0]

    assert "id" in result
    assert "text" in result
    assert "metadata" in result
    assert "score" in result
    assert result["metadata"]["codes"] == ["AA157", "AA100"]
