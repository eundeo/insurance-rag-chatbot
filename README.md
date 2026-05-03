# insurance-rag-chatbot

Local RAG chatbot for Korean insurance and medical fee notice documents.

## Project Overview

This project builds a local Retrieval-Augmented Generation chatbot for Korean health insurance notice PDF documents. Users ask questions in natural language, the system retrieves relevant clauses, and a local LLM generates source-grounded answers.

## Alpha Goals

- PDF parsing
- Hierarchical chunking
- ChromaDB vector indexing
- BM25 keyword indexing
- Hybrid retrieval
- Ollama local LLM connection
- Streamlit chat UI
- Smoke evaluation

## Local Development Environment

- Python 3.10+
- Ollama installed locally for later LLM integration
- Source PDFs placed manually under `data/raw/`

This repository does not call external LLM APIs. The alpha target is local development only.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Create a local `.env` file from the example.

```bash
cp .env.example .env
```

Default values:

```bash
PDF_PATH=data/raw/BZ202603053039374.pdf
PROCESSED_DIR=data/processed
INDEX_DIR=data/index
CHROMA_DIR=data/index/chroma
BM25_PATH=data/index/bm25.pkl
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
```

## Check Initial Setup

Run:

```bash
python scripts/check_setup.py
```

The script prints configuration values, creates required processed/index directories, checks directory existence, and reports whether the configured PDF file exists.

## PDF Parser Smoke Test

Place the source PDF under `data/raw/`, then run:

```bash
python scripts/parse_pdf_test.py --pdf data/raw/BZ202603053039374.pdf --limit 5
```

The script extracts page-level text and prints `page_no`, `text_length`, and a short `text_preview` for the first pages.

## BM25 Keyword Search

Build the BM25 index from `data/processed/chunks.jsonl`:

```bash
python scripts/build_bm25.py
```

Run a keyword search:

```bash
python scripts/search_bm25.py --query "재진 진찰료"
```

## Chroma Vector Search

Build the Chroma index from `data/processed/chunks.jsonl`:

```bash
python scripts/build_chroma.py
```

Run a semantic search:

```bash
python scripts/search_chroma.py --query "치과의원 재진 진찰료 야간 가산" --top-k 5
```

The first run may take time because the `BAAI/bge-m3` embedding model must be downloaded. `data/index/chroma` is ignored by Git and should not be committed.

## Hybrid Search

Hybrid search requires both BM25 and Chroma indexes:

```bash
python scripts/build_bm25.py
python scripts/build_chroma.py
python scripts/search_hybrid.py --query "재진 진찰료 야간 가산" --top-k 8
```

## Local RAG CLI

Start Ollama and pull the local model:

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

Prepare indexes:

```bash
python scripts/build_bm25.py
python scripts/build_chroma.py
```

Run the CLI:

```bash
python scripts/cli.py
```

Example questions:

```text
재진 진찰료 야간 가산 규정 알려줘
AA222는 어떤 항목이야?
치과에서 장애인 재진 진찰료 가산은?
오늘 날씨 어때?
```

For out-of-document questions, the expected answer is:

```text
이 질문은 제공된 보험 고시 문서와 직접 관련이 없어 답변할 수 없습니다.
```

## Streamlit UI

Prepare indexes and start Ollama:

```bash
python scripts/build_bm25.py
python scripts/build_chroma.py
ollama serve
```

Run the web UI:

```bash
streamlit run src/ui/streamlit_app.py
```

Example questions:

```text
재진 진찰료 야간 가산 규정 알려줘
AA222는 어떤 항목이야?
오늘 날씨 어때?
```

## Development Roadmap

- M1-0: Project scaffold
- M1-1: PDF parser
- M1-2: Hierarchical chunker
- M1-3: ChromaDB vector index
- M1-4: BM25 keyword index
- M1-5: Hybrid retriever
- M1-6: Ollama local LLM client
- M1-7: Streamlit chat UI
- M1-8: Smoke evaluation

## Non-Goals

- OCR
- Multi-user authentication
- Cloud deployment
- External LLM API calls
- Docker/CI/CD
- Automated legal judgment
