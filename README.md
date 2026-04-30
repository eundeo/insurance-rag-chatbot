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
