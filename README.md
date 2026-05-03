# insurance-rag-chatbot

보험 고시 PDF 문서를 기반으로 검색하고, 로컬 LLM으로 출처 기반 답변을 생성하는 RAG 챗봇입니다.

## 프로젝트 소개

이 프로젝트는 건강보험 고시 문서를 대상으로 다음 기능을 제공합니다.

- PDF 페이지 텍스트 추출
- 보험 고시 구조 기반 청킹
- BM25 키워드 검색
- ChromaDB 벡터 검색
- RRF 기반 Hybrid 검색
- Ollama 로컬 LLM 답변 생성
- Streamlit 웹 UI
- Smoke evaluation

## 시스템 구조

```text
PDF -> Chunk -> BM25 + Chroma -> Hybrid Retriever -> Ollama LLM -> CLI / Streamlit UI
```

## 설치 방법

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 환경 설정

```bash
cp .env.example .env
```

기본값:

```bash
PDF_PATH=data/raw/BZ202603053039374.pdf
PROCESSED_DIR=data/processed
INDEX_DIR=data/index
CHROMA_DIR=data/index/chroma
BM25_PATH=data/index/bm25.pkl
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
```

## 데이터 준비

PDF 원본 파일을 `data/raw/`에 넣습니다. `data/raw/`, `data/processed/`, `data/index/`는 Git에 커밋하지 않습니다.

## 실행 순서

초기 설정 확인:

```bash
python scripts/check_setup.py
```

청킹:

```bash
python scripts/ingest.py --stage chunks --pdf data/raw/BZ202603053039374.pdf
```

BM25 인덱스 생성:

```bash
python scripts/build_bm25.py
```

Chroma 인덱스 생성:

```bash
python scripts/build_chroma.py
```

Ollama 실행:

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

CLI 실행:

```bash
python scripts/cli.py
```

Streamlit UI 실행:

```bash
streamlit run src/ui/streamlit_app.py
```

## 검색 스크립트

BM25 검색:

```bash
python scripts/search_bm25.py --query "재진 진찰료" --top-k 5
```

Chroma 검색:

```bash
python scripts/search_chroma.py --query "치과의원 재진 진찰료 야간 가산" --top-k 5
```

Hybrid 검색:

```bash
python scripts/search_hybrid.py --query "재진 진찰료 야간 가산" --top-k 8
```

## 평가

Smoke evaluation:

```bash
python scripts/eval.py
```

평가 데이터는 `eval/smoke_qa.jsonl`에 있으며, 코드 기반 질의 5개와 의미 기반 질의 5개로 구성됩니다.

측정 지표:

- `Recall@8`: top 8 결과 중 expected page가 포함된 chunk가 있는지
- `Page Accuracy`: top 1 결과 페이지가 expected page와 같거나 +-1 범위인지
- `Keyword Match`: top 8 결과 텍스트에 기대 키워드가 포함되는지

## 예시 질문

```text
재진 진찰료 야간 가산 규정 알려줘
AA222는 어떤 항목이야?
치과에서 장애인 재진 진찰료 가산은?
오늘 날씨 어때?
```

문서 밖 질문은 다음처럼 차단됩니다.

```text
이 질문은 제공된 보험 고시 문서와 직접 관련이 없어 답변할 수 없습니다.
```

## 제한 사항

- 법적 효력이 있는 판단을 제공하지 않습니다.
- 의학적 최종 판단을 대신하지 않습니다.
- OCR은 지원하지 않습니다.
- 멀티턴 질의 재작성은 지원하지 않습니다.
- 세션 영속화와 멀티 사용자 인증은 없습니다.
- Docker/클라우드 배포는 포함하지 않습니다.

## 개발 로드맵

- M1: PDF parsing, hierarchical chunking
- M2: BM25, Chroma, Hybrid retrieval
- M3: Ollama RAG pipeline
- M4: Streamlit UI
- M5: Evaluation and documentation
