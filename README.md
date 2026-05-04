# insurance-rag-chatbot

건강보험 고시 PDF와 실손의료보험 약관 PDF를 기반으로 검색하고, 로컬 LLM으로 출처 기반 답변을 생성하는 RAG 챗봇입니다.

## 한눈에 보기

```text
PDF
  -> parse
  -> chunk
  -> BM25 + Chroma
  -> Hybrid Retriever
  -> Ollama LLM
  -> CLI / Streamlit UI
```

현재 구현 범위:

- PDF 페이지 텍스트 추출
- 고시/약관 문서 청킹
- BM25 키워드 검색
- ChromaDB 벡터 검색
- RRF 기반 Hybrid 검색
- Ollama 로컬 LLM 답변 생성
- Streamlit 챗 UI
- Smoke evaluation

## 폴더 구조

```text
data/
  raw/          # 원본 PDF 위치, Git 미추적
  processed/    # chunks.jsonl 생성 위치, Git 미추적
  index/        # BM25/Chroma 인덱스 위치, Git 미추적

src/
  parser/       # PDF parser, chunker
  retrieval/    # BM25, Chroma, Hybrid retriever
  llm/          # Ollama client, prompt
  rag/          # RAG pipeline
  ui/           # Streamlit app

scripts/        # 실행용 CLI 스크립트
tests/          # pytest
eval/           # smoke evaluation dataset
```

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

환경 파일을 생성합니다.

```bash
cp .env.example .env
```

기본 환경값:

```bash
PDF_PATH=data/raw/BZ202603053039374.pdf
PROCESSED_DIR=data/processed
INDEX_DIR=data/index
CHROMA_DIR=data/index/chroma
BM25_PATH=data/index/bm25.pkl
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
```

초기 설정 확인:

```bash
python scripts/check_setup.py
```

## 데이터 준비

RAG 대상 PDF를 `data/raw/`에 넣습니다.

현재 기준 입력 문서:

```text
data/raw/BZ202603053039374.pdf
data/raw/2.약관_신한 이지로운 실손의료보험(무배당)_20260401_0325.pdf
```

주의:

- `data/raw/`, `data/processed/`, `data/index/`는 Git에 커밋하지 않습니다.
- OCR은 지원하지 않습니다. PDF에서 텍스트가 추출 가능한 상태여야 합니다.

## 인덱스 생성

1. PDF 파싱 및 청킹:

```bash
python scripts/ingest.py --stage chunks --pdf data/raw/BZ202603053039374.pdf 'data/raw/2.약관_신한 이지로운 실손의료보험(무배당)_20260401_0325.pdf'
```

생성 파일:

```text
data/processed/chunks.jsonl
```

2. BM25 인덱스 생성:

```bash
python scripts/build_bm25.py
```

생성 파일:

```text
data/index/bm25.pkl
```

3. Chroma 벡터 인덱스 생성:

```bash
python scripts/build_chroma.py
```

생성 디렉토리:

```text
data/index/chroma/
```

최초 실행 시 `BAAI/bge-m3` 임베딩 모델 로드와 임베딩 생성 시간이 걸릴 수 있습니다.

## 로컬 LLM 준비

Ollama 서버를 실행합니다.

```bash
ollama serve
```

기본 모델을 내려받습니다.

```bash
ollama pull qwen2.5:7b-instruct
```

이 프로젝트는 외부 LLM API를 호출하지 않고, Ollama의 `/api/generate`를 통해 로컬 모델만 사용합니다.

## 실행

CLI 챗:

```bash
python scripts/cli.py
```

Streamlit UI:

```bash
streamlit run src/ui/streamlit_app.py
```

Streamlit에서 질문하면 답변과 함께 출처가 표시됩니다.

출처에는 다음 정보가 포함됩니다.

- 원본 PDF 파일명
- 페이지 범위
- section
- codes
- rrf_score
- 원문 preview

## 검색 확인

BM25 단독 검색:

```bash
python scripts/search_bm25.py --query "식도조루술 코드" --top-k 5
```

Chroma 단독 검색:

```bash
python scripts/search_chroma.py --query "N39.3 요실금 실손의료비 보상" --top-k 5
```

Hybrid 검색:

```bash
python scripts/search_hybrid.py --query "N39.3 요실금 실손의료비 보상가능" --top-k 5
```

예상되는 검색 예:

- `식도조루술` -> `Q2333 식도조루술`
- `N39.3 요실금` -> 실손 약관의 `제3조(보장종목별 보상내용)`

## 평가

Smoke evaluation 실행:

```bash
python scripts/eval.py
```

평가 데이터:

```text
eval/smoke_qa.jsonl
```

측정 지표:

- `Recall@8`: top 8 결과 중 기대 페이지가 포함된 chunk가 있는지
- `Page Accuracy`: top 1 결과 페이지가 기대 페이지와 같거나 +-1 범위인지
- `Keyword Match`: top 8 결과 텍스트에 기대 키워드가 포함되는지

## 테스트

```bash
pytest
```

## 예시 질문

```text
식도조루술의 코드를 알려줘
N39.3 요실금 실손의료비 보상가능 여부 알려줘
재진 진찰료 야간 가산 규정 알려줘
AA222는 어떤 항목이야?
오늘 날씨 어때?
```

문서 밖 질문은 LLM을 호출하지 않고 차단합니다.

```text
이 질문은 제공된 보험 고시 문서와 직접 관련이 없어 답변할 수 없습니다.
```

근거가 부족하거나 모델 답변이 문서 기반으로 유지되지 않으면 다음 답변을 반환합니다.

```text
제공된 문서 범위에서는 확인되지 않습니다.
```

## 현재 청킹 전략

고시 PDF:

- 수가/수술 코드 행을 가능한 한 독립 chunk로 분리합니다.
- 예: `자-233-1 Q2333 식도조루술`
- `item_no`, `fee_codes`, `codes`, `source_file` metadata를 저장합니다.

약관 PDF:

- `제n조(...)`, `제n관 ...` 형태를 section으로 인식합니다.
- `N39.3`처럼 소수점이 있는 진단코드를 보존합니다.
- `diagnosis_codes`, `codes`, `source_file` metadata를 저장합니다.

## 제한 사항

- 법적 효력이 있는 판단을 제공하지 않습니다.
- 의학적 최종 판단을 대신하지 않습니다.
- OCR은 지원하지 않습니다.
- 멀티턴 질의 재작성은 지원하지 않습니다.
- 문서 타입별 가중치 조정과 parent-child retrieval은 아직 구현하지 않았습니다.
- 세션 영속화와 멀티 사용자 인증은 없습니다.
- Docker/클라우드 배포는 포함하지 않습니다.

## 개발 로드맵

- M1: PDF parsing, hierarchical chunking
- M2: BM25, Chroma, Hybrid retrieval
- M3: Ollama RAG pipeline
- M4: Streamlit UI
- M5: Evaluation and documentation
- Next: 문서 타입별 검색 가중치, parent-child chunk, query decomposition
