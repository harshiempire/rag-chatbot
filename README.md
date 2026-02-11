# RAG-Chatbot (Backend + Frontend)

Full-stack RAG chat application with:
- `backend/`: FastAPI + PostgreSQL/pgvector + multi-provider RAG engine
- `frontend/`: React + Vite chat UI using structured SSE streaming

## Repository Layout

```text
RAG-Chatbot/
├── backend/
│   ├── app/                      # FastAPI app
│   ├── scripts/                  # Ingestion/vectorization helpers
│   ├── tests/                    # API smoke scripts
│   ├── docker-compose.postgres.yml
│   └── requirements.txt
└── frontend/
    ├── src/                      # React app
    └── package.json
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- Docker (for local PostgreSQL + pgvector)

## Quick Start

1. Start PostgreSQL + pgvector.

```bash
docker compose -f backend/docker-compose.postgres.yml up -d
```

2. Configure backend environment.

Create `backend/.env`:

```bash
ENCRYPTION_KEY=your-fernet-key
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_db

# Optional provider keys
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
OPENROUTER_API_KEY=

# Optional tuning
RAG_RETRIEVE_K=8
RAG_PROMPT_K=3
RAG_MAX_CHUNK_CHARS=1200
RAG_MAX_CONTEXT_CHARS=6000
OPENROUTER_MODEL=deepseek/deepseek-r1-0528:free
LOCAL_LLM_URL=http://localhost:11434/api/generate
LOCAL_LLM_MODEL=llama3.1:8b
```

3. Install and run backend.

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

4. Install and run frontend (new terminal).

```bash
cd frontend
npm install
npm run dev
```

Frontend defaults to `http://localhost:8000/api/v1`.
To override, create `frontend/.env.local`:

```bash
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

## Verify Services

- Backend health: `http://localhost:8000/health`
- Backend docs: `http://localhost:8000/docs`
- Frontend app: `http://localhost:5173`

## Backend API Workflow

1. Create a source.

```bash
curl -X POST "http://localhost:8000/api/v1/sources/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "eCFR Banking Regulations",
    "type": "ecfr_api",
    "config": {
      "title_number": 12,
      "date": "2024-01-01"
    },
    "classification": "public"
  }'
```

2. Run full pipeline (extract -> review/auto-approve -> vectorize -> publish).

```bash
curl -X POST "http://localhost:8000/api/v1/workflow/complete?source_id=<SOURCE_ID>"
```

3. Query RAG (non-streaming).

```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the capital requirements for banks under 12 CFR?",
    "llm_provider": "openrouter",
    "classification_filter": ["public"],
    "top_k": 8,
    "temperature": 0.7,
    "min_similarity": 0.2
  }'
```

4. Structured streaming endpoint used by frontend.

```bash
curl -N -X POST "http://localhost:8000/api/v1/rag/query/stream/events" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize key points from 12 CFR Chapter 12",
    "llm_provider": "local",
    "classification_filter": ["public"],
    "top_k": 5,
    "temperature": 0.7,
    "min_similarity": 0.2
  }'
```

Event format:

```json
{"type":"status|token|source|final|error|done","data":{}}
```

## Important Behavior

- Frontend provider selector currently supports `local` and `openrouter`.
- Streaming endpoints support `local` and `openrouter` only.
- `confidential`/`restricted` classification can only be queried with `local` provider.
- Chat history in frontend is stored in browser `localStorage` (key: `rag_chat_v1`).

## Useful Backend Endpoints

- `GET /health`
- `GET /api/v1/stats`
- `GET /api/v1/audit?limit=100`
- `GET /api/v1/documents/pending-review`
- `POST /api/v1/documents/{document_id}/review`
- `POST /api/v1/documents/{document_id}/vectorize`
- `POST /api/v1/documents/publish`

## Scripts

Run from `backend/` with backend virtual environment activated.

Ingest eCFR chapter batches:

```bash
python scripts/ingest_ecfr_chapter.py --title 12 --chapter XII --date 2024-01-01 --batch-size 5 --batch-index 0
```

Vectorize existing documents for a source:

```bash
python scripts/vectorize_documents.py --source-id <SOURCE_ID> --chunking-strategy auto --replace-existing
```

## Smoke Tests

Run from `backend/` while API is running:

```bash
python tests/test_query_local.py
python tests/test_query_openrouter.py
python tests/test_query_openai.py
python tests/test_query_gemini.py
```

Note: `tests/test_chunking_comparison.py` is currently out of date and not part of the runnable smoke suite.
