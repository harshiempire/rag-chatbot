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
RAG_MAX_QUESTION_CHARS=2200
RAG_MAX_CHUNK_CHARS=1200
RAG_MAX_CONTEXT_CHARS=6000
RAG_IVFFLAT_PROBES=10
RAG_MIN_DISTINCT_SECTIONS=2
LEGAL_ROUTER_HISTORY_TURNS=4
LEGAL_ROUTER_HISTORY_CHARS_PER_TURN=320
CHAT_SESSION_MAX_MESSAGES=200
CHAT_SESSION_MAX_MESSAGE_CONTENT_CHARS=8000
CHAT_SESSION_MAX_PAYLOAD_BYTES=1048576
CHAT_SESSION_ID_MAX_CHARS=128
CHAT_SESSION_TITLE_MAX_CHARS=200
CHAT_SESSION_LIST_MAX_LIMIT=500
CHAT_SESSION_SUMMARY_DEFAULT_LIMIT=200
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

RAG query and chat-session endpoints require authentication.
Create an account and fetch an access token first:

```bash
curl -X POST "http://localhost:8000/api/v1/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"your-password"}'

curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"your-password"}'

# Copy access_token from login response
export ACCESS_TOKEN="<paste-access-token>"
```

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
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the capital requirements for banks under 12 CFR?",
    "llm_provider": "openrouter",
    "classification_filter": ["public"],
    "source_id": "ecfr-title-12-chapter-xii",
    "top_k": 8,
    "temperature": 0.7,
    "min_similarity": 0.2
  }'
```

4. Structured streaming endpoint used by frontend.

```bash
curl -N -X POST "http://localhost:8000/api/v1/rag/query/stream/events" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize key points from 12 CFR Chapter 12",
    "llm_provider": "local",
    "classification_filter": ["public"],
    "source_id": "ecfr-title-12-chapter-xii",
    "top_k": 5,
    "temperature": 0.7,
    "min_similarity": 0.2
  }'
```

5. Save and list chat sessions.

```bash
curl -X PUT "http://localhost:8000/api/v1/chat/sessions/demo-session-1" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "demo-session-1",
    "title": "Capital requirements follow-up",
    "llmProvider": "openrouter",
    "createdAt": 1738730000000,
    "updatedAt": 1738730100000,
    "messages": [
      {"id":"m1","role":"user","content":"What are Basel III capital buffers?","createdAt":1738730000000},
      {"id":"m2","role":"assistant","content":"Basel III includes capital conservation and countercyclical buffers.","createdAt":1738730010000}
    ]
  }'

curl -X GET "http://localhost:8000/api/v1/chat/sessions/summary?limit=50" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

Event format:

```json
{"type":"status|token|source|final|error|done","data":{}}
```

## Important Behavior

- `confidential`/`restricted` classification can only be queried with `local` provider.
- Chat sessions are persisted in backend PostgreSQL per authenticated user (`chat_sessions` table).
- Frontend sidebar now reads lightweight metadata from `GET /api/v1/chat/sessions/summary` and fetches full sessions on demand.
- RAG requests accept `session_id` and optional `chat_history`; when `chat_history` is omitted, history is resolved from the saved session if present.
- The session id `summary` is reserved and cannot be used as a chat session id.
- The local browser only stores provider preference (`rag_provider_pref_v1:<userId>`), not full chat history.
- Frontend provider selector supports `local`, `openai`, `anthropic`, `google`, and `openrouter`.
- Streaming endpoints support `local`, `openai`, and `openrouter`.

## Useful Backend Endpoints

- `GET /health`
- `POST /api/v1/auth/signup`
- `POST /api/v1/auth/login`
- `POST /api/v1/auth/refresh`
- `POST /api/v1/auth/logout`
- `GET /api/v1/auth/me`
- `GET /api/v1/chat/sessions?limit=&offset=`
- `GET /api/v1/chat/sessions/summary?limit=&offset=`
- `GET /api/v1/chat/sessions/{session_id}`
- `PUT /api/v1/chat/sessions/{session_id}`
- `DELETE /api/v1/chat/sessions/{session_id}`
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
