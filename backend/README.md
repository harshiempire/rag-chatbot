# RAG-Enabled ETL Platform
> Secure data extraction, vectorization, and Retrieval-Augmented Generation with enterprise controls.

---

## 📁 Project Structure

```
RAG-Chatbot/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # Environment config
│   ├── core/
│   │   ├── enums.py             # All enumerations
│   │   ├── schemas.py           # Pydantic models
│   │   ├── secrets.py           # SecretManager (AES-256)
│   │   └── database.py          # PostgreSQL + pgvector
│   ├── extraction/
│   │   ├── ecfr.py              # eCFR API extractor
│   │   ├── dynamic.py           # Multi-source extractor
│   │   └── pii.py               # PII detection
│   ├── vectorization/
│   │   ├── chunker.py           # Structure-aware chunking
│   │   └── engine.py            # Embedding generation
│   ├── rag/
│   │   └── engine.py            # RAG query engine (multi-LLM)
│   └── api/
│       └── routes.py            # All HTTP endpoints
├── tests/                       # Test scripts
├── scripts/                     # CLI utilities
├── requirements.txt
├── .env                         # Environment variables
└── README.md
```

---

## 📦 Installation

### 1. Clone & Create Virtual Environment

```bash
git clone <your-repo-url>
cd RAG-Chatbot

# Create venv with uv
uv venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 3. Setup PostgreSQL with pgvector

```bash
docker run -d \
  --name rag-postgres \
  -e POSTGRES_PASSWORD=yourpassword \
  -e POSTGRES_DB=rag_db \
  -p 5432:5432 \
  ankane/pgvector
```

### 4. Configure Environment

Create a `.env` file in the project root:

```bash
ENCRYPTION_KEY=your-fernet-key-here
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/rag_db
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
GEMINI_API_KEY=your-gemini-key
OPENROUTER_API_KEY=your-openrouter-key
RAG_RETRIEVE_K=8
RAG_PROMPT_K=3
```

---

## 🚀 Running the Application

```bash
# Activate venv
source .venv/bin/activate

# Start the server (from project root)
uvicorn app.main:app --reload --port 8000

# Access API docs
open http://localhost:8000/docs
```

---

## 📚 Complete Workflow Example

### Step 1: Create Data Source

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

### Step 2: Run Complete Workflow (Extract → Review → Vectorize → Publish)

```bash
curl -X POST "http://localhost:8000/api/v1/workflow/complete?source_id=<SOURCE_ID>"
```

### Step 3: Review Pending Documents

```bash
# Get pending reviews
curl "http://localhost:8000/api/v1/documents/pending-review"

# Approve a document
curl -X POST "http://localhost:8000/api/v1/documents/<DOC_ID>/review" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "<DOC_ID>",
    "reviewer_id": "reviewer@company.com",
    "decision": "approve"
  }'
```

### Step 4: Query RAG System

```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the capital requirements for banks under 12 CFR?",
    "llm_provider": "openrouter",
    "classification_filter": ["public"],
    "top_k": 8,
    "temperature": 0.7
  }'
```

`top_k` controls retrieval upper-bound (capped by `RAG_RETRIEVE_K`), while prompt context is capped by `RAG_PROMPT_K`.

---

## 🧪 Running Tests

Tests are standalone HTTP scripts that call the running API:

```bash
# Make sure the server is running first, then in another terminal:
source .venv/bin/activate

python tests/test_query_openrouter.py
python tests/test_query_local.py
python tests/test_query_openai.py
python tests/test_query_gemini.py
python tests/test_chunking_comparison.py
```

---

## 🔐 Security Features

| Feature | Description |
|---------|-------------|
| **Data Classification** | `public`, `internal`, `confidential`, `restricted` levels |
| **LLM Access Control** | Restricted/confidential data can **only** use local LLMs |
| **Secret Encryption** | All API keys & passwords encrypted with AES-256 (Fernet) |
| **PII Detection** | Auto-detects emails, SSNs, phone/credit card numbers |
| **Audit Trail** | Every RAG query logged with user, provider, and classification |

---

## 🏗️ Architecture

```
┌─────────────┐
│   eCFR API  │ ──┐
│  (Title 12) │   │
└─────────────┘   │
                  │
┌─────────────┐   │    ┌──────────────┐
│  REST APIs  │ ──┼───▶│  Extractors  │
│ (OAuth2)    │   │    └──────┬───────┘
└─────────────┘   │           │
                  │           ▼
┌─────────────┐   │    ┌──────────────┐
│ Databases   │ ──┘    │ PII Detector │
└─────────────┘        └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │    Review    │
                       │   Workflow   │
                       └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │ Vectorization│
                       │  (Embeddings)│
                       └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │  PostgreSQL  │
                       │  + pgvector  │
                       └──────┬───────┘
                              │
                              ▼
┌─────────────┐        ┌──────────────┐
│   OpenAI    │◀───────│  RAG Engine  │
│  Anthropic  │        │  (Security)  │
│   Gemini    │        └──────┬───────┘
│ OpenRouter  │               │
│  Local LLM  │               ▼
└─────────────┘        ┌──────────────┐
                       │  Audit Log   │
                       └──────────────┘
```

---

## 🎯 Key Benefits

- **Multi-LLM Support** — OpenAI, Anthropic, Google Gemini, OpenRouter, Local (Ollama)
- **Enterprise Compliance** — Review workflow with data classification enforcement
- **Secret Security** — All credentials encrypted at rest with AES-256
- **Vector Search** — Fast semantic similarity via PostgreSQL + pgvector
- **Streaming** — Real-time streaming for local LLM responses
- **Complete Audit** — Every query logged for compliance

---

## 🔄 Maintenance

```bash
# Backup database
pg_dump rag_db > backup.sql

# Check stats
curl "http://localhost:8000/api/v1/stats"

# View audit log
curl "http://localhost:8000/api/v1/audit?limit=100"

# Health check
curl "http://localhost:8000/health"
```

---

**Start extracting regulations and querying with AI!** 🚀
