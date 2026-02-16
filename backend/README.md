# Backend README

The canonical full-stack documentation is at:
- `../README.md`

This folder contains the FastAPI backend (`app/`), utility scripts (`scripts/`), and API smoke tests (`tests/`).

## Run Backend Only

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Backend Health

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## Server Control Script

Use the helper script to run backend in the background and stop it safely:

```bash
cd backend
./scripts/server_ctl.sh start
./scripts/server_ctl.sh status
./scripts/server_ctl.sh logs
./scripts/server_ctl.sh stop
```

Do not run `uvicorn ... --reload` manually at the same time as `server_ctl.sh`.
The script now checks for existing listeners on the configured port and will refuse
to start a second instance.

## Latency Visibility / Tuning

- SSE timeline now includes a `routing` stage so pre-retrieval router time is visible.
- To disable LLM-based router calls (faster for legal workflows), set:

```bash
RAG_ENABLE_LLM_ROUTER=false
```

- To print retrieval diagnostics into backend logs (filters, retrieval question, top parts/sections), set:

```bash
RAG_DEBUG_RETRIEVAL_LOGS=true
```

- To look farther back for explicit legal refs when follow-up filters are empty, tune:

```bash
RAG_RETRIEVAL_REFERENCE_HISTORY_TURNS=12
```
