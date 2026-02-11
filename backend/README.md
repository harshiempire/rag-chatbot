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
