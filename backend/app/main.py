"""
FastAPI application entry point.

Creates the app, configures CORS middleware, registers the API router,
and initializes services on startup.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router, init_services
from app.config import (
    CORS_ALLOW_ORIGINS,
    DATABASE_URL,
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING_ENABLED,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG-Enabled ETL Platform",
    description="Secure data extraction, vectorization, and RAG with enterprise controls",
    version="1.0.0"
)

allowed_origins = [origin.strip() for origin in CORS_ALLOW_ORIGINS.split(",") if origin.strip()]
if not allowed_origins:
    allowed_origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(router)


@app.on_event("startup")
async def startup():
    """Initialize services on application startup"""
    # LangSmith tracing — must be set before any LangChain usage (Decision 2)
    if LANGSMITH_TRACING_ENABLED and LANGSMITH_API_KEY:
        import os as _os
        _os.environ["LANGCHAIN_TRACING_V2"] = "true"
        _os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
        _os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
        logger.info(f"🔍 LangSmith tracing enabled — project: {LANGSMITH_PROJECT}")
    else:
        logger.info("LangSmith tracing disabled (set LANGSMITH_TRACING=true to enable).")

    init_services(DATABASE_URL)
    logger.info("✅ Application started")


@app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
