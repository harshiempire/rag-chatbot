"""
FastAPI application entry point.

Creates the app, configures CORS middleware, registers the API router,
and initializes services on startup.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router, init_services
from app.config import DATABASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG-Enabled ETL Platform",
    description="Secure data extraction, vectorization, and RAG with enterprise controls",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(router)


@app.on_event("startup")
async def startup():
    """Initialize services on application startup"""
    init_services(DATABASE_URL)
    logger.info("✅ Application started")


@app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
