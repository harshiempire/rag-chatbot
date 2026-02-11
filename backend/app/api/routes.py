"""
FastAPI route handlers for the RAG-Chatbot platform.

All HTTP endpoints are defined here as a thin shell that delegates
business logic to the appropriate engine/service classes.
"""

import json
import logging
import uuid
import time
from typing import Dict, List, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.database import VectorDatabase
from app.core.enums import DataClassification, DocumentStatus, LLMProvider
from app.core.schemas import (
    ChunkingConfig,
    DataSourceDefinition,
    RAGQuery,
    RAGResponse,
    ReviewDecision,
    VectorizationConfig,
)
from app.extraction.dynamic import DynamicExtractor
from app.extraction.pii import PIIDetector
from app.rag.engine import RAGEngine, RAG_PROMPT_K, RAG_RETRIEVE_K
from app.vectorization.engine import (
    DEFAULT_EMBEDDING_MODEL,
    VectorizationEngine,
    is_regulatory_document,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# ---------------------------------------------------------------------------
# Service singletons (lazy-initialized)
# ---------------------------------------------------------------------------

_vector_db: VectorDatabase = None
_rag_engine: RAGEngine = None
_pii_detector = PIIDetector()

# In-memory storage (replace with DB in production)
data_sources: Dict[str, DataSourceDefinition] = {}


def _sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """Serialize a structured SSE event envelope."""
    payload = json.dumps({"type": event_type, "data": data}, default=str, ensure_ascii=False)
    return f"data: {payload}\n\n"


def _build_source_event(result: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Convert similarity-search result row into a frontend-friendly source payload."""
    chunk_metadata = result.get("chunk_metadata", {}) or {}
    doc_metadata = result.get("doc_metadata", {}) or {}

    title = (
        doc_metadata.get("title")
        or chunk_metadata.get("section")
        or chunk_metadata.get("heading")
        or f"Source {index}"
    )
    snippet = (result.get("content") or "").strip().replace("\n", " ")
    if len(snippet) > 320:
        snippet = f"{snippet[:317]}..."

    return {
        "id": result.get("chunk_id"),
        "title": title,
        "snippet": snippet,
        "score": float(result.get("similarity", 0.0)),
        "metadata": {
            "classification": result.get("classification"),
            "chunk_metadata": chunk_metadata,
            "doc_metadata": doc_metadata,
        },
    }


def init_services(db_url: str):
    """Initialize services with the given database URL. Called from main.py."""
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDatabase(db_url)


def get_vector_db() -> VectorDatabase:
    global _vector_db
    if _vector_db is None:
        raise RuntimeError("VectorDatabase not initialized. Call init_services() first.")
    return _vector_db


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine(get_vector_db())
    return _rag_engine


# ---------------------------------------------------------------------------
# Source Management
# ---------------------------------------------------------------------------

@router.post("/sources/", response_model=DataSourceDefinition)
async def create_source(source: DataSourceDefinition):
    """Create data source with encrypted secrets"""
    source.encrypt_secrets()
    data_sources[source.id] = source
    logger.info(f"✅ Created source: {source.name} ({source.id})")
    return source


@router.get("/sources/")
async def list_sources():
    """List all data sources"""
    return {"sources": list(data_sources.values()), "total": len(data_sources)}


@router.get("/sources/{source_id}")
async def get_source(source_id: str):
    """Get data source (secrets masked)"""
    if source_id not in data_sources:
        raise HTTPException(status_code=404, detail="Source not found")
    return data_sources[source_id]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@router.post("/extract/{source_id}")
async def extract_data(source_id: str):
    """Extract data from source"""
    if source_id not in data_sources:
        raise HTTPException(status_code=404, detail="Source not found")

    source = data_sources[source_id]
    extractor = DynamicExtractor(source)
    vector_db = get_vector_db()

    try:
        df = extractor.extract()
        logger.info(f"📥 Extracted {len(df)} rows from {source.name}")

        # Store as documents
        doc_ids = []
        pending_review_count = 0
        for idx, row in df.iterrows():
            doc_id = str(uuid.uuid4())
            content = row.get('content', str(row.to_dict()))
            metadata = row.get('metadata', {}) if 'metadata' in row else {}

            # PII detection
            pii_detected = _pii_detector.detect(content)
            status = DocumentStatus.PENDING_REVIEW.value if pii_detected else DocumentStatus.EXTRACTED.value
            if pii_detected:
                pending_review_count += 1

            vector_db.store_document(
                doc_id, source_id, content, metadata,
                source.classification.value, status, pii_detected
            )
            doc_ids.append(doc_id)

        return {
            "message": f"Extracted {len(doc_ids)} documents",
            "document_ids": doc_ids,
            "pending_review": pending_review_count
        }

    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Document Review
# ---------------------------------------------------------------------------

@router.get("/documents/pending-review")
async def get_pending_reviews():
    """Get documents pending review"""
    docs = get_vector_db().get_pending_reviews()
    return {"pending_count": len(docs), "documents": docs}


@router.post("/documents/{document_id}/review")
async def review_document(document_id: str, decision: ReviewDecision):
    """Review and approve/reject document"""
    status = DocumentStatus.APPROVED.value if decision.decision == "approve" else DocumentStatus.REJECTED.value
    classification = decision.classification_override.value if decision.classification_override else None

    get_vector_db().update_document_status(document_id, status, decision.reviewer_id, classification)

    logger.info(f"✅ Document {document_id} {decision.decision}ed by {decision.reviewer_id}")
    return {"message": f"Document {decision.decision}ed", "document_id": document_id}


# ---------------------------------------------------------------------------
# Vectorization
# ---------------------------------------------------------------------------

@router.post("/documents/{document_id}/vectorize")
async def vectorize_document(
    document_id: str,
    chunking_config: ChunkingConfig = ChunkingConfig(),
    vectorization_config: VectorizationConfig = VectorizationConfig()
):
    """Vectorize approved document"""
    vector_db = get_vector_db()

    with vector_db.get_connection() as conn:
        cur = conn.cursor()

        cur.execute("""
            SELECT content, metadata, classification, source_id
            FROM documents WHERE id = %s AND status = 'approved'
        """, (document_id,))

        row = cur.fetchone()
        if not row:
            cur.close()
            raise HTTPException(status_code=404, detail="Approved document not found")

        content, metadata_json, classification, source_id = row

        metadata = metadata_json if isinstance(metadata_json, dict) else (json.loads(metadata_json) if metadata_json else {})

        # Vectorize
        vectorization = VectorizationEngine(vectorization_config.embedding_model.value)
        strategy = vectorization_config.chunking_strategy.value
        use_structure_aware = (
            strategy == "structure_aware" or
            (strategy == "auto" and is_regulatory_document(source_id=source_id, metadata=metadata))
        )
        if use_structure_aware:
            vector_chunks = vectorization.vectorize_regulation(
                content,
                metadata,
                use_structure_aware=True,
                max_chunk_size=chunking_config.chunk_size,
                min_chunk_size=chunking_config.min_chunk_size,
            )
        else:
            vector_chunks = vectorization.vectorize_document(
                content, metadata,
                {'chunk_size': chunking_config.chunk_size, 'chunk_overlap': chunking_config.chunk_overlap}
            )

        # Store chunks
        for chunk in vector_chunks:
            vector_db.store_vector_chunk(
                chunk['id'], document_id, chunk['content'], chunk['embedding'],
                chunk['metadata'], chunk['chunk_index'], classification
            )

        # Update status
        cur.execute("UPDATE documents SET status = 'vectorized' WHERE id = %s", (document_id,))
        conn.commit()
        cur.close()

    logger.info(f"✅ Vectorized document {document_id}: {len(vector_chunks)} chunks")
    return {"message": "Document vectorized", "document_id": document_id, "num_chunks": len(vector_chunks)}


@router.post("/documents/publish")
async def publish_documents(document_ids: List[str]):
    """Publish vectorized documents for RAG"""
    vector_db = get_vector_db()

    with vector_db.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE documents SET status = 'published'
            WHERE id = ANY(%s) AND status = 'vectorized'
            RETURNING id
        """, (document_ids,))
        published = [str(row[0]) for row in cur.fetchall()]
        conn.commit()
        cur.close()

    logger.info(f"✅ Published {len(published)} documents")
    return {"message": f"Published {len(published)} documents", "published_ids": published}


# ---------------------------------------------------------------------------
# RAG Query
# ---------------------------------------------------------------------------

@router.post("/rag/query", response_model=RAGResponse)
async def rag_query_endpoint(query: RAGQuery, user_id: str = "user-123"):
    """Query RAG system with security controls"""
    try:
        response = get_rag_engine().query(query, user_id)
        logger.info(f"✅ RAG query by {user_id}: {query.question[:50]}...")
        return response
    except HTTPException as e:
        logger.error(f"❌ RAG query failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"❌ RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/query/stream")
async def rag_query_stream_endpoint(query: RAGQuery, user_id: str = "user-123"):
    """Streaming RAG response (text/plain) for LOCAL and OPENROUTER providers."""
    if query.llm_provider not in {LLMProvider.LOCAL, LLMProvider.OPENROUTER}:
        raise HTTPException(
            status_code=400,
            detail="Streaming is only supported for LOCAL and OPENROUTER providers.",
        )

    engine = get_rag_engine()

    classifications = query.classification_filter or [DataClassification.PUBLIC]
    engine._validate_llm_access(query.llm_provider, classifications)

    if engine.vectorization is None:
        engine.vectorization = VectorizationEngine(DEFAULT_EMBEDDING_MODEL)
    embed_start = time.perf_counter()
    query_embedding = engine.get_query_embedding(query.question)
    embed_ms = round((time.perf_counter() - embed_start) * 1000, 2)

    search_start = time.perf_counter()
    results = engine.vector_db.similarity_search(
        query_embedding,
        [c.value for c in classifications],
        min(max(query.top_k, 1), RAG_RETRIEVE_K),
        query.min_similarity
    )
    search_ms = round((time.perf_counter() - search_start) * 1000, 2)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    max_class = engine._get_max_classification([r["classification"] for r in results])

    prompt, prompt_context_count = engine._build_prompt(query.question, results)

    engine.vector_db.log_query(
        user_id, query.question, query.llm_provider.value, max_class.value, True,
        details={
            "retrieved_count": len(results),
            "prompt_context_count": prompt_context_count,
            "timings_ms": {"embed": embed_ms, "search": search_ms},
            "prompt_k": RAG_PROMPT_K,
            "retrieve_k": RAG_RETRIEVE_K,
        }
    )

    if query.llm_provider == LLMProvider.LOCAL:
        token_stream = engine._ollama_stream(prompt)
    else:
        token_stream = engine._openrouter_stream(prompt, temperature=query.temperature)

    return StreamingResponse(token_stream, media_type="text/plain")


@router.post("/rag/query/stream/events")
async def rag_query_stream_events_endpoint(query: RAGQuery, user_id: str = "user-123"):
    """
    Structured SSE streaming endpoint for frontend timeline/status rendering.
    Event envelope:
      {"type": "...", "data": {...}}
    """
    if query.llm_provider not in {LLMProvider.LOCAL, LLMProvider.OPENROUTER}:
        raise HTTPException(
            status_code=400,
            detail="Structured streaming is only supported for LOCAL and OPENROUTER providers.",
        )

    engine = get_rag_engine()
    classifications = query.classification_filter or [DataClassification.PUBLIC]
    engine._validate_llm_access(query.llm_provider, classifications)

    def event_generator():
        total_start = time.perf_counter()
        timings_ms: Dict[str, float] = {}
        try:
            if engine.vectorization is None:
                engine.vectorization = VectorizationEngine(DEFAULT_EMBEDDING_MODEL)

            yield _sse_event(
                "status",
                {
                    "stage": "embedding",
                    "state": "start",
                    "label": "Generating query embedding",
                },
            )
            embed_start = time.perf_counter()
            query_embedding = engine.get_query_embedding(query.question)
            embed_ms = round((time.perf_counter() - embed_start) * 1000, 2)
            timings_ms["embed"] = embed_ms
            yield _sse_event(
                "status",
                {
                    "stage": "embedding",
                    "state": "done",
                    "label": "Embedding complete",
                    "meta": {"duration_ms": embed_ms},
                },
            )

            retrieval_k = min(max(query.top_k, 1), RAG_RETRIEVE_K)
            yield _sse_event(
                "status",
                {
                    "stage": "retrieval",
                    "state": "start",
                    "label": "Searching vector index",
                    "meta": {"retrieve_k": retrieval_k, "min_similarity": query.min_similarity},
                },
            )
            search_start = time.perf_counter()
            results = engine.vector_db.similarity_search(
                query_embedding,
                [c.value for c in classifications],
                retrieval_k,
                query.min_similarity,
            )
            search_ms = round((time.perf_counter() - search_start) * 1000, 2)
            timings_ms["search"] = search_ms

            if not results:
                yield _sse_event(
                    "error",
                    {"code": "NO_RESULTS", "message": "No relevant documents found"},
                )
                yield _sse_event("done", {})
                return

            for idx, result in enumerate(results, start=1):
                yield _sse_event("source", _build_source_event(result, idx))

            yield _sse_event(
                "status",
                {
                    "stage": "retrieval",
                    "state": "done",
                    "label": "Relevant chunks retrieved",
                    "meta": {"duration_ms": search_ms, "retrieved_count": len(results)},
                },
            )

            yield _sse_event(
                "status",
                {
                    "stage": "prompt_build",
                    "state": "start",
                    "label": "Building prompt context",
                },
            )
            prompt_start = time.perf_counter()
            prompt, prompt_context_count = engine._build_prompt(query.question, results)
            prompt_ms = round((time.perf_counter() - prompt_start) * 1000, 2)
            timings_ms["prompt_build"] = prompt_ms
            yield _sse_event(
                "status",
                {
                    "stage": "prompt_build",
                    "state": "done",
                    "label": "Prompt ready",
                    "meta": {
                        "duration_ms": prompt_ms,
                        "prompt_context_count": prompt_context_count,
                        "prompt_k": RAG_PROMPT_K,
                    },
                },
            )

            yield _sse_event(
                "status",
                {
                    "stage": "generation",
                    "state": "start",
                    "label": "Generating answer",
                },
            )

            llm_start = time.perf_counter()
            answer_parts: List[str] = []
            if query.llm_provider == LLMProvider.LOCAL:
                stream_iter = engine._ollama_stream(prompt)
                error_code = "LOCAL_LLM_ERROR"
            else:
                stream_iter = engine._openrouter_stream(prompt, temperature=query.temperature)
                error_code = "OPENROUTER_STREAM_ERROR"

            for token in stream_iter:
                if token and token.lstrip().startswith("[Error]"):
                    yield _sse_event("error", {"code": error_code, "message": token.strip()})
                    yield _sse_event(
                        "status",
                        {
                            "stage": "generation",
                            "state": "done",
                            "label": "Generation failed",
                        },
                    )
                    yield _sse_event("done", {})
                    return

                if token:
                    answer_parts.append(token)
                    yield _sse_event("token", {"text": token})

            llm_ms = round((time.perf_counter() - llm_start) * 1000, 2)
            timings_ms["llm"] = llm_ms

            answer = "".join(answer_parts).strip()
            total_ms = round((time.perf_counter() - total_start) * 1000, 2)
            timings_ms["total"] = total_ms

            max_class = engine._get_max_classification([r["classification"] for r in results])

            engine.vector_db.log_query(
                user_id,
                query.question,
                query.llm_provider.value,
                max_class.value,
                True,
                details={
                    "retrieved_count": len(results),
                    "prompt_context_count": prompt_context_count,
                    "timings_ms": timings_ms,
                    "prompt_k": RAG_PROMPT_K,
                    "retrieve_k": RAG_RETRIEVE_K,
                },
            )

            yield _sse_event(
                "status",
                {
                    "stage": "generation",
                    "state": "done",
                    "label": "Generation complete",
                    "meta": {"duration_ms": llm_ms},
                },
            )
            yield _sse_event(
                "final",
                {
                    "answer": answer,
                    "timings_ms": timings_ms,
                    "retrieved_count": len(results),
                    "prompt_context_count": prompt_context_count,
                },
            )
            yield _sse_event("done", {})
        except Exception as e:
            logger.error(f"❌ Structured stream failed: {e}")
            yield _sse_event("error", {"code": "STREAM_FAILURE", "message": str(e)})
            yield _sse_event("done", {})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@router.post("/workflow/complete")
async def complete_workflow(
    source_id: str,
    auto_approve_public: bool = True,
    chunking_config: ChunkingConfig = ChunkingConfig(),
    vectorization_config: VectorizationConfig = VectorizationConfig()
):
    """Complete workflow: Extract → Review → Vectorize → Publish"""
    workflow_id = str(uuid.uuid4())
    vector_db = get_vector_db()

    try:
        # Step 1: Extract
        logger.info(f"[{workflow_id}] Step 1: Extracting")
        extract_result = await extract_data(source_id)
        doc_ids = extract_result['document_ids']

        # Step 2: Auto-approve public docs without PII
        with vector_db.get_connection() as conn:
            cur = conn.cursor()

            if auto_approve_public:
                cur.execute("""
                    UPDATE documents SET status = 'approved'
                    WHERE id = ANY(%s) AND classification = 'public' AND pii_detected = FALSE
                """, (doc_ids,))
                conn.commit()

            # Step 3: Get approved
            cur.execute("SELECT id FROM documents WHERE id = ANY(%s) AND status = 'approved'", (doc_ids,))
            approved_ids = [str(row[0]) for row in cur.fetchall()]
            cur.close()

        # Step 4: Vectorize
        logger.info(f"[{workflow_id}] Step 2: Vectorizing {len(approved_ids)} docs")
        for doc_id in approved_ids:
            await vectorize_document(doc_id, chunking_config, vectorization_config)

        # Step 5: Publish
        await publish_documents(approved_ids)

        logger.info(f"[{workflow_id}] ✅ Workflow complete")
        return {
            "workflow_id": workflow_id,
            "extracted": len(doc_ids),
            "approved": len(approved_ids),
            "pending_review": len(doc_ids) - len(approved_ids)
        }

    except Exception as e:
        logger.error(f"[{workflow_id}] ❌ Workflow failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Stats & Audit
# ---------------------------------------------------------------------------

@router.get("/stats")
async def get_stats():
    """Get platform statistics"""
    return get_vector_db().get_stats()


@router.get("/audit")
async def get_audit_log(limit: int = 100):
    """Get audit log"""
    vector_db = get_vector_db()

    with vector_db.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT timestamp, user_id, query, llm_provider, classification, success
            FROM audit_log ORDER BY timestamp DESC LIMIT %s
        """, (limit,))

        logs = [{
            'timestamp': row[0].isoformat(),
            'user_id': row[1],
            'query': row[2],
            'llm_provider': row[3],
            'classification': row[4],
            'success': row[5]
        } for row in cur.fetchall()]

        cur.close()

    return {"logs": logs, "total": len(logs)}
