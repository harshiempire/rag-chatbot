"""
FastAPI route handlers for the RAG-Chatbot platform.

All HTTP endpoints are defined here as a thin shell that delegates
business logic to the appropriate engine/service classes.
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Query, Response
from fastapi.responses import StreamingResponse

from app.config import (
    AUTH_COOKIE_SAMESITE,
    AUTH_COOKIE_SECURE,
    RAG_MAX_QUESTION_CHARS,
    REFRESH_COOKIE_NAME,
    REFRESH_TOKEN_EXPIRE_DAYS,
)
from app.core.auth import (
    AuthError,
    create_access_token,
    decode_access_token,
    hash_password,
    hash_refresh_token,
    issue_refresh_token,
    normalize_email,
    verify_password,
)
from app.core.database import VectorDatabase
from app.core.enums import DataClassification, DocumentStatus, LLMProvider
from app.core.schemas import (
    AuthMessageResponse,
    AuthTokenResponse,
    ChatSessionPayload,
    ChatSessionSummaryPayload,
    ChunkingConfig,
    DataSourceDefinition,
    RAGQuery,
    RAGResponse,
    ReviewDecision,
    UserLoginRequest,
    UserPublic,
    UserSignupRequest,
    VectorizationConfig,
)
from app.extraction.dynamic import DynamicExtractor
from app.extraction.pii import PIIDetector
from app.rag.engine import (
    RAG_PROMPT_K,
    RAG_RETRIEVE_K,
    RAGEngine,
)
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
CHAT_HISTORY_MAX_TURNS = 12
CHAT_HISTORY_MAX_CHARS_PER_TURN = 600
CHAT_HISTORY_MAX_TOTAL_CHARS = 2800
CHAT_SESSION_MAX_MESSAGES = int(os.getenv("CHAT_SESSION_MAX_MESSAGES", "200"))
CHAT_SESSION_MAX_MESSAGE_CONTENT_CHARS = int(
    os.getenv("CHAT_SESSION_MAX_MESSAGE_CONTENT_CHARS", "8000")
)
CHAT_SESSION_MAX_PAYLOAD_BYTES = int(
    os.getenv("CHAT_SESSION_MAX_PAYLOAD_BYTES", "1048576")
)
CHAT_SESSION_ID_MAX_CHARS = int(os.getenv("CHAT_SESSION_ID_MAX_CHARS", "128"))
CHAT_SESSION_TITLE_MAX_CHARS = int(os.getenv("CHAT_SESSION_TITLE_MAX_CHARS", "200"))
CHAT_SESSION_LIST_MAX_LIMIT = int(os.getenv("CHAT_SESSION_LIST_MAX_LIMIT", "500"))
CHAT_SESSION_SUMMARY_DEFAULT_LIMIT = int(
    os.getenv("CHAT_SESSION_SUMMARY_DEFAULT_LIMIT", "200")
)
RESERVED_CHAT_SESSION_IDS = {"summary"}

LEGAL_ROUTER_HISTORY_TURNS = int(os.getenv("LEGAL_ROUTER_HISTORY_TURNS", "4"))
LEGAL_ROUTER_HISTORY_CHARS_PER_TURN = int(
    os.getenv("LEGAL_ROUTER_HISTORY_CHARS_PER_TURN", "320")
)


def _validate_chat_session_id(session_id: str) -> None:
    if (session_id or "").strip().lower() in RESERVED_CHAT_SESSION_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Session id '{session_id}' is reserved and cannot be used.",
        )


def _extract_json_object(raw: str) -> Dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    parsed = json.loads(candidate)
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _format_router_history(history: List[Dict[str, str]]) -> str:
    selected = history[-LEGAL_ROUTER_HISTORY_TURNS :]
    if not selected:
        return "none"

    lines: List[str] = []
    for turn in selected:
        role = "user" if turn.get("role") == "user" else "assistant"
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        lines.append(
            f"{role}: {content[:LEGAL_ROUTER_HISTORY_CHARS_PER_TURN]}"
        )
    return "\n".join(lines) if lines else "none"


def _decide_query_mode_with_llm(
    engine: RAGEngine, query: RAGQuery, history: List[Dict[str, str]]
) -> Tuple[str, str]:
    router_prompt = (
        "You are a routing controller for an eCFR legal assistant.\n"
        "Choose exactly one mode for this request:\n"
        "- rag: use retrieval over legal/regulatory documents.\n"
        "- direct: no retrieval needed (greeting/chit-chat/meta/capabilities).\n"
        "- deny: refuse unsafe or disallowed requests.\n\n"
        "Policy:\n"
        "- This assistant provides legal information, not legal advice.\n"
        "- Use rag for legal interpretation, compliance, citations, or document-grounded follow-ups.\n"
        "- Use direct for social talk and capability/boundary clarification.\n"
        "- Use deny for requests enabling wrongdoing or for personalized legal advice that should be handled by a licensed attorney.\n"
        "- If uncertain, choose rag.\n\n"
        "Output STRICT JSON only:\n"
        '{"mode":"rag|direct|deny","reason":"short reason"}\n\n'
        f"User question:\n{query.question.strip()}\n\n"
        f"Recent conversation:\n{_format_router_history(history)}"
    )

    decision_query = query.model_copy(update={"temperature": 0.0})
    try:
        raw = engine._generate_answer(decision_query, router_prompt)
    except Exception as exc:
        logger.warning(f"Router decision failed, defaulting to rag: {exc}")
        return "rag", "router_error"

    parsed = _extract_json_object(raw)
    if not parsed:
        logger.warning("Router returned non-JSON output, defaulting to rag")
        return "rag", "router_parse_error"

    mode = str(parsed.get("mode", "rag")).strip().lower()
    if mode not in {"rag", "direct", "deny"}:
        mode = "rag"
    reason = str(parsed.get("reason", "")).strip()[:240]
    return mode, reason


def _build_non_rag_prompt(question: str, mode: str, reason: str) -> str:
    mode_instruction = (
        "Respond helpfully and briefly to the user."
        if mode == "direct"
        else (
            "Politely refuse the request and explain the boundary. "
            "Offer a safe legal-information alternative."
        )
    )
    return (
        "You are an eCFR legal assistant.\n"
        "Capabilities:\n"
        "- Help users understand legal/regulatory topics and answer document-grounded legal questions.\n"
        "- Explain what the assistant can and cannot do.\n\n"
        "Boundaries:\n"
        "- Do not provide personalized legal advice or representation.\n"
        "- Do not help with wrongdoing or evasion.\n"
        "- Be transparent about limits and suggest asking a licensed attorney when appropriate.\n\n"
        f"Routing decision: {mode}\n"
        f"Routing reason: {reason or 'n/a'}\n\n"
        f"{mode_instruction}\n\n"
        f"User message:\n{question.strip()}"
    )


def _empty_rag_response(answer: str, llm_provider: LLMProvider, llm_ms: float = 0.0) -> RAGResponse:
    total_ms = round(llm_ms, 2)
    return RAGResponse(
        answer=answer,
        sources=[],
        classification=DataClassification.PUBLIC,
        llm_provider=llm_provider,
        retrieved_count=0,
        prompt_context_count=0,
        total_ms=total_ms,
        timings_ms={
            "embed": 0.0,
            "search": 0.0,
            "prompt_build": 0.0,
            "llm": round(llm_ms, 2),
            "total": total_ms,
        },
    )


def _log_non_rag_query(
    engine: RAGEngine,
    user_id: str,
    query: RAGQuery,
    mode: str,
    reason: str,
    llm_ms: float = 0.0,
) -> None:
    try:
        engine.vector_db.log_query(
            user_id,
            query.question,
            query.llm_provider.value,
            DataClassification.PUBLIC.value,
            True,
            details={
                "route_mode": mode,
                "route_reason": reason,
                "retrieved_count": 0,
                "prompt_context_count": 0,
                "source_id": query.source_id,
                "timings_ms": {
                    "embed": 0.0,
                    "search": 0.0,
                    "prompt_build": 0.0,
                    "llm": round(llm_ms, 2),
                    "total": round(llm_ms, 2),
                },
                "prompt_k": RAG_PROMPT_K,
                "retrieve_k": 0,
            },
        )
    except Exception as exc:
        logger.warning(f"Failed to log non-RAG query: {exc}")


def _stream_from_provider(engine: RAGEngine, query: RAGQuery, prompt: str):
    if query.llm_provider == LLMProvider.LOCAL:
        return engine._ollama_stream(prompt)
    if query.llm_provider == LLMProvider.OPENROUTER:
        return engine._openrouter_stream(prompt, temperature=query.temperature)
    raise HTTPException(
        status_code=400,
        detail=f"Streaming not supported for provider {query.llm_provider.value}.",
    )


def _streaming_non_rag_response(engine: RAGEngine, query: RAGQuery, prompt: str) -> StreamingResponse:
    return StreamingResponse(_stream_from_provider(engine, query, prompt), media_type="text/plain")


def _sse_non_rag_response(
    engine: RAGEngine, query: RAGQuery, prompt: str, user_id: str, mode: str, reason: str
) -> StreamingResponse:
    def event_generator():
        total_start = time.perf_counter()
        answer_parts: List[str] = []
        try:
            yield _sse_event(
                "status",
                {
                    "stage": "generation",
                    "state": "start",
                    "label": "Generating response",
                },
            )
            stream_iter = _stream_from_provider(engine, query, prompt)
            error_code = (
                "LOCAL_LLM_ERROR"
                if query.llm_provider == LLMProvider.LOCAL
                else "OPENROUTER_STREAM_ERROR"
            )
            for token in stream_iter:
                if token and token.lstrip().startswith("[Error]"):
                    yield _sse_event("error", {"code": error_code, "message": token.strip()})
                    yield _sse_event("done", {})
                    return
                if token:
                    answer_parts.append(token)
                    yield _sse_event("token", {"text": token})

            llm_ms = round((time.perf_counter() - total_start) * 1000, 2)
            answer = "".join(answer_parts).strip()
            _log_non_rag_query(engine, user_id, query, mode, reason, llm_ms=llm_ms)
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
                    "timings_ms": {
                        "embed": 0.0,
                        "search": 0.0,
                        "prompt_build": 0.0,
                        "llm": llm_ms,
                        "total": llm_ms,
                    },
                    "retrieved_count": 0,
                    "prompt_context_count": 0,
                },
            )
            yield _sse_event("done", {})
        except Exception as exc:
            logger.error(f"❌ Non-RAG SSE stream failed: {exc}")
            yield _sse_event("error", {"code": "STREAM_FAILURE", "message": str(exc)})
            yield _sse_event("done", {})

    return StreamingResponse(event_generator(), media_type="text/event-stream")

    return None


def _sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """Serialize a structured SSE event envelope."""
    payload = json.dumps(
        {"type": event_type, "data": data}, default=str, ensure_ascii=False
    )
    return f"data: {payload}\n\n"


def _build_source_event(result: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Convert similarity-search result row into a frontend-friendly source payload."""
    chunk_metadata = result.get("chunk_metadata", {}) or {}
    doc_metadata = result.get("doc_metadata", {}) or {}
    source_id = result.get("source_id")

    title = (
        doc_metadata.get("title")
        or chunk_metadata.get("section")
        or chunk_metadata.get("heading")
        or source_id
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
            "source_id": source_id,
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
        raise RuntimeError(
            "VectorDatabase not initialized. Call init_services() first."
        )
    return _vector_db


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine(get_vector_db())
    return _rag_engine


def _build_user_public(user: Dict[str, Any]) -> UserPublic:
    return UserPublic(
        id=str(user["id"]),
        email=user["email"],
        created_at=user["created_at"],
    )


def _set_refresh_cookie(response: Response, refresh_token: str) -> None:
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=refresh_token,
        httponly=True,
        secure=AUTH_COOKIE_SECURE,
        samesite=AUTH_COOKIE_SAMESITE,
        max_age=REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        path="/",
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(key=REFRESH_COOKIE_NAME, path="/")


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header.")
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        raise HTTPException(status_code=401, detail="Invalid authorization header.")
    return parts[1]


def get_current_user(
    authorization: str | None = Header(default=None),
    vector_db: VectorDatabase = Depends(get_vector_db),
) -> Dict[str, Any]:
    token = _extract_bearer_token(authorization)
    try:
        payload = decode_access_token(token)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    user = vector_db.get_user_by_id(payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    return user


def _normalize_chat_history(
    messages: List[Dict[str, Any]] | None,
) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in messages or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        trimmed = content.strip()
        if not trimmed:
            continue
        normalized.append(
            {
                "role": role,
                "content": trimmed[:CHAT_HISTORY_MAX_CHARS_PER_TURN],
            }
        )
    return normalized[-CHAT_HISTORY_MAX_TURNS:]


def _resolve_history_for_query(
    query: RAGQuery, current_user: Dict[str, Any]
) -> List[Dict[str, str]]:
    history = _normalize_chat_history(query.chat_history)
    if history:
        return history

    if not query.session_id:
        return []

    session = get_vector_db().get_chat_session(
        str(current_user["id"]), query.session_id
    )
    if not session:
        return []
    return _normalize_chat_history(session.get("messages", []))


def _enforce_chat_session_limits(payload: ChatSessionPayload) -> None:
    if len(payload.id) > CHAT_SESSION_ID_MAX_CHARS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Session id length is {len(payload.id)}; "
                f"maximum allowed is {CHAT_SESSION_ID_MAX_CHARS}."
            ),
        )
    if len(payload.title) > CHAT_SESSION_TITLE_MAX_CHARS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Session title length is {len(payload.title)}; "
                f"maximum allowed is {CHAT_SESSION_TITLE_MAX_CHARS}."
            ),
        )

    messages = payload.messages or []
    if len(messages) > CHAT_SESSION_MAX_MESSAGES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Session has {len(messages)} messages; "
                f"maximum allowed is {CHAT_SESSION_MAX_MESSAGES}."
            ),
        )

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if (
            isinstance(content, str)
            and len(content) > CHAT_SESSION_MAX_MESSAGE_CONTENT_CHARS
        ):
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Message at index {idx} has content length {len(content)}; "
                    f"maximum allowed is {CHAT_SESSION_MAX_MESSAGE_CONTENT_CHARS}."
                ),
            )

    payload_size_bytes = len(
        json.dumps(
            payload.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if payload_size_bytes > CHAT_SESSION_MAX_PAYLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Session payload size is {payload_size_bytes} bytes; "
                f"maximum allowed is {CHAT_SESSION_MAX_PAYLOAD_BYTES} bytes."
            ),
        )


def _question_with_conversation_context(
    question: str, history: List[Dict[str, str]]
) -> str:
    if not history:
        return question

    question_text = question or ""
    preamble = (
        "Use the prior conversation context when relevant. "
        "If history conflicts with retrieved legal context, prioritize retrieved context."
    )
    history_header = "Conversation History:\n"
    question_header = "\n\nCurrent User Question:\n"
    fixed_chars = (
        len(preamble)
        + 2  # \n\n between preamble and history header
        + len(history_header)
        + len(question_header)
        + len(question_text)
    )
    history_budget = min(
        CHAT_HISTORY_MAX_TOTAL_CHARS,
        max(0, RAG_MAX_QUESTION_CHARS - fixed_chars),
    )
    if history_budget <= 0:
        return question_text

    lines_reversed: List[str] = []
    consumed_chars = 0
    for message in reversed(history):
        prefix = "User" if message["role"] == "user" else "Assistant"
        line = f"{prefix}: {message['content']}"
        next_size = len(line) + (1 if lines_reversed else 0)
        remaining = history_budget - consumed_chars
        if remaining <= 0:
            break
        if next_size > remaining:
            if remaining > 64:
                line = f"{line[: max(0, remaining - 4)].rstrip()}..."
                lines_reversed.append(line)
                consumed_chars = history_budget
            break
        lines_reversed.append(line)
        consumed_chars += next_size

    lines = list(reversed(lines_reversed))
    omission_marker = "[Earlier turns omitted due to prompt budget]"
    if len(lines) < len(history):
        trimmed_lines = lines[:]
        candidate_lines = [omission_marker, *trimmed_lines]
        while trimmed_lines and len("\n".join(candidate_lines)) > history_budget:
            trimmed_lines.pop(0)
            candidate_lines = [omission_marker, *trimmed_lines]
        if len("\n".join(candidate_lines)) <= history_budget:
            lines = candidate_lines

    history_block = "\n".join(lines).strip()
    if not history_block:
        return question_text

    return (
        f"{preamble}\n\n{history_header}{history_block}{question_header}{question_text}"
    )


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


@router.post("/auth/signup", response_model=AuthMessageResponse, status_code=201)
async def signup(payload: UserSignupRequest):
    """Create an email+password user account."""
    vector_db = get_vector_db()
    created_user = vector_db.create_user(
        email=normalize_email(payload.email),
        password_hash=hash_password(payload.password),
    )
    if not created_user:
        raise HTTPException(status_code=409, detail="Email is already registered.")
    return AuthMessageResponse(message="Account created successfully. Please log in.")


@router.post("/auth/login", response_model=AuthTokenResponse)
async def login(payload: UserLoginRequest, response: Response):
    """Authenticate a user and return access token + refresh cookie."""
    vector_db = get_vector_db()
    user = vector_db.get_user_by_email(normalize_email(payload.email))
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    access_token, expires_in = create_access_token(
        user_id=str(user["id"]), email=user["email"]
    )
    refresh_token, refresh_token_hash, refresh_expires_at = issue_refresh_token()
    vector_db.create_refresh_token(
        user_id=str(user["id"]),
        token_hash=refresh_token_hash,
        expires_at=refresh_expires_at,
    )
    _set_refresh_cookie(response, refresh_token)

    return AuthTokenResponse(
        access_token=access_token,
        expires_in=expires_in,
        user=_build_user_public(user),
    )


@router.post("/auth/refresh", response_model=AuthTokenResponse)
async def refresh_access_token(
    response: Response,
    refresh_token: str | None = Cookie(default=None, alias=REFRESH_COOKIE_NAME),
):
    """Rotate refresh token and return a fresh access token."""
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh token is missing.")

    vector_db = get_vector_db()
    refresh_token_hash = hash_refresh_token(refresh_token)
    refresh_record = vector_db.get_valid_refresh_token(refresh_token_hash)
    if not refresh_record:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=401, detail="Refresh token is invalid or expired."
        )

    user = refresh_record["user"]
    new_refresh_token, new_refresh_hash, new_refresh_expires_at = issue_refresh_token()
    try:
        vector_db.rotate_refresh_token(
            old_token_hash=refresh_token_hash,
            new_token_hash=new_refresh_hash,
            expires_at=new_refresh_expires_at,
        )
    except ValueError as exc:
        _clear_refresh_cookie(response)
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    access_token, expires_in = create_access_token(
        user_id=str(user["id"]), email=user["email"]
    )
    _set_refresh_cookie(response, new_refresh_token)

    return AuthTokenResponse(
        access_token=access_token,
        expires_in=expires_in,
        user=_build_user_public(user),
    )


@router.post("/auth/logout", response_model=AuthMessageResponse)
async def logout(
    response: Response,
    refresh_token: str | None = Cookie(default=None, alias=REFRESH_COOKIE_NAME),
):
    """Invalidate refresh token cookie and server-side token record."""
    if refresh_token:
        get_vector_db().revoke_refresh_token(hash_refresh_token(refresh_token))
    _clear_refresh_cookie(response)
    return AuthMessageResponse(message="Logged out successfully.")


@router.get("/auth/me", response_model=UserPublic)
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get the currently authenticated user."""
    return _build_user_public(current_user)


# ---------------------------------------------------------------------------
# Chat Sessions
# ---------------------------------------------------------------------------


@router.get("/chat/sessions", response_model=List[ChatSessionPayload])
async def list_chat_sessions(
    current_user: Dict[str, Any] = Depends(get_current_user),
    limit: int | None = Query(default=None, ge=1, le=CHAT_SESSION_LIST_MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
):
    """List chat sessions for the authenticated user."""
    return get_vector_db().list_chat_sessions(
        str(current_user["id"]), limit=limit, offset=offset
    )


@router.get("/chat/sessions/summary", response_model=List[ChatSessionSummaryPayload])
async def list_chat_session_summaries(
    current_user: Dict[str, Any] = Depends(get_current_user),
    limit: int = Query(
        default=CHAT_SESSION_SUMMARY_DEFAULT_LIMIT,
        ge=1,
        le=CHAT_SESSION_LIST_MAX_LIMIT,
    ),
    offset: int = Query(default=0, ge=0),
):
    """List chat session metadata for the authenticated user."""
    return get_vector_db().list_chat_session_summaries(
        str(current_user["id"]), limit=limit, offset=offset
    )


@router.get("/chat/sessions/{session_id}", response_model=ChatSessionPayload)
async def get_chat_session(
    session_id: str, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Fetch a single chat session for the authenticated user."""
    session = get_vector_db().get_chat_session(str(current_user["id"]), session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return session


@router.put("/chat/sessions/{session_id}", response_model=ChatSessionPayload)
async def save_chat_session(
    session_id: str,
    payload: ChatSessionPayload,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Create or update a chat session payload for the authenticated user."""
    if payload.id != session_id:
        raise HTTPException(
            status_code=400, detail="Path session id must match payload id."
        )
    _validate_chat_session_id(session_id)
    _enforce_chat_session_limits(payload)
    try:
        return get_vector_db().save_chat_session(
            str(current_user["id"]), payload.model_dump(mode="json")
        )
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.delete("/chat/sessions/{session_id}", response_model=AuthMessageResponse)
async def delete_chat_session(
    session_id: str, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a chat session for the authenticated user."""
    deleted = get_vector_db().delete_chat_session(str(current_user["id"]), session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return AuthMessageResponse(message="Chat session deleted.")


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
            content = row.get("content", str(row.to_dict()))
            metadata = row.get("metadata", {}) if "metadata" in row else {}

            # PII detection
            pii_detected = _pii_detector.detect(content)
            status = (
                DocumentStatus.PENDING_REVIEW.value
                if pii_detected
                else DocumentStatus.EXTRACTED.value
            )
            if pii_detected:
                pending_review_count += 1

            vector_db.store_document(
                doc_id,
                source_id,
                content,
                metadata,
                source.classification.value,
                status,
                pii_detected,
            )
            doc_ids.append(doc_id)

        return {
            "message": f"Extracted {len(doc_ids)} documents",
            "document_ids": doc_ids,
            "pending_review": pending_review_count,
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
    status = (
        DocumentStatus.APPROVED.value
        if decision.decision == "approve"
        else DocumentStatus.REJECTED.value
    )
    classification = (
        decision.classification_override.value
        if decision.classification_override
        else None
    )

    get_vector_db().update_document_status(
        document_id, status, decision.reviewer_id, classification
    )

    logger.info(
        f"✅ Document {document_id} {decision.decision}ed by {decision.reviewer_id}"
    )
    return {"message": f"Document {decision.decision}ed", "document_id": document_id}


# ---------------------------------------------------------------------------
# Vectorization
# ---------------------------------------------------------------------------


@router.post("/documents/{document_id}/vectorize")
async def vectorize_document(
    document_id: str,
    chunking_config: ChunkingConfig = ChunkingConfig(),
    vectorization_config: VectorizationConfig = VectorizationConfig(),
):
    """Vectorize approved document"""
    vector_db = get_vector_db()

    with vector_db.get_connection() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT content, metadata, classification, source_id
            FROM documents WHERE id = %s AND status = 'approved'
        """,
            (document_id,),
        )

        row = cur.fetchone()
        if not row:
            cur.close()
            raise HTTPException(status_code=404, detail="Approved document not found")

        content, metadata_json, classification, source_id = row

        metadata = (
            metadata_json
            if isinstance(metadata_json, dict)
            else (json.loads(metadata_json) if metadata_json else {})
        )

        # Vectorize
        vectorization = VectorizationEngine(vectorization_config.embedding_model.value)
        strategy = vectorization_config.chunking_strategy.value
        use_structure_aware = strategy == "structure_aware" or (
            strategy == "auto"
            and is_regulatory_document(source_id=source_id, metadata=metadata)
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
                content,
                metadata,
                {
                    "chunk_size": chunking_config.chunk_size,
                    "chunk_overlap": chunking_config.chunk_overlap,
                },
            )

        # Store chunks
        for chunk in vector_chunks:
            vector_db.store_vector_chunk(
                chunk["id"],
                document_id,
                chunk["content"],
                chunk["embedding"],
                chunk["metadata"],
                chunk["chunk_index"],
                classification,
            )

        # Update status
        cur.execute(
            "UPDATE documents SET status = 'vectorized' WHERE id = %s", (document_id,)
        )
        conn.commit()
        cur.close()

    logger.info(f"✅ Vectorized document {document_id}: {len(vector_chunks)} chunks")
    return {
        "message": "Document vectorized",
        "document_id": document_id,
        "num_chunks": len(vector_chunks),
    }


@router.post("/documents/publish")
async def publish_documents(document_ids: List[str]):
    """Publish vectorized documents for RAG"""
    vector_db = get_vector_db()

    with vector_db.get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE documents SET status = 'published'
            WHERE id = ANY(%s) AND status = 'vectorized'
            RETURNING id
        """,
            (document_ids,),
        )
        published = [str(row[0]) for row in cur.fetchall()]
        conn.commit()
        cur.close()

    logger.info(f"✅ Published {len(published)} documents")
    return {
        "message": f"Published {len(published)} documents",
        "published_ids": published,
    }


# ---------------------------------------------------------------------------
# RAG Query
# ---------------------------------------------------------------------------


@router.post("/rag/query", response_model=RAGResponse)
async def rag_query_endpoint(
    query: RAGQuery,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Query RAG system with security controls"""
    engine = get_rag_engine()
    classifications = query.classification_filter or [DataClassification.PUBLIC]
    engine._validate_llm_access(query.llm_provider, classifications)

    user_id = str(current_user["id"])
    history = _resolve_history_for_query(query, current_user)
    prompt_question = _question_with_conversation_context(query.question, history)
    route_mode, route_reason = _decide_query_mode_with_llm(engine, query, history)
    if route_mode in {"direct", "deny"}:
        llm_start = time.perf_counter()
        non_rag_prompt = _build_non_rag_prompt(query.question, route_mode, route_reason)
        answer = engine._generate_answer(query, non_rag_prompt)
        llm_ms = round((time.perf_counter() - llm_start) * 1000, 2)
        _log_non_rag_query(
            engine, user_id, query, route_mode, route_reason, llm_ms=llm_ms
        )
        return _empty_rag_response(answer, query.llm_provider, llm_ms=llm_ms)
    try:
        response = engine.query(query, user_id, prompt_question=prompt_question)
        logger.info(f"✅ RAG query by {user_id}: {query.question[:50]}...")
        return response
    except HTTPException as e:
        logger.error(f"❌ RAG query failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"❌ RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/query/stream")
async def rag_query_stream_endpoint(
    query: RAGQuery,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Streaming RAG response (text/plain) for LOCAL and OPENROUTER providers."""
    user_id = str(current_user["id"])
    history = _resolve_history_for_query(query, current_user)
    prompt_question = _question_with_conversation_context(query.question, history)
    if query.llm_provider not in {LLMProvider.LOCAL, LLMProvider.OPENROUTER}:
        raise HTTPException(
            status_code=400,
            detail="Streaming is only supported for LOCAL and OPENROUTER providers.",
        )

    engine = get_rag_engine()

    classifications = query.classification_filter or [DataClassification.PUBLIC]
    engine._validate_llm_access(query.llm_provider, classifications)
    route_mode, route_reason = _decide_query_mode_with_llm(engine, query, history)
    if route_mode in {"direct", "deny"}:
        non_rag_prompt = _build_non_rag_prompt(query.question, route_mode, route_reason)
        _log_non_rag_query(engine, user_id, query, route_mode, route_reason, llm_ms=0.0)
        return _streaming_non_rag_response(engine, query, non_rag_prompt)

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
        query.min_similarity,
        source_id=query.source_id,
    )
    search_ms = round((time.perf_counter() - search_start) * 1000, 2)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    max_class = engine._get_max_classification([r["classification"] for r in results])

    prompt, prompt_context_count = engine._build_prompt(prompt_question, results)

    engine.vector_db.log_query(
        user_id,
        query.question,
        query.llm_provider.value,
        max_class.value,
        True,
        details={
            "retrieved_count": len(results),
            "prompt_context_count": prompt_context_count,
            "source_id": query.source_id,
            "timings_ms": {"embed": embed_ms, "search": search_ms},
            "prompt_k": RAG_PROMPT_K,
            "retrieve_k": RAG_RETRIEVE_K,
            "history_turn_count": len(history),
        },
    )

    if query.llm_provider == LLMProvider.LOCAL:
        token_stream = engine._ollama_stream(prompt)
    else:
        token_stream = engine._openrouter_stream(prompt, temperature=query.temperature)

    return StreamingResponse(token_stream, media_type="text/plain")


@router.post("/rag/query/stream/events")
async def rag_query_stream_events_endpoint(
    query: RAGQuery,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
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
    user_id = str(current_user["id"])
    history = _resolve_history_for_query(query, current_user)
    prompt_question = _question_with_conversation_context(query.question, history)
    classifications = query.classification_filter or [DataClassification.PUBLIC]
    engine._validate_llm_access(query.llm_provider, classifications)
    route_mode, route_reason = _decide_query_mode_with_llm(engine, query, history)
    if route_mode in {"direct", "deny"}:
        non_rag_prompt = _build_non_rag_prompt(query.question, route_mode, route_reason)
        return _sse_non_rag_response(
            engine, query, non_rag_prompt, user_id, route_mode, route_reason
        )

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
                    "meta": {
                        "retrieve_k": retrieval_k,
                        "min_similarity": query.min_similarity,
                        "source_id": query.source_id,
                    },
                },
            )
            search_start = time.perf_counter()
            results = engine.vector_db.similarity_search(
                query_embedding,
                [c.value for c in classifications],
                retrieval_k,
                query.min_similarity,
                source_id=query.source_id,
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
            prompt, prompt_context_count = engine._build_prompt(
                prompt_question, results
            )
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
                stream_iter = engine._openrouter_stream(
                    prompt, temperature=query.temperature
                )
                error_code = "OPENROUTER_STREAM_ERROR"

            for token in stream_iter:
                if token and token.lstrip().startswith("[Error]"):
                    yield _sse_event(
                        "error", {"code": error_code, "message": token.strip()}
                    )
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

            max_class = engine._get_max_classification(
                [r["classification"] for r in results]
            )

            engine.vector_db.log_query(
                user_id,
                query.question,
                query.llm_provider.value,
                max_class.value,
                True,
                details={
                    "retrieved_count": len(results),
                    "prompt_context_count": prompt_context_count,
                    "source_id": query.source_id,
                    "timings_ms": timings_ms,
                    "prompt_k": RAG_PROMPT_K,
                    "retrieve_k": RAG_RETRIEVE_K,
                    "history_turn_count": len(history),
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
    vectorization_config: VectorizationConfig = VectorizationConfig(),
):
    """Complete workflow: Extract → Review → Vectorize → Publish"""
    workflow_id = str(uuid.uuid4())
    vector_db = get_vector_db()

    try:
        # Step 1: Extract
        logger.info(f"[{workflow_id}] Step 1: Extracting")
        extract_result = await extract_data(source_id)
        doc_ids = extract_result["document_ids"]

        # Step 2: Auto-approve public docs without PII
        with vector_db.get_connection() as conn:
            cur = conn.cursor()

            if auto_approve_public:
                cur.execute(
                    """
                    UPDATE documents SET status = 'approved'
                    WHERE id = ANY(%s) AND classification = 'public' AND pii_detected = FALSE
                """,
                    (doc_ids,),
                )
                conn.commit()

            # Step 3: Get approved
            cur.execute(
                "SELECT id FROM documents WHERE id = ANY(%s) AND status = 'approved'",
                (doc_ids,),
            )
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
            "pending_review": len(doc_ids) - len(approved_ids),
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
        cur.execute(
            """
            SELECT timestamp, user_id, query, llm_provider, classification, success
            FROM audit_log ORDER BY timestamp DESC LIMIT %s
        """,
            (limit,),
        )

        logs = [
            {
                "timestamp": row[0].isoformat(),
                "user_id": row[1],
                "query": row[2],
                "llm_provider": row[3],
                "classification": row[4],
                "success": row[5],
            }
            for row in cur.fetchall()
        ]

        cur.close()

    return {"logs": logs, "total": len(logs)}
