"""
FastAPI route handlers for the RAG-Chatbot platform.

All HTTP endpoints are defined here as a thin shell that delegates
business logic to the appropriate engine/service classes.
"""

import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

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
RAG_ENABLE_LLM_ROUTER = os.getenv("RAG_ENABLE_LLM_ROUTER", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RETRIEVAL_HISTORY_USER_TURNS = int(os.getenv("RAG_RETRIEVAL_HISTORY_USER_TURNS", "2"))
RETRIEVAL_CONTEXT_MAX_CHARS = int(os.getenv("RAG_RETRIEVAL_CONTEXT_MAX_CHARS", "700"))
RETRIEVAL_REFERENCE_HISTORY_TURNS = int(
    os.getenv("RAG_RETRIEVAL_REFERENCE_HISTORY_TURNS", "12")
)
FOLLOWUP_REFERENCE_PATTERN = re.compile(
    r"\b(this|that|these|those|it|they|them|same|such|above|below|here|there)\b",
    re.IGNORECASE,
)
EXPLICIT_LEGAL_REFERENCE_PATTERN = re.compile(
    r"(§\s*\d+[\w.-]*|\bpart\s+\d+[\w-]*\b|\bsubpart\s+[a-z0-9]+\b|\btitle\s+\d+\b|\b\d+\s*cfr\b)",
    re.IGNORECASE,
)
SECTION_NUMERIC_PATTERN = re.compile(r"\b\d{3,4}\.\d{1,3}[a-z]?(?:\([a-z0-9]+\))?\b", re.IGNORECASE)
LEGAL_KEYWORD_HINTS = (
    "fhfa",
    "cfr",
    "allocation",
    "targeted fund",
    "subsidy",
    "regulation",
    "rule",
    "part ",
    "section",
    "chapter",
    "bank",
)
RAG_DEBUG_RETRIEVAL_LOGS = os.getenv("RAG_DEBUG_RETRIEVAL_LOGS", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


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


def _looks_legal_or_doc_query(question: str, history: List[Dict[str, str]]) -> bool:
    text = (question or "").strip().lower()
    if not text:
        return False
    if EXPLICIT_LEGAL_REFERENCE_PATTERN.search(text):
        return True
    if SECTION_NUMERIC_PATTERN.search(text):
        return True
    if any(keyword in text for keyword in LEGAL_KEYWORD_HINTS):
        return True
    if history and _is_followup_reference_question(text):
        return True
    if text.endswith("?") and len(text.split()) >= 6:
        return True
    return False


def _decide_query_mode(
    engine: RAGEngine, query: RAGQuery, history: List[Dict[str, str]]
) -> Tuple[str, str, float]:
    """Decide routing mode and return (mode, reason, routing_ms)."""
    if not RAG_ENABLE_LLM_ROUTER:
        return "rag", "router_disabled", 0.0

    if _looks_legal_or_doc_query(query.question, history):
        return "rag", "heuristic_legal_query", 0.0

    route_start = time.perf_counter()
    mode, reason = _decide_query_mode_with_llm(engine, query, history)
    routing_ms = round((time.perf_counter() - route_start) * 1000, 2)
    return mode, reason, routing_ms


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


def _build_missing_reference_anchor_answer() -> str:
    return (
        "I need the controlling legal reference to answer this follow-up accurately. "
        "Please include the exact section/part (for example, `12 CFR §1291.12(c)`), "
        "or ask this in the same chat after the prior question so I can resolve "
        "what “same section” refers to."
    )


def _requires_reference_anchor_clarification(
    question: str, history: List[Dict[str, str]], metadata_filters: Dict[str, Any]
) -> bool:
    question_text = (question or "").strip()
    if not question_text:
        return False
    if not _is_followup_reference_question(question_text):
        return False
    if EXPLICIT_LEGAL_REFERENCE_PATTERN.search(question_text):
        return False
    if metadata_filters:
        return False
    history_refs = _recent_reference_turns(
        history,
        max_turns=max(RETRIEVAL_REFERENCE_HISTORY_TURNS, LEGAL_ROUTER_HISTORY_TURNS),
    )
    return len(history_refs) == 0


def _stream_static_text_response(answer: str) -> StreamingResponse:
    def event_generator():
        yield answer

    return StreamingResponse(event_generator(), media_type="text/plain")


def _sse_static_answer_response(answer: str, label: str = "Clarification needed") -> StreamingResponse:
    def event_generator():
        timings = {
            "routing": 0.0,
            "embed": 0.0,
            "search": 0.0,
            "prompt_build": 0.0,
            "llm": 0.0,
            "total": 0.0,
        }
        yield _sse_event(
            "status",
            {
                "stage": "generation",
                "state": "done",
                "label": label,
            },
        )
        yield _sse_event(
            "final",
            {
                "answer": answer,
                "timings_ms": timings,
                "retrieved_count": 0,
                "prompt_context_count": 0,
            },
        )
        yield _sse_event("done", {})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _empty_rag_response(
    answer: str,
    llm_provider: LLMProvider,
    llm_ms: float = 0.0,
    routing_ms: float = 0.0,
) -> RAGResponse:
    total_ms = round(llm_ms + routing_ms, 2)
    return RAGResponse(
        answer=answer,
        sources=[],
        classification=DataClassification.PUBLIC,
        llm_provider=llm_provider,
        retrieved_count=0,
        prompt_context_count=0,
        total_ms=total_ms,
        timings_ms={
            "routing": round(routing_ms, 2),
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
    routing_ms: float = 0.0,
) -> None:
    try:
        total_ms = round(llm_ms + routing_ms, 2)
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
                    "routing": round(routing_ms, 2),
                    "embed": 0.0,
                    "search": 0.0,
                    "prompt_build": 0.0,
                    "llm": round(llm_ms, 2),
                    "total": total_ms,
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
    if query.llm_provider == LLMProvider.OPENAI:
        return engine._openai_stream(prompt, temperature=query.temperature)
    if query.llm_provider == LLMProvider.OPENROUTER:
        return engine._openrouter_stream(prompt, temperature=query.temperature)
    raise HTTPException(
        status_code=400,
        detail=f"Streaming not supported for provider {query.llm_provider.value}.",
    )


def _streaming_non_rag_response(engine: RAGEngine, query: RAGQuery, prompt: str) -> StreamingResponse:
    return StreamingResponse(_stream_from_provider(engine, query, prompt), media_type="text/plain")


def _sse_non_rag_response(
    engine: RAGEngine,
    query: RAGQuery,
    prompt: str,
    user_id: str,
    mode: str,
    reason: str,
    routing_ms: float = 0.0,
) -> StreamingResponse:
    def event_generator():
        total_start = time.perf_counter()
        answer_parts: List[str] = []
        try:
            yield _sse_event(
                "status",
                {
                    "stage": "routing",
                    "state": "done",
                    "label": "Routing decision complete",
                    "meta": {
                        "duration_ms": round(routing_ms, 2),
                        "route_mode": mode,
                        "route_reason": reason,
                    },
                },
            )
            yield _sse_event(
                "status",
                {
                    "stage": "generation",
                    "state": "start",
                    "label": "Generating response",
                },
            )
            stream_iter = _stream_from_provider(engine, query, prompt)
            error_code_map = {
                LLMProvider.LOCAL: "LOCAL_LLM_ERROR",
                LLMProvider.OPENAI: "OPENAI_STREAM_ERROR",
                LLMProvider.OPENROUTER: "OPENROUTER_STREAM_ERROR",
            }
            error_code = error_code_map.get(query.llm_provider, "STREAM_ERROR")
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
            _log_non_rag_query(
                engine,
                user_id,
                query,
                mode,
                reason,
                llm_ms=llm_ms,
                routing_ms=routing_ms,
            )
            total_ms = round(llm_ms + routing_ms, 2)
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
                        "routing": round(routing_ms, 2),
                        "embed": 0.0,
                        "search": 0.0,
                        "prompt_build": 0.0,
                        "llm": llm_ms,
                        "total": total_ms,
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


def _recent_user_turns(
    history: List[Dict[str, str]], max_turns: int = RETRIEVAL_HISTORY_USER_TURNS
) -> List[str]:
    if max_turns <= 0:
        return []
    user_turns = [
        (turn.get("content") or "").strip()
        for turn in history
        if turn.get("role") == "user" and (turn.get("content") or "").strip()
    ]
    if not user_turns:
        return []
    # Keep most recent distinct turns to avoid repeated follow-up loops.
    distinct_reversed: List[str] = []
    seen_normalized = set()
    for text in reversed(user_turns):
        normalized = text.lower()
        if normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)
        distinct_reversed.append(text)
        if len(distinct_reversed) >= max_turns:
            break
    return list(reversed(distinct_reversed))


def _recent_reference_turns(
    history: List[Dict[str, str]], max_turns: int = LEGAL_ROUTER_HISTORY_TURNS
) -> List[str]:
    if max_turns <= 0:
        return []
    reference_turns: List[str] = []
    for turn in history[-max_turns:]:
        if turn.get("role") != "user":
            continue
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if EXPLICIT_LEGAL_REFERENCE_PATTERN.search(content):
            reference_turns.append(content[:RETRIEVAL_CONTEXT_MAX_CHARS])
    return reference_turns[-max_turns:]


def _normalize_conversation_context(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    normalized: List[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if text:
            normalized.append(text[:RETRIEVAL_CONTEXT_MAX_CHARS])
    return normalized[-RETRIEVAL_HISTORY_USER_TURNS:]


def _is_followup_reference_question(question: str) -> bool:
    text = (question or "").strip()
    if not text:
        return False
    if EXPLICIT_LEGAL_REFERENCE_PATTERN.search(text):
        return False
    lower = text.lower()
    if lower.startswith(("what about", "and ", "does that", "can that")):
        return True
    if len(lower.split()) <= 10 and FOLLOWUP_REFERENCE_PATTERN.search(lower):
        return True
    return bool(FOLLOWUP_REFERENCE_PATTERN.search(lower))


def _question_for_retrieval(
    question: str, history: List[Dict[str, str]], conversation_context: Any
) -> str:
    question_text = (question or "").strip()
    if not question_text:
        return ""

    context_turns = _normalize_conversation_context(conversation_context)
    if not context_turns:
        context_turns = _recent_user_turns(history)
    while context_turns and context_turns[-1].strip().lower() == question_text.lower():
        context_turns = context_turns[:-1]

    if not context_turns and _is_followup_reference_question(question_text):
        context_turns = _recent_user_turns(
            history, max(max(RETRIEVAL_HISTORY_USER_TURNS, 2) + 2, 4)
        )
        while (
            context_turns
            and context_turns[-1].strip().lower() == question_text.lower()
        ):
            context_turns = context_turns[:-1]

    if not context_turns or not _is_followup_reference_question(question_text):
        return question_text[:RAG_MAX_QUESTION_CHARS]

    anchor = context_turns[-1][:RETRIEVAL_CONTEXT_MAX_CHARS]
    reference_hints = _recent_reference_turns(history)
    hints_text = ""
    if reference_hints:
        hints_text = (
            "\n\nRelevant cited references from recent conversation:\n"
            + "\n".join(reference_hints[-2:])
        )
    retrieval_question = (
        f"Prior user context:\n{anchor}{hints_text}\n\nFollow-up question:\n{question_text}"
    )
    return retrieval_question[:RAG_MAX_QUESTION_CHARS]


def _merge_metadata_filters(
    explicit_filters: Optional[Dict[str, Any]],
    retrieval_question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    def _unique_preserve_order(values: List[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _apply_if_missing(key: str, values: List[str]) -> None:
        if key in filters:
            return
        unique_values = _unique_preserve_order(values)
        if not unique_values:
            return
        filters[key] = unique_values[0] if len(unique_values) == 1 else unique_values

    filters: Dict[str, Any] = {}
    if isinstance(explicit_filters, dict):
        for key, value in explicit_filters.items():
            key_text = str(key or "").strip().lower()
            if not key_text:
                continue
            filters[key_text] = value

    def _extract_reference_values(text: str) -> Dict[str, List[str]]:
        title_values = re.findall(r"\btitle\s+(\d+)\b", text, flags=re.IGNORECASE)
        chapter_values = [
            match.upper()
            for match in re.findall(
                r"\bchapter\s+([ivxlcdm]+|\d+)\b", text, flags=re.IGNORECASE
            )
        ]
        part_values = [
            match.lower()
            for match in re.findall(
                r"\bpart\s+(\d+[a-z]?)\b", text, flags=re.IGNORECASE
            )
        ]
        section_values = [
            match.lower()
            for match in re.findall(r"§\s*([\d.]+[a-z]?)", text, flags=re.IGNORECASE)
        ]
        return {
            "title": title_values,
            "chapter": chapter_values,
            "part": part_values,
            "section": section_values,
        }

    current_values = _extract_reference_values(retrieval_question or "")
    current_has_explicit_refs = any(
        current_values[key] for key in ("title", "chapter", "part", "section")
    )
    title_values = current_values["title"]
    chapter_values = current_values["chapter"]
    part_values = current_values["part"]
    section_values = current_values["section"]

    _apply_if_missing("title", title_values)
    _apply_if_missing("chapter", chapter_values)
    _apply_if_missing("part", part_values)
    _apply_if_missing("section", section_values)

    if "part" not in filters and section_values:
        section_parts = [section.split(".", 1)[0] for section in section_values if "." in section]
        _apply_if_missing("part", section_parts)

    if history and not current_has_explicit_refs:
        history_window = history[-max(RETRIEVAL_REFERENCE_HISTORY_TURNS, LEGAL_ROUTER_HISTORY_TURNS) :]
        history_text = "\n".join(
            (turn.get("content") or "").strip()[:RETRIEVAL_CONTEXT_MAX_CHARS]
            for turn in history_window
            if turn.get("role") == "user" and (turn.get("content") or "").strip()
        )
        if history_text:
            history_values = _extract_reference_values(history_text)
            _apply_if_missing("title", history_values["title"])
            _apply_if_missing("chapter", history_values["chapter"])
            _apply_if_missing("part", history_values["part"])
            _apply_if_missing("section", history_values["section"])
            if "part" not in filters and history_values["section"]:
                section_parts = [
                    section.split(".", 1)[0]
                    for section in history_values["section"]
                    if "." in section
                ]
                _apply_if_missing("part", section_parts)

    return filters


def _source_metadata_value(source: Dict[str, Any], key: str) -> Optional[str]:
    chunk_metadata = source.get("chunk_metadata") or {}
    doc_metadata = source.get("doc_metadata") or {}
    value = chunk_metadata.get(key)
    if value in (None, ""):
        value = doc_metadata.get(key)
    if value in (None, ""):
        return None
    return str(value).strip()


def _summarize_retrieval_sources(
    sources: List[Dict[str, Any]], limit: int = 8
) -> Dict[str, Any]:
    part_counts: Dict[str, int] = {}
    section_counts: Dict[str, int] = {}
    top_refs: List[Dict[str, Any]] = []

    for source in sources:
        part = _source_metadata_value(source, "part") or "(none)"
        section = _source_metadata_value(source, "section") or "(none)"
        part_counts[part] = part_counts.get(part, 0) + 1
        section_counts[section] = section_counts.get(section, 0) + 1

    for idx, source in enumerate(sources[: max(limit, 0)], start=1):
        top_refs.append(
            {
                "rank": idx,
                "chunk_id": source.get("chunk_id"),
                "part": _source_metadata_value(source, "part"),
                "section": _source_metadata_value(source, "section"),
                "heading": _source_metadata_value(source, "heading")
                or _source_metadata_value(source, "section_header"),
                "similarity": source.get("similarity"),
                "lexical_score": source.get("lexical_score"),
                "hybrid_score": source.get("hybrid_score"),
            }
        )

    return {
        "retrieved_count": len(sources),
        "part_distribution": part_counts,
        "section_distribution": section_counts,
        "top_refs": top_refs,
    }


def _log_retrieval_debug(
    event: str,
    query: RAGQuery,
    retrieval_question: str,
    metadata_filters: Dict[str, Any],
    route_mode: str,
    route_reason: str,
    history_turns: int,
    sources: Optional[List[Dict[str, Any]]] = None,
    timings_ms: Optional[Dict[str, Any]] = None,
) -> None:
    if not RAG_DEBUG_RETRIEVAL_LOGS:
        return
    payload: Dict[str, Any] = {
        "event": event,
        "question": (query.question or "")[:220],
        "retrieval_question": (retrieval_question or "")[:320],
        "retrieval_mode": query.retrieval_mode,
        "metadata_filters": metadata_filters or {},
        "route_mode": route_mode,
        "route_reason": route_reason,
        "history_turns": history_turns,
    }
    if sources is not None:
        payload["retrieval_summary"] = _summarize_retrieval_sources(sources)
    if timings_ms:
        payload["timings_ms"] = timings_ms
    logger.info("[RAG_DEBUG] %s", json.dumps(payload, ensure_ascii=True, default=str))


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
    retrieval_question = _question_for_retrieval(
        query.question, history, query.conversation_context
    )
    metadata_filters = _merge_metadata_filters(
        query.metadata_filters, retrieval_question, history
    )
    query_for_rag = query.model_copy(update={"metadata_filters": metadata_filters})
    if _requires_reference_anchor_clarification(
        query.question, history, query_for_rag.metadata_filters or {}
    ):
        answer = _build_missing_reference_anchor_answer()
        _log_retrieval_debug(
            event="query_sync_clarification",
            query=query,
            retrieval_question=retrieval_question,
            metadata_filters=query_for_rag.metadata_filters or {},
            route_mode="direct",
            route_reason="missing_reference_anchor",
            history_turns=len(history),
        )
        _log_non_rag_query(
            engine,
            user_id,
            query,
            "direct",
            "missing_reference_anchor",
            llm_ms=0.0,
            routing_ms=0.0,
        )
        return _empty_rag_response(
            answer,
            query.llm_provider,
            llm_ms=0.0,
            routing_ms=0.0,
        )
    route_mode, route_reason, route_ms = _decide_query_mode(engine, query, history)
    if route_mode in {"direct", "deny"}:
        llm_start = time.perf_counter()
        non_rag_prompt = _build_non_rag_prompt(query.question, route_mode, route_reason)
        answer = engine._generate_answer(query, non_rag_prompt)
        llm_ms = round((time.perf_counter() - llm_start) * 1000, 2)
        _log_non_rag_query(
            engine,
            user_id,
            query,
            route_mode,
            route_reason,
            llm_ms=llm_ms,
            routing_ms=route_ms,
        )
        return _empty_rag_response(
            answer,
            query.llm_provider,
            llm_ms=llm_ms,
            routing_ms=route_ms,
        )
    try:
        _log_retrieval_debug(
            event="query_sync_start",
            query=query,
            retrieval_question=retrieval_question,
            metadata_filters=query_for_rag.metadata_filters or {},
            route_mode=route_mode,
            route_reason=route_reason,
            history_turns=len(history),
        )
        response = engine.query(
            query_for_rag,
            user_id,
            prompt_question=prompt_question,
            retrieval_question=retrieval_question,
        )
        response.timings_ms["routing"] = round(route_ms, 2)
        response.total_ms = round(response.total_ms + route_ms, 2)
        response.timings_ms["total"] = response.total_ms
        _log_retrieval_debug(
            event="query_sync_done",
            query=query,
            retrieval_question=retrieval_question,
            metadata_filters=query_for_rag.metadata_filters or {},
            route_mode=route_mode,
            route_reason=route_reason,
            history_turns=len(history),
            sources=response.sources,
            timings_ms=response.timings_ms,
        )
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
    """Streaming RAG response (text/plain) for LOCAL, OPENAI, and OPENROUTER providers."""
    user_id = str(current_user["id"])
    history = _resolve_history_for_query(query, current_user)
    prompt_question = _question_with_conversation_context(query.question, history)
    retrieval_question = _question_for_retrieval(
        query.question, history, query.conversation_context
    )
    metadata_filters = _merge_metadata_filters(
        query.metadata_filters, retrieval_question, history
    )
    query_for_rag = query.model_copy(update={"metadata_filters": metadata_filters})
    if query.llm_provider not in {
        LLMProvider.LOCAL,
        LLMProvider.OPENAI,
        LLMProvider.OPENROUTER,
    }:
        raise HTTPException(
            status_code=400,
            detail="Streaming is only supported for LOCAL, OPENAI, and OPENROUTER providers.",
        )

    engine = get_rag_engine()

    classifications = query.classification_filter or [DataClassification.PUBLIC]
    engine._validate_llm_access(query.llm_provider, classifications)
    if _requires_reference_anchor_clarification(
        query.question, history, query_for_rag.metadata_filters or {}
    ):
        answer = _build_missing_reference_anchor_answer()
        _log_retrieval_debug(
            event="query_stream_clarification",
            query=query,
            retrieval_question=retrieval_question,
            metadata_filters=query_for_rag.metadata_filters or {},
            route_mode="direct",
            route_reason="missing_reference_anchor",
            history_turns=len(history),
        )
        _log_non_rag_query(
            engine,
            user_id,
            query,
            "direct",
            "missing_reference_anchor",
            llm_ms=0.0,
            routing_ms=0.0,
        )
        return _stream_static_text_response(answer)
    route_mode, route_reason, route_ms = _decide_query_mode(engine, query, history)
    if route_mode in {"direct", "deny"}:
        non_rag_prompt = _build_non_rag_prompt(query.question, route_mode, route_reason)
        _log_non_rag_query(
            engine,
            user_id,
            query,
            route_mode,
            route_reason,
            llm_ms=0.0,
            routing_ms=route_ms,
        )
        return _streaming_non_rag_response(engine, query, non_rag_prompt)

    _log_retrieval_debug(
        event="query_stream_start",
        query=query,
        retrieval_question=retrieval_question,
        metadata_filters=query_for_rag.metadata_filters or {},
        route_mode=route_mode,
        route_reason=route_reason,
        history_turns=len(history),
    )

    if engine.vectorization is None:
        engine.vectorization = VectorizationEngine(DEFAULT_EMBEDDING_MODEL)
    embed_start = time.perf_counter()
    query_embedding = engine.get_query_embedding(retrieval_question)
    embed_ms = round((time.perf_counter() - embed_start) * 1000, 2)

    search_start = time.perf_counter()
    results = engine.vector_db.similarity_search(
        query_embedding,
        [c.value for c in classifications],
        min(max(query.top_k, 1), RAG_RETRIEVE_K),
        query.min_similarity,
        source_id=query.source_id,
        query_text=retrieval_question,
        metadata_filters=query_for_rag.metadata_filters,
        retrieval_mode=query.retrieval_mode,
    )
    search_ms = round((time.perf_counter() - search_start) * 1000, 2)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    max_class = engine._get_max_classification([r["classification"] for r in results])

    prompt, prompt_context_count = engine._build_prompt(prompt_question, results)
    timings_for_debug = {
        "routing": round(route_ms, 2),
        "embed": embed_ms,
        "search": search_ms,
    }
    _log_retrieval_debug(
        event="query_stream_retrieval_done",
        query=query,
        retrieval_question=retrieval_question,
        metadata_filters=query_for_rag.metadata_filters or {},
        route_mode=route_mode,
        route_reason=route_reason,
        history_turns=len(history),
        sources=results,
        timings_ms=timings_for_debug,
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
            "timings_ms": {"embed": embed_ms, "search": search_ms},
            "prompt_k": RAG_PROMPT_K,
            "retrieve_k": RAG_RETRIEVE_K,
            "history_turn_count": len(history),
            "retrieval_mode": query.retrieval_mode,
            "metadata_filters": query_for_rag.metadata_filters or {},
            "routing_ms": round(route_ms, 2),
            "retrieval_question": retrieval_question,
            "retrieval_debug": engine._build_retrieval_debug(results, limit=8),
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
    if query.llm_provider not in {
        LLMProvider.LOCAL,
        LLMProvider.OPENAI,
        LLMProvider.OPENROUTER,
    }:
        raise HTTPException(
            status_code=400,
            detail="Structured streaming is only supported for LOCAL, OPENAI, and OPENROUTER providers.",
        )

    engine = get_rag_engine()
    user_id = str(current_user["id"])
    history = _resolve_history_for_query(query, current_user)
    prompt_question = _question_with_conversation_context(query.question, history)
    retrieval_question = _question_for_retrieval(
        query.question, history, query.conversation_context
    )
    metadata_filters = _merge_metadata_filters(
        query.metadata_filters, retrieval_question, history
    )
    query_for_rag = query.model_copy(update={"metadata_filters": metadata_filters})
    classifications = query.classification_filter or [DataClassification.PUBLIC]
    engine._validate_llm_access(query.llm_provider, classifications)
    if _requires_reference_anchor_clarification(
        query.question, history, query_for_rag.metadata_filters or {}
    ):
        answer = _build_missing_reference_anchor_answer()
        _log_retrieval_debug(
            event="query_sse_clarification",
            query=query,
            retrieval_question=retrieval_question,
            metadata_filters=query_for_rag.metadata_filters or {},
            route_mode="direct",
            route_reason="missing_reference_anchor",
            history_turns=len(history),
        )
        _log_non_rag_query(
            engine,
            user_id,
            query,
            "direct",
            "missing_reference_anchor",
            llm_ms=0.0,
            routing_ms=0.0,
        )
        return _sse_static_answer_response(answer)
    route_mode, route_reason, route_ms = _decide_query_mode(engine, query, history)
    if route_mode in {"direct", "deny"}:
        non_rag_prompt = _build_non_rag_prompt(query.question, route_mode, route_reason)
        return _sse_non_rag_response(
            engine,
            query,
            non_rag_prompt,
            user_id,
            route_mode,
            route_reason,
            routing_ms=route_ms,
        )

    _log_retrieval_debug(
        event="query_sse_start",
        query=query,
        retrieval_question=retrieval_question,
        metadata_filters=query_for_rag.metadata_filters or {},
        route_mode=route_mode,
        route_reason=route_reason,
        history_turns=len(history),
    )

    def event_generator():
        total_start = time.perf_counter()
        timings_ms: Dict[str, float] = {}
        try:
            if engine.vectorization is None:
                engine.vectorization = VectorizationEngine(DEFAULT_EMBEDDING_MODEL)

            timings_ms["routing"] = round(route_ms, 2)
            yield _sse_event(
                "status",
                {
                    "stage": "routing",
                    "state": "done",
                    "label": "Routing decision complete",
                    "meta": {
                        "duration_ms": round(route_ms, 2),
                        "route_mode": route_mode,
                        "route_reason": route_reason,
                    },
                },
            )

            yield _sse_event(
                "status",
                {
                    "stage": "embedding",
                    "state": "start",
                    "label": "Generating query embedding",
                },
            )
            embed_start = time.perf_counter()
            query_embedding = engine.get_query_embedding(retrieval_question)
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
                        "retrieval_mode": query.retrieval_mode,
                        "applied_filters": query_for_rag.metadata_filters or {},
                        "retrieval_question_preview": retrieval_question[:220],
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
                query_text=retrieval_question,
                metadata_filters=query_for_rag.metadata_filters,
                retrieval_mode=query.retrieval_mode,
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

            retrieval_debug = engine._build_retrieval_debug(results, limit=8)
            _log_retrieval_debug(
                event="query_sse_retrieval_done",
                query=query,
                retrieval_question=retrieval_question,
                metadata_filters=query_for_rag.metadata_filters or {},
                route_mode=route_mode,
                route_reason=route_reason,
                history_turns=len(history),
                sources=results,
                timings_ms={
                    "routing": round(route_ms, 2),
                    "embed": embed_ms,
                    "search": search_ms,
                },
            )
            for idx, result in enumerate(results, start=1):
                yield _sse_event("source", _build_source_event(result, idx))

            yield _sse_event(
                "status",
                {
                    "stage": "retrieval",
                    "state": "done",
                    "label": "Relevant chunks retrieved",
                    "meta": {
                        "duration_ms": search_ms,
                        "retrieved_count": len(results),
                        "part_distribution": retrieval_debug.get("part_distribution", {}),
                    },
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
            first_token_ms: Optional[float] = None
            output_chunks = 0
            if query.llm_provider == LLMProvider.LOCAL:
                stream_iter = engine._ollama_stream(prompt)
                error_code = "LOCAL_LLM_ERROR"
            elif query.llm_provider == LLMProvider.OPENAI:
                stream_iter = engine._openai_stream(prompt, temperature=query.temperature)
                error_code = "OPENAI_STREAM_ERROR"
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
                    if first_token_ms is None:
                        first_token_ms = round(
                            (time.perf_counter() - llm_start) * 1000, 2
                        )
                    answer_parts.append(token)
                    output_chunks += 1
                    yield _sse_event("token", {"text": token})

            llm_ms = round((time.perf_counter() - llm_start) * 1000, 2)
            timings_ms["llm"] = llm_ms
            if first_token_ms is not None:
                timings_ms["llm_first_token"] = first_token_ms
            timings_ms["llm_output_chunks"] = float(output_chunks)

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
                    "retrieval_mode": query.retrieval_mode,
                    "metadata_filters": query_for_rag.metadata_filters or {},
                    "retrieval_question": retrieval_question,
                    "retrieval_debug": retrieval_debug,
                    "route_mode": route_mode,
                    "route_reason": route_reason,
                },
            )
            _log_retrieval_debug(
                event="query_sse_done",
                query=query,
                retrieval_question=retrieval_question,
                metadata_filters=query_for_rag.metadata_filters or {},
                route_mode=route_mode,
                route_reason=route_reason,
                history_turns=len(history),
                sources=results,
                timings_ms=timings_ms,
            )

            yield _sse_event(
                "status",
                {
                    "stage": "generation",
                    "state": "done",
                    "label": "Generation complete",
                    "meta": {
                        "duration_ms": llm_ms,
                        "first_token_ms": first_token_ms,
                        "output_chunks": output_chunks,
                    },
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
