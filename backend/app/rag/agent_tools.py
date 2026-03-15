"""
LangChain tool definitions for the RAG agent.

Three tools returned by make_tools():
  - knowledge_base_search  : hybrid vector + lexical search over ingested docs
  - create_zammad_ticket   : creates a training request ticket in Zammad
  - generate_report        : compiles a structured markdown report from retrieved docs
"""

import logging
import os
from typing import List, Optional

from langchain.tools import tool

from app.core.database import VectorDatabase
from app.core.enums import DataClassification
from app.vectorization.engine import get_cached_embedding_model, DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_TOOL_RETRIEVE_K = int(os.getenv("AGENT_RETRIEVE_K", "8"))
_TOOL_MIN_SIMILARITY = float(os.getenv("AGENT_MIN_SIMILARITY", "0.3"))


def make_tools(
    vector_db: VectorDatabase,
    user_id: str,
    source_id: Optional[str] = None,
) -> List:
    """Return the three agent tools bound to the given vector_db instance."""

    embedding_model = get_cached_embedding_model(DEFAULT_EMBEDDING_MODEL)

    def _embed(text: str):
        return embedding_model.encode(text).tolist()

    def _search(query: str):
        embedding = _embed(query.strip()[:500])
        return vector_db.similarity_search(
            embedding,
            [DataClassification.PUBLIC.value],
            k=_TOOL_RETRIEVE_K,
            min_similarity=_TOOL_MIN_SIMILARITY,
            source_id=source_id,
            query_text=query.strip()[:500],
        )

    @tool
    def knowledge_base_search(query: str) -> str:
        """Search the regulatory knowledge base for documents relevant to a query.
        Always call this first before attempting to answer any regulatory question.
        Returns formatted excerpts with section references, or a NO_RESULTS sentinel.
        Input: a plain-text search query (max 500 chars)."""

        results = _search(query)
        if not results:
            return f"NO_RESULTS: No relevant documents found for: {query[:200]}"

        lines = [f"Found {len(results)} document(s):\n"]
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            heading = meta.get("heading", meta.get("section", f"doc-{i}"))
            content = r.get("content", "")[:600]
            lines.append(f"[{i}] {heading}\n{content}\n")
        return "\n".join(lines)

    @tool
    def create_zammad_ticket(question: str, reason: str) -> str:
        """Create a training ticket in Zammad when the knowledge base cannot answer.
        Call this ONLY after knowledge_base_search returns NO_RESULTS.
        Inputs: question (the unanswered question), reason (why training data is needed).
        Returns: the ticket URL or a skip notice if Zammad is not configured."""

        from app.config import ZAMMAD_URL, TICKET_SUBMIT_URL
        from app.rag.engine import RAGEngine

        if not ZAMMAD_URL:
            return f"TICKET_SKIPPED: Zammad not configured. Question logged: {question[:200]}"

        url = RAGEngine._create_zammad_ticket(question, reason=reason)
        if url and url != TICKET_SUBMIT_URL:
            return f"TICKET_CREATED: {url}"
        return f"TICKET_FAILED: Could not create ticket, fallback URL: {url}"

    @tool
    def generate_report(topic: str) -> str:
        """Generate a structured markdown report on a regulatory topic.
        Searches the knowledge base and compiles all relevant information.
        Use when the user explicitly asks for a report, summary, or overview.
        Input: the topic or regulation area to report on (max 300 chars)."""

        results = _search(topic[:300])
        if not results:
            return f"NO_RESULTS: No documents found for topic: {topic[:200]}"

        lines = [
            f"# Report: {topic}\n",
            f"*Based on {len(results)} document(s) from the knowledge base.*\n",
        ]
        for r in results:
            meta = r.get("metadata", {})
            heading = meta.get("heading", meta.get("section", "Unknown"))
            content = r.get("content", "")[:800]
            part = meta.get("part", "")
            title_num = meta.get("title", "")
            ref = f"{title_num} CFR Part {part}" if part and title_num else "Unknown"
            lines.append(f"\n## {heading}\n**Reference:** {ref}\n\n{content}")

        return "\n".join(lines)

    return [knowledge_base_search, create_zammad_ticket, generate_report]
