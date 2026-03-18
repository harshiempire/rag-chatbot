"""
LangChain tool definitions for the RAG agent.

All retrieval work delegates to the RAGEngine instance so the agent uses
the *identical* embedding, search, and metadata-extraction code as the
normal pipeline.  LangChain only decides *when* to call each tool.

Three tools returned by make_tools():
  - knowledge_base_search  : hybrid vector + lexical search (via RAGEngine)
  - create_zammad_ticket   : creates a training request ticket in Zammad
  - generate_report        : compiles a structured markdown report from retrieved docs
"""

import logging
import os
from typing import List, Optional

from langchain.tools import tool

logger = logging.getLogger(__name__)

_TOOL_RETRIEVE_K = int(os.getenv("AGENT_RETRIEVE_K", "20"))  # match RAG_RETRIEVE_K=20
_TOOL_MIN_SIMILARITY = float(os.getenv("AGENT_MIN_SIMILARITY", "0.2"))  # match normal pipeline


def make_tools(
    engine,  # RAGEngine — passed in so tools share the same instance as normal flow
    user_id: str,
    source_id: Optional[str] = None,
    result_store: Optional[List] = None,
    ticket_store: Optional[List] = None,
) -> List:
    """
    Return the three agent tools bound to the given RAGEngine.

    By accepting the engine (not the raw VectorDatabase), every tool call
    flows through the same get_query_embedding / similarity_search /
    _extract_source_metadata methods that the normal pipeline uses.
    """
    from app.core.enums import DataClassification

    def _search(query: str) -> List:
        """Delegate to RAGEngine — identical embedding + hybrid retrieval."""
        embedding = engine.get_query_embedding(query.strip()[:500])
        return engine.vector_db.similarity_search(
            embedding,
            [DataClassification.PUBLIC.value],
            top_k=_TOOL_RETRIEVE_K,
            min_similarity=_TOOL_MIN_SIMILARITY,
            source_id=source_id,
            query_text=query.strip()[:500],
            retrieval_mode="hybrid",  # same default as the normal SSE pipeline
        )

    def _format_results(results: List) -> str:
        """Format retrieved chunks using RAGEngine's metadata extractor.

        Shows ALL retrieved results so the LLM has access to every chunk,
        including Part 1277 chunks that rank at positions 9-20 (beyond what
        a top-8 cut would include). Header is kept identical to the original
        "Found N document(s):" so the system-prompt Case A trigger fires.
        """
        lines = [f"Found {len(results)} document(s):\n"]
        for i, chunk in enumerate(results, 1):
            meta = engine._extract_source_metadata(chunk)
            # Build a compact metadata line identical to _build_prompt
            meta_parts = []
            for key in ("title", "part", "section", "heading", "source"):
                val = meta.get(key)
                if val not in (None, ""):
                    meta_parts.append(f"{key}={val}")
            sim = meta.get("similarity")
            if isinstance(sim, (float, int)):
                meta_parts.append(f"similarity={float(sim):.3f}")
            metadata_line = "; ".join(meta_parts) if meta_parts else "metadata=none"

            content = (chunk.get("content") or "").strip()[:600]
            lines.append(f"[Source {i}]\nMetadata: {metadata_line}\n{content}\n")
        return "\n".join(lines)

    @tool
    def knowledge_base_search(query: str) -> str:
        """Search the regulatory knowledge base for documents relevant to a query.
        Always call this first before attempting to answer any regulatory question.
        Returns formatted excerpts with section references, or a NO_RESULTS sentinel.
        Input: a plain-text search query (max 500 chars)."""

        results = _search(query)
        if result_store is not None:
            result_store.clear()
            result_store.extend(results)
        if not results:
            return f"NO_RESULTS: No relevant documents found for: {query[:200]}"
        return _format_results(results)

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
            # Store URL in the shared ticket_store so agent.py can read it
            # directly from on_tool_end without relying on LLM answer regex.
            if ticket_store is not None:
                ticket_store.clear()
                ticket_store.append(url)
            return f"TICKET_CREATED: {url}"
        return f"TICKET_FAILED: Could not create ticket, fallback URL: {url}"

    @tool
    def generate_report(topic: str) -> str:
        """Generate a structured markdown report on a regulatory topic.
        Searches the knowledge base and compiles all relevant information.
        Use when the user explicitly asks for a report, summary, or overview.
        Input: the topic or regulation area to report on (max 300 chars)."""

        results = _search(topic[:300])
        if result_store is not None:
            result_store.clear()
            result_store.extend(results)
        if not results:
            return f"NO_RESULTS: No documents found for topic: {topic[:200]}"

        lines = [
            f"# Report: {topic}\n",
            f"*Based on {len(results)} document(s) from the knowledge base.*\n",
        ]
        for chunk in results:
            meta = engine._extract_source_metadata(chunk)
            heading = meta.get("heading") or meta.get("section") or "Unknown"
            content = (chunk.get("content") or "").strip()[:800]
            part = meta.get("part") or ""
            title_num = meta.get("title") or ""
            ref = f"{title_num} CFR Part {part}" if part and title_num else "Unknown"
            lines.append(f"\n## {heading}\n**Reference:** {ref}\n\n{content}")

        return "\n".join(lines)

    return [knowledge_base_search, create_zammad_ticket, generate_report]
