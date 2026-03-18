"""
LangChain 1.x tool-calling agent for the RAG chatbot.

Uses langchain.agents.create_agent (LangGraph-backed CompiledStateGraph)
with astream_events for real-time SSE streaming.

Emits the same SSE event sequence as the normal RAG pipeline:
  status(retrieval/start) → source* → status(retrieval/done)
  → status(prompt_build/start) → status(prompt_build/done)
  → status(generation/start) → token* → status(generation/done)
  → final
  (routes.py appends done)

Decision logic (enforced by the system prompt):
  1. ALWAYS call knowledge_base_search first.
  2. Results found → answer grounded, cite section references.
  3. NO_RESULTS → call create_zammad_ticket, then answer ungrounded.
  4. User asks for report/summary → call generate_report instead.
"""

import logging
import os
import re
import time
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage

from app.core.database import VectorDatabase
from app.core.enums import LLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a regulatory compliance assistant with access to a knowledge base of \
federal regulations.

TOOLS (use each at most once per turn):
1. knowledge_base_search — retrieve relevant regulatory documents.
2. create_zammad_ticket — log a knowledge gap. Call ONLY when knowledge_base_search \
   returns a string starting with "NO_RESULTS:".
3. generate_report — structured markdown report. Call ONLY when the user explicitly \
   uses the words "report", "summary", or "overview".

EXACT WORKFLOW — follow these steps in order, no deviations:

Step 1 — Search (choose ONE branch, then proceed to Step 2):

  Branch R — Report request (user explicitly says "report", "summary", or "overview"):
    Call generate_report with the user's question. Skip the other branch entirely.

  Branch S — All other questions:
    Call knowledge_base_search with the user's question. Do this EXACTLY ONCE.
    Do NOT also call generate_report.

Step 2 — Decide based on the tool output:

  Case A — Output starts with "Found N document(s):":
    • Answer immediately using ONLY the returned documents.
    • Cite EVERY relevant section by its exact § number (e.g., § 1277.4, § 1277.6, § 1277.20).
      Always list each section individually — never combine them into ranges like "§ 1277.21–§ 1277.24".
    • Use the exact regulatory terminology and defined terms from the documents
      (e.g., "capital stock", "membership stock", "retained earnings", "board of directors",
      "adequately capitalized", "activity-based capital requirement").
    • Cover ALL relevant sub-topics found across the sources, not just the first one.
    • Do NOT call any more tools.
    • Do NOT add any disclaimer.
    • Do NOT say you could not find information.

  Case B — Output starts with "NO_RESULTS:":
    • Call create_zammad_ticket once to log the knowledge gap.
    • Then answer using your general knowledge — do NOT add any disclaimer or warning text.
      The user interface will automatically display a disclaimer banner.
    • Be concise and factual; note that official sources should be consulted.

IMPORTANT: Never call knowledge_base_search more than once per turn. \
Each branch in Step 1 is mutually exclusive — call exactly one tool in Step 1, never both.
"""

_TICKET_URL_RE = re.compile(r"TICKET_CREATED:\s*(https?://\S+)")

# Tools that do a KB search and should trigger retrieval status + source events
_SEARCH_TOOLS = {"knowledge_base_search", "generate_report"}

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _build_llm(llm_provider: LLMProvider, model: Optional[str] = None):
    """Build a LangChain chat model for the given provider."""
    if llm_provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            streaming=True,
        )
    if llm_provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            streaming=True,
        )
    # Fallback: try OpenAI then Anthropic
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            streaming=True,
        )
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        streaming=True,
    )


# ---------------------------------------------------------------------------
# Source event builder (mirrors routes._build_source_event)
# ---------------------------------------------------------------------------


def _build_source_event(result: Dict, index: int) -> Dict:
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


# ---------------------------------------------------------------------------
# Public streaming entry point
# ---------------------------------------------------------------------------


async def run_agent_stream(
    question: str,
    vector_db: VectorDatabase,
    llm_provider: LLMProvider,
    user_id: str,
    source_id: Optional[str] = None,
    model: Optional[str] = None,
) -> AsyncGenerator[Tuple[str, Dict], None]:
    """
    Run the LangChain 1.x agent and yield (event_type, data) tuples.

    Uses create_agent (LangGraph CompiledStateGraph) with astream_events.

    Yields events that match the normal pipeline:
      status(retrieval/start|done), source, status(prompt_build/start|done),
      status(generation/start|done), token, tool_call, tool_result, final
    """
    from langchain.agents import create_agent

    from app.rag.agent_tools import make_tools
    from app.rag.engine import RAGEngine

    # Share the same RAGEngine instance so tools call the identical
    # embedding, search, and metadata methods as the normal pipeline.
    engine = RAGEngine(vector_db)

    # Shared store — each search tool clears+fills this on every call so we
    # always have the latest raw DB results when on_tool_end fires.
    retrieved_docs: List[Dict] = []

    # Single-element list; create_zammad_ticket writes the URL here so we
    # don't have to rely on fragile regex over the LLM's final answer text.
    auto_ticket_url: List[str] = []

    tools = make_tools(engine, user_id, source_id, retrieved_docs, auto_ticket_url)
    llm = _build_llm(llm_provider, model)

    agent = create_agent(llm, tools, system_prompt=_SYSTEM_PROMPT)

    # -----------------------------------------------------------------------
    # Timing & state tracking
    # -----------------------------------------------------------------------
    total_start = time.perf_counter()
    timings_ms: Dict[str, float] = {}

    retrieval_start: Optional[float] = None
    generation_started = False
    llm_start: Optional[float] = None
    full_answer = ""
    # Guard: only capture LLM tokens AFTER at least one tool has finished.
    # Without this, the first LLM call (tool-selection reasoning) fires
    # on_chat_model_stream too, contaminating full_answer and triggering
    # generation/start before retrieval even begins.
    tools_completed = False

    # NOTE: routes.py already emits status(agent/start) before calling us,
    # so we do NOT emit a duplicate here.

    try:
        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=question)]},
            version="v2",
            config={"recursion_limit": 10},  # allows up to ~4 tool calls while debugging
        ):
            etype = event["event"]
            tool_name: str = event.get("name", "")

            # ---- Tool starts -----------------------------------------------
            if etype == "on_tool_start":
                tool_input = str(event["data"].get("input", ""))[:300]
                logger.info("Agent tool_start: %s | input: %s", tool_name, tool_input[:120])
                yield "tool_call", {"tool": tool_name, "input": tool_input}

                # A new tool call means the LLM is still in orchestration mode,
                # not in final-answer mode. Reset token accumulation so that any
                # tokens captured between the previous tool_end and this tool_start
                # (i.e. inter-tool reasoning) are discarded.
                if generation_started:
                    logger.info(
                        "Agent: new tool '%s' started after generation began — "
                        "discarding %d chars of inter-tool tokens",
                        tool_name, len(full_answer),
                    )
                    # Close the open generation/start with a matching generation/done
                    # so the frontend statusHistory never has a dangling in-progress
                    # "Generating answer" step when additional tools are called.
                    yield "status", {
                        "stage": "generation",
                        "state": "done",
                        "label": "Generating answer (superseded by tool call)",
                        "meta": {
                            "duration_ms": round(
                                (time.perf_counter() - llm_start) * 1000, 2
                            )
                        },
                    }
                    generation_started = False
                    full_answer = ""
                tools_completed = False  # block token capture until this tool ends

                if tool_name in _SEARCH_TOOLS:
                    retrieval_start = time.perf_counter()
                    yield "status", {
                        "stage": "retrieval",
                        "state": "start",
                        "label": "Searching knowledge base",
                    }

            # ---- Tool ends -------------------------------------------------
            elif etype == "on_tool_end":
                # In LangChain 1.x, on_tool_end may wrap the return value in a
                # ToolMessage. Extract .content so we get the plain string.
                raw_output = event["data"].get("output", "")
                output = (
                    raw_output.content
                    if hasattr(raw_output, "content")
                    else str(raw_output)
                )
                logger.info(
                    "Agent tool_end: %s | output[:120]: %s",
                    tool_name, output[:120].replace("\n", " "),
                )

                if tool_name in _SEARCH_TOOLS:
                    search_ms = round(
                        (time.perf_counter() - (retrieval_start or total_start)) * 1000,
                        2,
                    )
                    timings_ms["search"] = search_ms

                    if retrieved_docs:
                        # Emit source events from the shared store populated by the tool
                        for idx, doc in enumerate(retrieved_docs, 1):
                            yield "source", _build_source_event(doc, idx)
                        yield "status", {
                            "stage": "retrieval",
                            "state": "done",
                            "label": f"Retrieved {len(retrieved_docs)} chunk(s)",
                            "meta": {
                                "duration_ms": search_ms,
                                "retrieved_count": len(retrieved_docs),
                            },
                        }
                    else:
                        # NO_RESULTS path — mirror the normal pipeline's retrieval/done label
                        yield "status", {
                            "stage": "retrieval",
                            "state": "done",
                            "label": "No documents found — falling back to ungrounded answer",
                            "meta": {"duration_ms": search_ms, "retrieved_count": 0},
                        }

                    # Prompt building is handled internally by the agent;
                    # emit start+done as a near-zero pass-through so the UI
                    # shows the same stages as the normal pipeline.
                    yield "status", {
                        "stage": "prompt_build",
                        "state": "start",
                        "label": "Building prompt context",
                    }
                    yield "status", {
                        "stage": "prompt_build",
                        "state": "done",
                        "label": "Prompt ready",
                        "meta": {"duration_ms": 0.0},
                    }

                elif tool_name == "create_zammad_ticket":
                    # Capture ticket URL from tool output directly — no regex on full_answer.
                    # Output is "TICKET_CREATED: <url>" or "TICKET_FAILED: ..." or "TICKET_SKIPPED: ..."
                    m_ticket = re.search(r"TICKET_CREATED:\s*(https?://\S+)", output)
                    if m_ticket:
                        captured_url = m_ticket.group(1)
                        auto_ticket_url.clear()
                        auto_ticket_url.append(captured_url)
                        logger.info("Agent ticket created: %s", captured_url)
                    # Emit a timeline entry so the user can see this in the pipeline steps
                    ticket_status_label = (
                        "Knowledge gap logged — ticket created"
                        if auto_ticket_url
                        else "Knowledge gap logged (ticket unavailable)"
                    )
                    yield "status", {
                        "stage": "agent",
                        "state": "done",
                        "label": ticket_status_label,
                        "meta": {"ticket_url": auto_ticket_url[0] if auto_ticket_url else None},
                    }

                count_match = re.search(r"Found (\d+) document", output)
                count = int(count_match.group(1)) if count_match else 0
                yield "tool_result", {"output": output[:300], "count": count}
                tools_completed = True  # any tool done → next LLM call is final answer

            # ---- LLM streaming tokens (final answer only) ------------------
            elif etype == "on_chat_model_stream" and tools_completed:
                chunk = event["data"]["chunk"]
                token = chunk.content if isinstance(chunk.content, str) else ""
                if token:
                    if not generation_started:
                        generation_started = True
                        llm_start = time.perf_counter()
                        yield "status", {
                            "stage": "generation",
                            "state": "start",
                            "label": "Generating answer",
                        }
                    full_answer += token
                    yield "token", {"text": token}

    except Exception as exc:
        logger.error("Agent execution error: %s", exc)
        yield "error", {"code": "AGENT_ERROR", "message": str(exc)}
        return

    # -----------------------------------------------------------------------
    # Final event — mirrors the normal pipeline's final payload
    # -----------------------------------------------------------------------
    # llm_start is set on the first token of the final answer; if no tokens
    # arrived (edge case), fall back to total time.
    llm_ms = round((time.perf_counter() - (llm_start or total_start)) * 1000, 2)
    total_ms = round((time.perf_counter() - total_start) * 1000, 2)
    timings_ms["llm"] = llm_ms
    timings_ms["total"] = total_ms

    if generation_started:
        yield "status", {
            "stage": "generation",
            "state": "done",
            "label": "Generation complete",
            "meta": {"duration_ms": llm_ms},
        }

    logger.info(
        "Agent final: retrieved_docs=%d, full_answer[:200]=%s",
        len(retrieved_docs), full_answer[:200].replace("\n", " "),
    )
    is_grounded = bool(retrieved_docs) and "NO_RESULTS" not in full_answer

    # Prefer URL captured directly from create_zammad_ticket tool output.
    # Fall back to regex scan of answer text, then to TICKET_SUBMIT_URL so
    # the "Request training" button always has a destination when ungrounded.
    if auto_ticket_url:
        ticket_link: Optional[str] = auto_ticket_url[0]
    else:
        m = _TICKET_URL_RE.search(full_answer)
        if m:
            ticket_link = m.group(1)
        elif not is_grounded:
            # Ensure the frontend button always has somewhere to point even if
            # Zammad is down — mirror the normal pipeline's TICKET_SUBMIT_URL fallback.
            from app.config import TICKET_SUBMIT_URL as _FALLBACK_URL
            ticket_link = _FALLBACK_URL or None
        else:
            ticket_link = None

    # The agent forwards ALL retrieved docs to the LLM via the tool output —
    # no RAG_PROMPT_K cap applies here (unlike the normal pipeline which builds
    # a trimmed prompt).  Report the true count so analytics reflect reality.
    prompt_context_count = len(retrieved_docs)

    yield "final", {
        "answer": full_answer,
        "is_grounded": is_grounded,
        "ticket_link": ticket_link,
        "timings_ms": timings_ms,
        "retrieved_count": len(retrieved_docs),
        "prompt_context_count": prompt_context_count,
    }
