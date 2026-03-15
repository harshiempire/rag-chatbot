"""
LangChain tool-calling agent for the RAG chatbot.

Wraps knowledge_base_search, create_zammad_ticket, and generate_report in a
ReAct-style tool-calling agent that streams SSE events back to the FastAPI route
via an asyncio queue drained by run_agent_stream().

Decision logic (enforced by the system prompt):
  1. ALWAYS call knowledge_base_search first.
  2. Results found → answer grounded, cite section references.
  3. NO_RESULTS → call create_zammad_ticket, then answer ungrounded with disclaimer.
  4. User asks for report/summary → call generate_report instead.
"""

import asyncio
import logging
import os
import re
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.database import VectorDatabase
from app.core.enums import LLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a regulatory compliance assistant with access to a knowledge base of \
federal regulations. You have three tools:

1. knowledge_base_search — search the knowledge base. Call this FIRST for any \
   regulatory question.
2. create_zammad_ticket — create a training ticket when knowledge_base_search \
   returns NO_RESULTS. Call this only after a failed search.
3. generate_report — compile a structured markdown report when the user \
   explicitly asks for a report, summary, or overview.

Decision rules:
- ALWAYS call knowledge_base_search before answering a regulatory question.
- If search returns results: answer grounded in those results. Cite the section \
  references (e.g., § 1282.12).
- If search returns NO_RESULTS: call create_zammad_ticket to log the gap, then \
  answer using general knowledge with this exact disclaimer at the top: \
  "⚠️ Not grounded in trained data — verify with official sources."
- If the user asks for a "report", "summary", or "overview": call generate_report \
  instead of knowledge_base_search.
"""

# ---------------------------------------------------------------------------
# Async callback handler → SSE queue
# ---------------------------------------------------------------------------

_TICKET_URL_RE = re.compile(r"TICKET_CREATED:\s*(https?://\S+)")


class _SSECallbackHandler(AsyncCallbackHandler):
    """Captures agent events and pushes SSE-ready tuples into an asyncio queue."""

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs):
        if token:
            await self.queue.put(("token", {"text": token}))

    async def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs
    ):
        await self.queue.put(
            (
                "tool_call",
                {
                    "tool": serialized.get("name", "unknown"),
                    "input": str(input_str)[:300],
                },
            )
        )

    async def on_tool_end(self, output: str, **kwargs):
        output_str = str(output)
        # Parse result count for knowledge_base_search / generate_report
        count_match = re.search(r"Found (\d+) document", output_str)
        count = int(count_match.group(1)) if count_match else 0
        preview = output_str[:300]
        await self.queue.put(
            ("tool_result", {"output": preview, "count": count})
        )

    async def on_agent_finish(self, finish, **kwargs):
        output: str = finish.return_values.get("output", "")
        is_grounded = (
            "NO_RESULTS" not in output and "TICKET_" not in output
        )
        ticket_link: Optional[str] = None
        m = _TICKET_URL_RE.search(output)
        if m:
            ticket_link = m.group(1)

        await self.queue.put(
            (
                "final",
                {
                    "answer": output,
                    "is_grounded": is_grounded,
                    "ticket_link": ticket_link,
                },
            )
        )


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _build_llm(llm_provider: LLMProvider, model: Optional[str] = None):
    """Build a streaming LangChain chat model for the given provider."""
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
    Run the LangChain agent and yield (event_type, data) tuples.

    Consumed by the FastAPI SSE route to produce the text/event-stream response.
    Yields: status, tool_call, tool_result, token, final, (error)
    """
    from app.rag.agent_tools import make_tools

    # LangChain 0.3+ moved AgentExecutor out of __init__; try both locations
    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
    except ImportError:
        from langchain.agents.agent import AgentExecutor  # type: ignore[no-redef]
        from langchain.agents.tool_calling_agent.base import create_tool_calling_agent  # type: ignore[no-redef]

    queue: asyncio.Queue = asyncio.Queue()
    handler = _SSECallbackHandler(queue)

    tools = make_tools(vector_db, user_id, source_id)
    llm = _build_llm(llm_provider, model)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=5)

    done_event = asyncio.Event()

    async def _run():
        try:
            await executor.ainvoke(
                {"input": question},
                config={"callbacks": [handler]},
            )
        except Exception as exc:
            logger.error("Agent execution error: %s", exc)
            await queue.put(("error", {"message": str(exc)}))
        finally:
            done_event.set()

    asyncio.create_task(_run())

    # Drain queue until agent finishes or error
    while not (done_event.is_set() and queue.empty()):
        try:
            event_type, data = await asyncio.wait_for(queue.get(), timeout=0.1)
            yield event_type, data
            if event_type in ("final", "error"):
                # Drain any remaining buffered events then stop
                while not queue.empty():
                    yield await queue.get()
                break
        except asyncio.TimeoutError:
            continue
