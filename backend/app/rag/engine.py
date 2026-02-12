"""
RAG (Retrieval-Augmented Generation) engine.

Orchestrates the query pipeline: embed query → search vectors →
build prompt → call LLM → return structured response.
Supports OpenAI, Anthropic, Google Gemini, OpenRouter, and local Ollama.
"""

import json
import logging
import os
import time
from typing import Dict, List

import anthropic
import openai
import requests
from fastapi import HTTPException
from google import genai

from app.core.database import VectorDatabase
from app.core.enums import DataClassification, LLMProvider
from app.core.schemas import RAGQuery, RAGResponse
from app.vectorization.engine import DEFAULT_EMBEDDING_MODEL, VectorizationEngine

logger = logging.getLogger(__name__)

# Retrieval/prompt caps
RAG_RETRIEVE_K = int(os.getenv("RAG_RETRIEVE_K", "8"))
RAG_PROMPT_K = int(os.getenv("RAG_PROMPT_K", "3"))
MAX_CHUNK_CHARS = int(os.getenv("RAG_MAX_CHUNK_CHARS", "1200"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))
RAG_MIN_DISTINCT_SECTIONS = int(os.getenv("RAG_MIN_DISTINCT_SECTIONS", "2"))


class RAGEngine:
    """RAG system with security controls"""

    def __init__(self, vector_db: VectorDatabase):
        self.vector_db = vector_db
        self.vectorization = None

        # Initialize LLM clients
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        # Initialize Gemini (new SDK)
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            self.gemini_client = genai.Client(api_key=gemini_key)
        else:
            self.gemini_client = None

    def query(self, rag_query: RAGQuery, user_id: str) -> RAGResponse:
        """Execute RAG query with security"""
        total_start = time.perf_counter()
        timings_ms: Dict[str, float] = {}

        # Validate LLM provider based on classification
        classifications = rag_query.classification_filter or [DataClassification.PUBLIC]
        self._validate_llm_access(rag_query.llm_provider, classifications)

        embed_start = time.perf_counter()
        query_embedding = self.get_query_embedding(rag_query.question)
        timings_ms["embed"] = round((time.perf_counter() - embed_start) * 1000, 2)

        # Search similar chunks
        search_start = time.perf_counter()
        retrieval_k = min(max(rag_query.top_k, 1), RAG_RETRIEVE_K)
        results = self.vector_db.similarity_search(
            query_embedding,
            [c.value for c in classifications],
            retrieval_k,
            rag_query.min_similarity,
            source_id=rag_query.source_id,
        )
        timings_ms["search"] = round((time.perf_counter() - search_start) * 1000, 2)

        if not results:
            raise HTTPException(status_code=404, detail="No relevant documents found")

        # Get max classification
        max_class = self._get_max_classification([r["classification"] for r in results])

        prompt_start = time.perf_counter()
        prompt, prompt_context_count = self._build_prompt(rag_query.question, results)
        timings_ms["prompt_build"] = round(
            (time.perf_counter() - prompt_start) * 1000, 2
        )

        llm_start = time.perf_counter()
        answer = self._generate_answer(rag_query, prompt)
        timings_ms["llm"] = round((time.perf_counter() - llm_start) * 1000, 2)
        total_ms = round((time.perf_counter() - total_start) * 1000, 2)
        timings_ms["total"] = total_ms

        # Audit log
        self.vector_db.log_query(
            user_id,
            rag_query.question,
            rag_query.llm_provider.value,
            max_class.value,
            True,
            details={
                "retrieved_count": len(results),
                "prompt_context_count": prompt_context_count,
                "source_id": rag_query.source_id,
                "timings_ms": timings_ms,
            },
        )

        return RAGResponse(
            answer=answer,
            sources=results,
            classification=max_class,
            llm_provider=rag_query.llm_provider,
            retrieved_count=len(results),
            prompt_context_count=prompt_context_count,
            total_ms=total_ms,
            timings_ms=timings_ms,
        )

    def _validate_llm_access(
        self, llm_provider: LLMProvider, classifications: List[DataClassification]
    ):
        """Ensure restricted data doesn't go to public LLMs"""
        restricted = [DataClassification.RESTRICTED, DataClassification.CONFIDENTIAL]

        if any(c in classifications for c in restricted):
            if llm_provider != LLMProvider.LOCAL:
                raise HTTPException(
                    status_code=403,
                    detail="RESTRICTED/CONFIDENTIAL data can only use local LLM models",
                )

    def _get_max_classification(self, classifications: List[str]) -> DataClassification:
        """Get highest classification level"""
        hierarchy = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
        max_level = max(hierarchy.get(c, 0) for c in classifications)
        for name, level in hierarchy.items():
            if level == max_level:
                return DataClassification(name)
        return DataClassification.PUBLIC

    def get_query_embedding(self, question: str) -> List[float]:
        """Generate query embedding with canonical embedding model."""
        if self.vectorization is None:
            self.vectorization = VectorizationEngine(DEFAULT_EMBEDDING_MODEL)
        return self.vectorization.generate_embeddings([question])[0]

    @staticmethod
    def _pick_metadata_value(chunk_metadata: Dict, doc_metadata: Dict, *keys):
        for key in keys:
            chunk_val = chunk_metadata.get(key)
            if chunk_val not in (None, ""):
                return chunk_val
            doc_val = doc_metadata.get(key)
            if doc_val not in (None, ""):
                return doc_val
        return None

    def _extract_source_metadata(self, chunk: Dict) -> Dict:
        chunk_metadata = chunk.get("chunk_metadata") or {}
        doc_metadata = chunk.get("doc_metadata") or {}
        return {
            "source_id": chunk.get("source_id"),
            "source": self._pick_metadata_value(chunk_metadata, doc_metadata, "source"),
            "title": self._pick_metadata_value(chunk_metadata, doc_metadata, "title"),
            "chapter": self._pick_metadata_value(
                chunk_metadata, doc_metadata, "chapter"
            ),
            "part": self._pick_metadata_value(chunk_metadata, doc_metadata, "part"),
            "section": self._pick_metadata_value(
                chunk_metadata, doc_metadata, "section"
            ),
            "heading": self._pick_metadata_value(
                chunk_metadata, doc_metadata, "heading", "section_header"
            ),
            "similarity": chunk.get("similarity"),
        }

    def _build_prompt(
        self, question: str, context_chunks: List[Dict]
    ) -> tuple[str, int]:
        """Build the RAG prompt from context chunks"""
        chunks = context_chunks[: min(len(context_chunks), RAG_PROMPT_K)]
        parts = []
        section_keys = set()
        total = 0
        used = 0
        for i, chunk in enumerate(chunks):
            text = (chunk.get("content") or "").strip()
            if not text:
                continue

            metadata = self._extract_source_metadata(chunk)
            # section_keys.add logic moved to after capacity check

            metadata_items = []
            for field in [
                "source_id",
                "title",
                "chapter",
                "part",
                "section",
                "heading",
                "source",
            ]:
                value = metadata.get(field)
                if value not in (None, ""):
                    metadata_items.append(f"{field}={value}")
            similarity = metadata.get("similarity")
            if isinstance(similarity, (float, int)):
                metadata_items.append(f"similarity={float(similarity):.3f}")

            text = text[:MAX_CHUNK_CHARS]
            metadata_line = (
                "; ".join(metadata_items) if metadata_items else "metadata=none"
            )
            block = f"[Source {i + 1}]\nMetadata: {metadata_line}\n{text}"
            if total + len(block) > MAX_CONTEXT_CHARS:
                remaining = max(0, MAX_CONTEXT_CHARS - total)
                if remaining > 200:
                    parts.append(block[:remaining])
                    used += 1
                    section_key_parts = [
                        str(metadata.get("chapter") or ""),
                        str(metadata.get("part") or ""),
                        str(metadata.get("section") or metadata.get("heading") or ""),
                    ]
                    if any(section_key_parts):
                        section_keys.add("|".join(section_key_parts))
                break
            parts.append(block)
            total += len(block)
            used += 1
            section_key_parts = [
                str(metadata.get("chapter") or ""),
                str(metadata.get("part") or ""),
                str(metadata.get("section") or metadata.get("heading") or ""),
            ]
            if any(section_key_parts):
                section_keys.add("|".join(section_key_parts))
        context = "\n\n".join(parts)

        coverage_note = ""
        distinct_sections = len(section_keys)
        if used > 0 and distinct_sections < RAG_MIN_DISTINCT_SECTIONS:
            coverage_note = (
                "\nCoverage Note:\n"
                f"- Retrieved context spans only {distinct_sections} distinct section(s).\n"
                "- Provide a partial answer, explicitly state coverage is limited, and avoid chapter-wide claims.\n"
            )
        prompt = f"""Based on the regulatory documents below, answer the question.

Context:
{context}
{coverage_note}

Question: {question}

Provide a detailed answer based only on the information above. If you don't know, say so."""
        return prompt, used

    def _generate_answer(self, query: RAGQuery, prompt: str) -> str:
        """Generate answer using LLM"""
        if query.llm_provider == LLMProvider.OPENAI:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a regulatory compliance expert.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=query.temperature,
            )
            return response.choices[0].message.content

        elif query.llm_provider == LLMProvider.ANTHROPIC:
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
                temperature=query.temperature,
            )
            return response.content[0].text

        elif query.llm_provider == LLMProvider.GOOGLE:
            if not self.gemini_client:
                raise HTTPException(
                    status_code=500, detail="GEMINI_API_KEY not configured"
                )
            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={"temperature": query.temperature, "max_output_tokens": 2048},
            )
            return response.text

        elif query.llm_provider == LLMProvider.OPENROUTER:
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_key:
                raise HTTPException(
                    status_code=500, detail="OPENROUTER_API_KEY not configured"
                )

            openrouter_model = os.getenv(
                "OPENROUTER_MODEL", "deepseek/deepseek-r1-0528:free"
            )

            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openrouter_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:8000",
                        "X-Title": "RAG-Chatbot",
                    },
                    json={
                        "model": openrouter_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a regulatory compliance expert.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": query.temperature,
                        "max_tokens": 2048,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except requests.exceptions.Timeout:
                raise HTTPException(status_code=504, detail="OpenRouter timeout")
            except requests.exceptions.HTTPError as e:
                error_detail = e.response.text if e.response else str(e)
                logger.error(f"OpenRouter HTTP error: {error_detail}")
                raise HTTPException(
                    status_code=e.response.status_code if e.response else 500,
                    detail=f"OpenRouter error: {error_detail}",
                )
            except Exception as e:
                logger.error(f"OpenRouter error: {e}")
                raise HTTPException(
                    status_code=500, detail=f"OpenRouter error: {str(e)}"
                )

        elif query.llm_provider == LLMProvider.LOCAL:
            local_url = os.getenv(
                "LOCAL_LLM_URL", "http://localhost:11434/api/generate"
            )
            local_model = os.getenv("LOCAL_LLM_MODEL", "llama3.1:8b")
            try:
                response = requests.post(
                    local_url,
                    json={"model": local_model, "prompt": prompt, "stream": False},
                    timeout=300,
                )
                response.raise_for_status()
                return response.json().get("response", "")
            except requests.exceptions.Timeout:
                raise HTTPException(
                    status_code=504,
                    detail="Ollama timeout. Is it running? Start: ollama serve",
                )
            except requests.exceptions.ConnectionError:
                raise HTTPException(
                    status_code=503,
                    detail=f"Cannot connect to Ollama at {local_url}. Start: ollama serve",
                )
            except Exception as e:
                logger.error(f"Local LLM error: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Local LLM error: {str(e)}"
                )

        raise HTTPException(
            status_code=400, detail=f"Unsupported LLM: {query.llm_provider}"
        )

    def _ollama_stream(self, prompt: str):
        """Stream response from local Ollama LLM"""
        local_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/api/generate")
        local_model = os.getenv("LOCAL_LLM_MODEL", "llama3.1:8b")
        timeout = float(os.getenv("LOCAL_LLM_STREAM_TIMEOUT", "600"))

        try:
            with requests.post(
                local_url,
                json={"model": local_model, "prompt": prompt, "stream": True},
                stream=True,
                timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        yield line
                        continue
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
        except requests.exceptions.Timeout:
            yield "\n\n[Error] Ollama timeout while streaming. Try reducing context further.\n"
        except requests.exceptions.ConnectionError:
            yield f"\n\n[Error] Cannot connect to Ollama at {local_url}. Ensure `ollama serve` is running.\n"
        except Exception as e:
            yield f"\n\n[Error] Local LLM streaming error: {str(e)}\n"

    def _openrouter_stream(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048
    ):
        """Stream response from OpenRouter (OpenAI-compatible SSE)."""
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            yield "\n\n[Error] OPENROUTER_API_KEY not configured.\n"
            return

        openrouter_model = os.getenv(
            "OPENROUTER_MODEL", "deepseek/deepseek-r1-0528:free"
        )
        timeout = float(os.getenv("OPENROUTER_STREAM_TIMEOUT", "300"))

        try:
            with requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "RAG-Chatbot",
                },
                json={
                    "model": openrouter_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a regulatory compliance expert.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
                stream=True,
                timeout=timeout,
            ) as resp:
                resp.raise_for_status()

                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue

                    line = line.strip()
                    # OpenRouter may send SSE comment keep-alives like ": OPENROUTER PROCESSING"
                    if line.startswith(":"):
                        continue

                    payload = (
                        line[len("data:") :].strip()
                        if line.startswith("data:")
                        else line
                    )
                    if not payload or payload == "[DONE]":
                        if payload == "[DONE]":
                            break
                        continue

                    try:
                        data = json.loads(payload)
                    except Exception:
                        continue

                    error = data.get("error")
                    if error:
                        message = (
                            error.get("message")
                            if isinstance(error, dict)
                            else str(error)
                        )
                        yield f"\n\n[Error] OpenRouter stream error: {message}\n"
                        break

                    choices = data.get("choices") or []
                    if not choices:
                        continue

                    choice = choices[0] or {}
                    delta = choice.get("delta") or {}
                    token = delta.get("content", "")
                    if token:
                        yield token

                    if choice.get("finish_reason") == "error":
                        yield "\n\n[Error] OpenRouter stream terminated with finish_reason=error.\n"
                        break
        except requests.exceptions.Timeout:
            yield "\n\n[Error] OpenRouter timeout while streaming.\n"
        except requests.exceptions.ConnectionError:
            yield "\n\n[Error] Cannot connect to OpenRouter.\n"
        except requests.exceptions.HTTPError as e:
            detail = e.response.text if e.response is not None else str(e)
            yield f"\n\n[Error] OpenRouter HTTP error: {detail}\n"
        except Exception as e:
            yield f"\n\n[Error] OpenRouter streaming error: {str(e)}\n"
