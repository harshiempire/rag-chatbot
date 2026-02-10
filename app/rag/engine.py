"""
RAG (Retrieval-Augmented Generation) engine.

Orchestrates the query pipeline: embed query → search vectors →
build prompt → call LLM → return structured response.
Supports OpenAI, Anthropic, Google Gemini, OpenRouter, and local Ollama.
"""

import json
import logging
import os
from typing import Dict, List

import anthropic
import openai
import requests
from fastapi import HTTPException
from google import genai

from app.core.database import VectorDatabase
from app.core.enums import DataClassification, LLMProvider
from app.core.schemas import RAGQuery, RAGResponse
from app.vectorization.engine import VectorizationEngine

logger = logging.getLogger(__name__)

# RAG prompt caps (to avoid local LLM timeouts)
MAX_TOP_K = int(os.getenv("RAG_MAX_TOP_K", "3"))
MAX_CHUNK_CHARS = int(os.getenv("RAG_MAX_CHUNK_CHARS", "1200"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))


class RAGEngine:
    """RAG system with security controls"""

    def __init__(self, vector_db: VectorDatabase):
        self.vector_db = vector_db
        self.vectorization = None

        # Initialize LLM clients
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

        # Initialize Gemini (new SDK)
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key:
            self.gemini_client = genai.Client(api_key=gemini_key)
        else:
            self.gemini_client = None

    def query(self, rag_query: RAGQuery, user_id: str) -> RAGResponse:
        """Execute RAG query with security"""

        # Validate LLM provider based on classification
        classifications = rag_query.classification_filter or [DataClassification.PUBLIC]
        self._validate_llm_access(rag_query.llm_provider, classifications)

        # Generate query embedding (lazy init to avoid startup blocking)
        if self.vectorization is None:
            self.vectorization = VectorizationEngine()
        query_embedding = self.vectorization.generate_embeddings([rag_query.question])[0]

        # Search similar chunks
        results = self.vector_db.similarity_search(
            query_embedding,
            [c.value for c in classifications],
            rag_query.top_k,
            rag_query.min_similarity
        )

        if not results:
            raise HTTPException(status_code=404, detail="No relevant documents found")

        # Get max classification
        max_class = self._get_max_classification([r['classification'] for r in results])

        # Generate answer
        answer = self._generate_answer(rag_query, results)

        # Audit log
        self.vector_db.log_query(
            user_id, rag_query.question, rag_query.llm_provider.value,
            max_class.value, True
        )

        return RAGResponse(
            answer=answer,
            sources=results,
            classification=max_class,
            llm_provider=rag_query.llm_provider
        )

    def _validate_llm_access(self, llm_provider: LLMProvider, classifications: List[DataClassification]):
        """Ensure restricted data doesn't go to public LLMs"""
        restricted = [DataClassification.RESTRICTED, DataClassification.CONFIDENTIAL]

        if any(c in classifications for c in restricted):
            if llm_provider != LLMProvider.LOCAL:
                raise HTTPException(
                    status_code=403,
                    detail="RESTRICTED/CONFIDENTIAL data can only use local LLM models"
                )

    def _get_max_classification(self, classifications: List[str]) -> DataClassification:
        """Get highest classification level"""
        hierarchy = {'public': 0, 'internal': 1, 'confidential': 2, 'restricted': 3}
        max_level = max(hierarchy.get(c, 0) for c in classifications)
        for name, level in hierarchy.items():
            if level == max_level:
                return DataClassification(name)
        return DataClassification.PUBLIC

    def _build_prompt(self, question: str, context_chunks: List[Dict]) -> str:
        """Build the RAG prompt from context chunks"""
        chunks = context_chunks[:min(len(context_chunks), MAX_TOP_K)]
        parts = []
        total = 0
        for i, chunk in enumerate(chunks):
            text = (chunk.get("content") or "").strip()
            if not text:
                continue
            text = text[:MAX_CHUNK_CHARS]
            block = f"[Source {i+1}]\n{text}"
            if total + len(block) > MAX_CONTEXT_CHARS:
                remaining = max(0, MAX_CONTEXT_CHARS - total)
                if remaining > 200:
                    parts.append(block[:remaining])
                break
            parts.append(block)
            total += len(block)
        context = "\n\n".join(parts)

        return f"""Based on the regulatory documents below, answer the question.

Context:
{context}

Question: {question}

Provide a detailed answer based only on the information above. If you don't know, say so."""

    def _generate_answer(self, query: RAGQuery, context_chunks: List[Dict]) -> str:
        """Generate answer using LLM"""
        prompt = self._build_prompt(query.question, context_chunks)

        if query.llm_provider == LLMProvider.OPENAI:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a regulatory compliance expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=query.temperature
            )
            return response.choices[0].message.content

        elif query.llm_provider == LLMProvider.ANTHROPIC:
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
                temperature=query.temperature
            )
            return response.content[0].text

        elif query.llm_provider == LLMProvider.GOOGLE:
            if not self.gemini_client:
                raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={
                    "temperature": query.temperature,
                    "max_output_tokens": 2048
                }
            )
            return response.text

        elif query.llm_provider == LLMProvider.OPENROUTER:
            openrouter_key = os.getenv('OPENROUTER_API_KEY')
            if not openrouter_key:
                raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

            openrouter_model = os.getenv('OPENROUTER_MODEL', 'deepseek/deepseek-r1-0528:free')

            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openrouter_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:8000",
                        "X-Title": "RAG-Chatbot"
                    },
                    json={
                        "model": openrouter_model,
                        "messages": [
                            {"role": "system", "content": "You are a regulatory compliance expert."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": query.temperature,
                        "max_tokens": 2048
                    },
                    timeout=120
                )
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            except requests.exceptions.Timeout:
                raise HTTPException(status_code=504, detail="OpenRouter timeout")
            except requests.exceptions.HTTPError as e:
                error_detail = e.response.text if e.response else str(e)
                logger.error(f"OpenRouter HTTP error: {error_detail}")
                raise HTTPException(status_code=e.response.status_code if e.response else 500, detail=f"OpenRouter error: {error_detail}")
            except Exception as e:
                logger.error(f"OpenRouter error: {e}")
                raise HTTPException(status_code=500, detail=f"OpenRouter error: {str(e)}")

        elif query.llm_provider == LLMProvider.LOCAL:
            local_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/api/generate")
            local_model = os.getenv("LOCAL_LLM_MODEL", "llama3.1:8b")
            try:
                response = requests.post(
                    local_url,
                    json={"model": local_model, "prompt": prompt, "stream": False},
                    timeout=300
                )
                response.raise_for_status()
                return response.json().get('response', '')
            except requests.exceptions.Timeout:
                raise HTTPException(status_code=504, detail="Ollama timeout. Is it running? Start: ollama serve")
            except requests.exceptions.ConnectionError:
                raise HTTPException(status_code=503, detail=f"Cannot connect to Ollama at {local_url}. Start: ollama serve")
            except Exception as e:
                logger.error(f"Local LLM error: {e}")
                raise HTTPException(status_code=500, detail=f"Local LLM error: {str(e)}")

        raise HTTPException(status_code=400, detail=f"Unsupported LLM: {query.llm_provider}")

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
