"""
Vector database layer using PostgreSQL + pgvector.

Manages connection pooling, schema initialization, document storage,
vector similarity search, and audit logging.
"""

import json
import logging
import os
import re
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg2
from psycopg2 import pool
from pgvector.psycopg2 import register_vector

logger = logging.getLogger(__name__)
ALLOWED_METADATA_FILTER_KEYS = {
    "title",
    "chapter",
    "part",
    "section",
    "heading",
    "source",
    "type",
    "date",
}
FUZZY_METADATA_FILTER_KEYS = {"heading"}
LEXICAL_QUERY_STOPWORDS = {
    "the",
    "and",
    "or",
    "in",
    "on",
    "to",
    "of",
    "for",
    "with",
    "by",
    "from",
    "under",
    "this",
    "that",
    "these",
    "those",
    "what",
    "which",
    "who",
    "can",
    "could",
    "does",
    "is",
    "are",
    "be",
    "vs",
    "versus",
    "compare",
    "comparison",
}


class VectorDatabase:
    """PostgreSQL with pgvector for embeddings storage"""

    def __init__(self, connection_string: str, min_conn: int = 2, max_conn: int = 10):
        self.conn_string = connection_string
        # Use connection pooling for better performance
        self._pool = pool.ThreadedConnectionPool(min_conn, max_conn, connection_string)
        logger.info(f"✅ Connection pool initialized (min={min_conn}, max={max_conn})")
        self.setup_database()

    @contextmanager
    def get_connection(self):
        """Get database connection from pool with pgvector registered"""
        conn = self._pool.getconn()
        try:
            register_vector(conn)
            yield conn
        finally:
            self._pool.putconn(conn)

    def get_connection_simple(self):
        """Legacy: Get connection without context manager (for compatibility)"""
        conn = self._pool.getconn()
        register_vector(conn)
        return conn

    def return_connection(self, conn):
        """Return connection to pool"""
        self._pool.putconn(conn)

    def setup_database(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cur = conn.cursor()

            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

            # Documents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY,
                    source_id VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    classification VARCHAR(50) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    reviewed_at TIMESTAMP,
                    reviewed_by VARCHAR(255),
                    pii_detected BOOLEAN DEFAULT FALSE,
                    CONSTRAINT chk_classification CHECK (classification IN ('public', 'internal', 'confidential', 'restricted')),
                    CONSTRAINT chk_status CHECK (status IN ('pending_extraction', 'extracted', 'pending_review', 'approved', 'rejected', 'vectorized', 'published'))
                )
            """)

            # Vector chunks table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vector_chunks (
                    id UUID PRIMARY KEY,
                    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    embedding vector(384),
                    metadata JSONB,
                    chunk_index INTEGER,
                    classification VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Create vector similarity index
            cur.execute("""
                CREATE INDEX IF NOT EXISTS vector_chunks_embedding_idx 
                ON vector_chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)

            # Audit log table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    timestamp TIMESTAMP DEFAULT NOW(),
                    user_id VARCHAR(255),
                    action VARCHAR(100),
                    document_id UUID,
                    query TEXT,
                    llm_provider VARCHAR(50),
                    classification VARCHAR(50),
                    success BOOLEAN,
                    details JSONB
                )
            """)

            # Users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    email VARCHAR(255) NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Refresh tokens table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS refresh_tokens (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token_hash VARCHAR(128) NOT NULL UNIQUE,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    revoked_at TIMESTAMP
                )
            """)

            # Chat sessions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    llm_provider VARCHAR(50) NOT NULL,
                    created_at BIGINT NOT NULL,
                    updated_at BIGINT NOT NULL,
                    messages JSONB NOT NULL DEFAULT '[]'::jsonb
                )
            """)
            cur.execute(
                """
                SELECT data_type
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'chat_sessions'
                  AND column_name = 'id'
                LIMIT 1
                """
            )
            id_column = cur.fetchone()
            if id_column and id_column[0] != "text":
                logger.info("Migrating chat_sessions.id column to TEXT")
                cur.execute(
                    "ALTER TABLE chat_sessions ALTER COLUMN id TYPE TEXT USING id::text"
                )

            # Create indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_classification ON documents(classification)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_vector_chunks_doc ON vector_chunks(document_id)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_vector_chunks_content_fts "
                "ON vector_chunks USING GIN (to_tsvector('english', content))"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin "
                "ON documents USING GIN (metadata)"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user_id ON refresh_tokens(user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_refresh_tokens_expires_at ON refresh_tokens(expires_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at ON chat_sessions(updated_at DESC)")

            conn.commit()
            cur.close()

        logger.info("✅ Database initialized with pgvector")

    def store_document(self, doc_id: str, source_id: str, content: str,
                      metadata: Dict, classification: str, status: str, pii_detected: bool):
        """Store document"""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO documents (id, source_id, content, metadata, classification, status, pii_detected)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE 
                SET content = EXCLUDED.content, 
                    status = EXCLUDED.status,
                    metadata = EXCLUDED.metadata
            """, (doc_id, source_id, content, json.dumps(metadata), classification, status, pii_detected))
            conn.commit()
            cur.close()

    def store_vector_chunk(self, chunk_id: str, doc_id: str, content: str,
                          embedding: List[float], metadata: Dict, chunk_index: int, classification: str):
        """Store vector chunk"""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO vector_chunks (id, document_id, content, embedding, metadata, chunk_index, classification)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (chunk_id, doc_id, content, np.array(embedding), json.dumps(metadata), chunk_index, classification))
            conn.commit()
            cur.close()

    @staticmethod
    def _parse_json_metadata(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    def _row_to_similarity_result(self, row: Any) -> Dict[str, Any]:
        lexical_score = (
            float(row[7])
            if len(row) > 7 and row[7] is not None
            else None
        )
        return {
            "chunk_id": str(row[0]),
            "content": row[1],
            "chunk_metadata": self._parse_json_metadata(row[2]),
            "doc_metadata": self._parse_json_metadata(row[3]),
            "source_id": row[4],
            "classification": row[5],
            "similarity": float(row[6]),
            "lexical_score": lexical_score,
        }

    @staticmethod
    def _normalize_metadata_filters(
        metadata_filters: Optional[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        if not isinstance(metadata_filters, dict):
            return {}

        normalized: Dict[str, List[str]] = {}
        for raw_key, raw_value in metadata_filters.items():
            key = str(raw_key or "").strip().lower()
            if key not in ALLOWED_METADATA_FILTER_KEYS:
                continue

            values: List[str] = []
            if isinstance(raw_value, (list, tuple, set)):
                for item in raw_value:
                    item_text = str(item).strip().lower()
                    if item_text:
                        values.append(item_text)
            else:
                value_text = str(raw_value).strip().lower()
                if value_text:
                    values.append(value_text)

            if values:
                normalized[key] = values

        return normalized

    @staticmethod
    def _extract_lexical_terms(text: str) -> List[str]:
        tokens = re.findall(r"[a-z]{3,}|\d{2,4}", (text or "").lower())
        return [token for token in tokens if token not in LEXICAL_QUERY_STOPWORDS]

    @classmethod
    def _build_relaxed_lexical_query(
        cls,
        query_text: str,
        normalized_filters: Dict[str, List[str]],
    ) -> str:
        candidates: List[str] = cls._extract_lexical_terms(query_text)
        for filter_key in ("part", "section", "chapter", "title"):
            for raw_value in normalized_filters.get(filter_key, []):
                candidates.extend(cls._extract_lexical_terms(str(raw_value)))

        unique_terms: List[str] = []
        seen = set()
        for token in candidates:
            if token in seen:
                continue
            seen.add(token)
            unique_terms.append(token)

        if not unique_terms:
            return ""
        return " OR ".join(unique_terms[:12])

    @staticmethod
    def _merge_ranked_results(
        primary: List[Dict[str, Any]],
        secondary: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen = set()
        for result in [*primary, *secondary]:
            chunk_id = result.get("chunk_id")
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            merged.append(result)
            if len(merged) >= top_k:
                break
        return merged[:top_k]

    def _build_metadata_filter_clause(
        self, normalized_filters: Dict[str, List[str]]
    ) -> tuple[str, List[Any]]:
        if not normalized_filters:
            return "", []

        clauses: List[str] = []
        params: List[Any] = []
        for key, values in normalized_filters.items():
            if key == "section":
                # Prefer chunk-level section markers over document-level section metadata.
                # This avoids false matches where a single document row contains content from
                # adjacent sections (e.g., 1240.22 text under doc section=1240.20).
                field_expr = (
                    "LOWER(COALESCE("
                    "vc.metadata->>'section',"
                    "substring(vc.metadata->>'section_header' from '([0-9]+\\.[0-9]+[a-z]?)'),"
                    "substring(vc.metadata->>'heading' from '([0-9]+\\.[0-9]+[a-z]?)'),"
                    "d.metadata->>'section',"
                    "substring(d.metadata->>'heading' from '([0-9]+\\.[0-9]+[a-z]?)'),"
                    "''"
                    "))"
                )
            else:
                field_expr = (
                    f"LOWER(COALESCE(vc.metadata->>'{key}', d.metadata->>'{key}', ''))"
                )
            if key in FUZZY_METADATA_FILTER_KEYS:
                like_clauses: List[str] = []
                for value in values:
                    like_clauses.append(f"{field_expr} LIKE %s")
                    params.append(f"%{value}%")
                if like_clauses:
                    clauses.append(f"({' OR '.join(like_clauses)})")
            elif len(values) == 1:
                clauses.append(f"{field_expr} = %s")
                params.append(values[0])
            else:
                clauses.append(f"{field_expr} = ANY(%s)")
                params.append(values)

        if not clauses:
            return "", []
        return " AND " + " AND ".join(clauses), params

    def _dense_similarity_search(
        self,
        cur: Any,
        query_embedding: List[float],
        classifications: List[str],
        top_k: int,
        min_similarity: float,
        source_id: Optional[str],
        normalized_filters: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        filter_clause, filter_params = self._build_metadata_filter_clause(
            normalized_filters
        )
        query_params: List[Any] = [
            np.array(query_embedding),
            classifications,
            source_id,
            source_id,
            np.array(query_embedding),
            min_similarity,
        ]
        query_params.extend(filter_params)
        query_params.extend([np.array(query_embedding), top_k])

        cur.execute(
            f"""
            SELECT
                vc.id,
                vc.content,
                vc.metadata,
                d.metadata as doc_metadata,
                d.source_id,
                vc.classification,
                1 - (vc.embedding <=> %s::vector) as similarity,
                NULL::double precision as lexical_score
            FROM vector_chunks vc
            JOIN documents d ON vc.document_id = d.id
            WHERE vc.classification = ANY(%s)
              AND d.status = 'published'
              AND (%s IS NULL OR d.source_id = %s)
              AND 1 - (vc.embedding <=> %s::vector) >= %s
              {filter_clause}
            ORDER BY vc.embedding <=> %s::vector
            LIMIT %s
            """,
            query_params,
        )
        return [self._row_to_similarity_result(row) for row in cur.fetchall()]

    def _lexical_similarity_search(
        self,
        cur: Any,
        query_embedding: List[float],
        query_text: str,
        classifications: List[str],
        top_k: int,
        source_id: Optional[str],
        normalized_filters: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        filter_clause, filter_params = self._build_metadata_filter_clause(
            normalized_filters
        )
        query_params: List[Any] = [
            np.array(query_embedding),
            query_text,
            classifications,
            source_id,
            source_id,
            query_text,
        ]
        query_params.extend(filter_params)
        query_params.extend([np.array(query_embedding), top_k])

        cur.execute(
            f"""
            SELECT
                vc.id,
                vc.content,
                vc.metadata,
                d.metadata as doc_metadata,
                d.source_id,
                vc.classification,
                1 - (vc.embedding <=> %s::vector) as similarity,
                ts_rank_cd(
                    to_tsvector('english', vc.content),
                    websearch_to_tsquery('english', %s)
                ) as lexical_score
            FROM vector_chunks vc
            JOIN documents d ON vc.document_id = d.id
            WHERE vc.classification = ANY(%s)
              AND d.status = 'published'
              AND (%s IS NULL OR d.source_id = %s)
              AND to_tsvector('english', vc.content) @@ websearch_to_tsquery('english', %s)
              {filter_clause}
            ORDER BY lexical_score DESC, vc.embedding <=> %s::vector
            LIMIT %s
            """,
            query_params,
        )
        return [self._row_to_similarity_result(row) for row in cur.fetchall()]

    @staticmethod
    def _fuse_dense_and_lexical(
        dense_results: List[Dict[str, Any]],
        lexical_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not dense_results:
            return lexical_results[:top_k]
        if not lexical_results:
            return dense_results[:top_k]

        rank_constant_raw = os.getenv("RAG_RRF_K", "60")
        dual_channel_boost_raw = os.getenv("RAG_HYBRID_DUAL_CHANNEL_BOOST", "0.05")
        try:
            rank_constant = max(int(rank_constant_raw), 1)
        except ValueError:
            rank_constant = 60
        try:
            dual_channel_boost = max(float(dual_channel_boost_raw), 0.0)
        except ValueError:
            dual_channel_boost = 0.05

        fused: Dict[str, Dict[str, Any]] = {}
        for rank, result in enumerate(dense_results, start=1):
            chunk_id = result["chunk_id"]
            fused[chunk_id] = {
                "result": result.copy(),
                "hybrid_score": 1.0 / (rank_constant + rank),
                "channels": {"dense"},
            }

        for rank, result in enumerate(lexical_results, start=1):
            chunk_id = result["chunk_id"]
            addition = 1.0 / (rank_constant + rank)
            if chunk_id not in fused:
                fused[chunk_id] = {
                    "result": result.copy(),
                    "hybrid_score": addition,
                    "channels": {"lexical"},
                }
                continue

            fused_entry = fused[chunk_id]
            fused_entry["hybrid_score"] += addition
            fused_entry["channels"].add("lexical")
            fused_entry["result"]["similarity"] = max(
                float(fused_entry["result"].get("similarity", 0.0)),
                float(result.get("similarity", 0.0)),
            )
            lexical_score = result.get("lexical_score")
            if lexical_score is not None:
                existing_lexical = fused_entry["result"].get("lexical_score")
                if existing_lexical is None:
                    fused_entry["result"]["lexical_score"] = float(lexical_score)
                else:
                    fused_entry["result"]["lexical_score"] = max(
                        float(existing_lexical), float(lexical_score)
                    )

        ranked: List[Dict[str, Any]] = []
        for fused_entry in fused.values():
            channels = fused_entry["channels"]
            hybrid_score = fused_entry["hybrid_score"]
            if len(channels) > 1:
                hybrid_score += dual_channel_boost
            result = fused_entry["result"]
            result["hybrid_score"] = hybrid_score
            ranked.append(result)

        ranked.sort(
            key=lambda item: (
                float(item.get("hybrid_score", 0.0)),
                float(item.get("similarity", 0.0)),
                float(item.get("lexical_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return ranked[:top_k]

    @staticmethod
    def _extract_part_from_result(result: Dict[str, Any]) -> str:
        chunk_metadata = result.get("chunk_metadata") or {}
        doc_metadata = result.get("doc_metadata") or {}
        for source in (chunk_metadata, doc_metadata):
            value = source.get("part")
            if value in (None, ""):
                continue
            return str(value).strip().lower()
        return ""

    @staticmethod
    def _extract_section_key_from_result(result: Dict[str, Any]) -> str:
        chunk_metadata = result.get("chunk_metadata") or {}
        doc_metadata = result.get("doc_metadata") or {}
        section = (
            chunk_metadata.get("section")
            or doc_metadata.get("section")
            or chunk_metadata.get("heading")
            or chunk_metadata.get("section_header")
            or doc_metadata.get("heading")
            or doc_metadata.get("section_header")
        )
        if section in (None, ""):
            return ""
        return str(section).strip().lower()

    @staticmethod
    def _normalize_section_identifier(raw: str) -> str:
        text = str(raw or "").strip().lower()
        if not text:
            return ""
        match = re.search(r"(\d+\.\d+[a-z]?)", text)
        if match:
            return match.group(1)
        return text

    @classmethod
    def _extract_subsection_hints(cls, query_text: str) -> List[str]:
        hints: List[str] = []
        for match in re.findall(r"§\s*\d+(?:\.\d+)*\(([a-z0-9]+)\)", query_text, flags=re.IGNORECASE):
            token = str(match or "").strip().lower()
            if token and token not in hints:
                hints.append(token)
        for match in re.findall(r"\bsubsection\s+\(([a-z0-9]+)\)", query_text, flags=re.IGNORECASE):
            token = str(match or "").strip().lower()
            if token and token not in hints:
                hints.append(token)
        return hints

    @classmethod
    def _rerank_with_query_anchors(
        cls,
        ranked_results: List[Dict[str, Any]],
        query_text: str,
        normalized_filters: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        if not ranked_results:
            return ranked_results

        section_filter = ""
        section_values = normalized_filters.get("section", [])
        if len(section_values) == 1:
            section_filter = cls._normalize_section_identifier(section_values[0])

        subsection_hints = cls._extract_subsection_hints(query_text or "")
        if not section_filter and not subsection_hints:
            return ranked_results

        rescored: List[tuple[tuple[float, float, float, float], Dict[str, Any]]] = []
        for result in ranked_results:
            hybrid_score = float(result.get("hybrid_score", 0.0) or 0.0)
            similarity_score = float(result.get("similarity", 0.0) or 0.0)
            lexical_score = float(result.get("lexical_score", 0.0) or 0.0)
            base_score = hybrid_score if hybrid_score > 0 else similarity_score + (0.01 * lexical_score)
            boost = 0.0

            if section_filter:
                result_section = cls._normalize_section_identifier(
                    cls._extract_section_key_from_result(result)
                )
                if result_section == section_filter:
                    boost += 0.08
                else:
                    # Penalize near-matches like 1240.205 when targeting 1240.20.
                    if result_section and (
                        result_section.startswith(section_filter)
                        or section_filter.startswith(result_section)
                    ):
                        boost -= 0.03

            if subsection_hints:
                content_text = str(result.get("content") or "").lower()
                section_text = cls._extract_section_key_from_result(result).lower()
                for hint in subsection_hints:
                    if (
                        f"({hint})" in content_text
                        or f"({hint})" in section_text
                        or f"paragraph ({hint})" in content_text
                        or f"subsection ({hint})" in content_text
                    ):
                        boost += 0.06
                        break

            rescored.append(
                (
                    (
                        base_score + boost,
                        hybrid_score,
                        similarity_score,
                        lexical_score,
                    ),
                    result,
                )
            )

        rescored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in rescored]

    def _apply_required_part_coverage(
        self,
        ranked_results: List[Dict[str, Any]],
        normalized_filters: Dict[str, List[str]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        required_parts = [
            str(part).strip().lower()
            for part in normalized_filters.get("part", [])
            if str(part).strip()
        ]
        if len(required_parts) <= 1:
            return ranked_results[:top_k]

        selected: List[Dict[str, Any]] = []
        selected_ids = set()
        selected_sections_by_part: Dict[str, set] = {}
        required_per_part = 1
        if top_k >= len(required_parts) * 2:
            required_per_part = 2

        for slot in range(required_per_part):
            for required_part in required_parts:
                part_hits = 0
                for existing in selected:
                    if self._extract_part_from_result(existing) == required_part:
                        part_hits += 1
                if part_hits > slot:
                    continue

                fallback_result: Optional[Dict[str, Any]] = None
                for result in ranked_results:
                    chunk_id = result.get("chunk_id")
                    if not chunk_id or chunk_id in selected_ids:
                        continue
                    part_value = self._extract_part_from_result(result)
                    if part_value == required_part:
                        section_key = self._extract_section_key_from_result(result)
                        used_sections = selected_sections_by_part.setdefault(
                            required_part, set()
                        )
                        if slot > 0 and section_key and section_key in used_sections:
                            if fallback_result is None:
                                fallback_result = result
                            continue
                        selected.append(result)
                        selected_ids.add(chunk_id)
                        if section_key:
                            used_sections.add(section_key)
                        break
                else:
                    if fallback_result is not None:
                        chunk_id = fallback_result.get("chunk_id")
                        section_key = self._extract_section_key_from_result(
                            fallback_result
                        )
                        if chunk_id and chunk_id not in selected_ids:
                            selected.append(fallback_result)
                            selected_ids.add(chunk_id)
                            if section_key:
                                selected_sections_by_part.setdefault(
                                    required_part, set()
                                ).add(section_key)

        for result in ranked_results:
            chunk_id = result.get("chunk_id")
            if not chunk_id or chunk_id in selected_ids:
                continue
            selected.append(result)
            selected_ids.add(chunk_id)
            if len(selected) >= top_k:
                break

        return selected[:top_k]

    def similarity_search(
        self,
        query_embedding: List[float],
        classifications: List[str],
        top_k: int = 5,
        min_similarity: float = 0.7,
        source_id: Optional[str] = None,
        query_text: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        retrieval_mode: str = "dense",
    ):
        """Perform dense or hybrid similarity search."""
        with self.get_connection() as conn:
            cur = conn.cursor()

            probes_raw = os.getenv("RAG_IVFFLAT_PROBES", "10")
            try:
                probes = max(int(probes_raw), 1)
                cur.execute(f"SET LOCAL ivfflat.probes = {probes}")
            except ValueError:
                logger.warning(
                    f"Invalid RAG_IVFFLAT_PROBES='{probes_raw}', using PostgreSQL default."
                )

            mode = (retrieval_mode or "dense").strip().lower()
            if mode not in {"dense", "hybrid"}:
                mode = "dense"

            normalized_filters = self._normalize_metadata_filters(metadata_filters)
            candidate_multiplier_raw = os.getenv("RAG_HYBRID_CANDIDATE_MULTIPLIER", "4")
            max_candidates_raw = os.getenv("RAG_HYBRID_MAX_CANDIDATES", "120")
            try:
                candidate_multiplier = max(int(candidate_multiplier_raw), 1)
            except ValueError:
                candidate_multiplier = 4
            try:
                max_candidates = max(int(max_candidates_raw), top_k)
            except ValueError:
                max_candidates = max(120, top_k)

            candidate_k = min(max(top_k * candidate_multiplier, top_k), max_candidates)
            dense_results = self._dense_similarity_search(
                cur,
                query_embedding=query_embedding,
                classifications=classifications,
                top_k=candidate_k if mode == "hybrid" else top_k,
                min_similarity=min_similarity,
                source_id=source_id,
                normalized_filters=normalized_filters,
            )
            query_text_clean = (query_text or "").strip()
            if mode != "hybrid" or not query_text_clean:
                if query_text_clean:
                    dense_results = self._rerank_with_query_anchors(
                        dense_results, query_text_clean, normalized_filters
                    )
                cur.close()
                return self._apply_required_part_coverage(
                    dense_results, normalized_filters, top_k
                )

            try:
                lexical_results = self._lexical_similarity_search(
                    cur,
                    query_embedding=query_embedding,
                    query_text=query_text_clean,
                    classifications=classifications,
                    top_k=candidate_k,
                    source_id=source_id,
                    normalized_filters=normalized_filters,
                )
            except Exception as exc:
                logger.warning(
                    f"Lexical retrieval failed; falling back to dense-only search: {exc}"
                )
                dense_results = self._rerank_with_query_anchors(
                    dense_results, query_text_clean, normalized_filters
                )
                cur.close()
                return self._apply_required_part_coverage(
                    dense_results, normalized_filters, top_k
                )

            relaxed_query_text = self._build_relaxed_lexical_query(
                query_text_clean, normalized_filters
            )
            if (
                relaxed_query_text
                and relaxed_query_text != query_text_clean
                and len(lexical_results) < min(3, candidate_k)
            ):
                try:
                    relaxed_results = self._lexical_similarity_search(
                        cur,
                        query_embedding=query_embedding,
                        query_text=relaxed_query_text,
                        classifications=classifications,
                        top_k=candidate_k,
                        source_id=source_id,
                        normalized_filters=normalized_filters,
                    )
                    lexical_results = self._merge_ranked_results(
                        lexical_results, relaxed_results, candidate_k
                    )
                except Exception as exc:
                    logger.warning(
                        f"Relaxed lexical retrieval failed; using strict lexical results only: {exc}"
                    )

            cur.close()
            ranked = self._fuse_dense_and_lexical(
                dense_results, lexical_results, max(candidate_k, top_k)
            )
            ranked = self._rerank_with_query_anchors(
                ranked, query_text_clean, normalized_filters
            )
            return self._apply_required_part_coverage(
                ranked, normalized_filters, top_k
            )

    def log_query(self, user_id: str, query: str, llm_provider: str,
                  classification: str, success: bool, details: Dict = None):
        """Log RAG query for audit"""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO audit_log (user_id, action, query, llm_provider, classification, success, details)
                VALUES (%s, 'rag_query', %s, %s, %s, %s, %s)
            """, (user_id, query, llm_provider, classification, success, json.dumps(details) if details else None))
            conn.commit()
            cur.close()

    def get_pending_reviews(self, limit: int = 100):
        """Get documents pending review"""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, content, metadata, classification, created_at, pii_detected
                FROM documents
                WHERE status = 'pending_review'
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))

            docs = []
            for row in cur.fetchall():
                docs.append({
                    'id': str(row[0]),
                    'content_preview': row[1][:300] + '...' if len(row[1]) > 300 else row[1],
                    'metadata': json.loads(row[2]) if row[2] else {},
                    'classification': row[3],
                    'created_at': row[4].isoformat(),
                    'pii_detected': row[5]
                })

            cur.close()
            return docs

    def update_document_status(self, doc_id: str, status: str,
                              reviewer_id: str = None, classification: str = None):
        """Update document status after review"""
        with self.get_connection() as conn:
            cur = conn.cursor()

            if classification:
                cur.execute("""
                    UPDATE documents 
                    SET status = %s, classification = %s, reviewed_at = NOW(), reviewed_by = %s
                    WHERE id = %s
                """, (status, classification, reviewer_id, doc_id))
            else:
                cur.execute("""
                    UPDATE documents 
                    SET status = %s, reviewed_at = NOW(), reviewed_by = %s
                    WHERE id = %s
                """, (status, reviewer_id, doc_id))

            conn.commit()
            cur.close()

    def get_stats(self):
        """Get platform statistics"""
        with self.get_connection() as conn:
            cur = conn.cursor()

            # Documents by status
            cur.execute("SELECT status, COUNT(*) FROM documents GROUP BY status")
            status_counts = {row[0]: row[1] for row in cur.fetchall()}

            # Documents by classification
            cur.execute("SELECT classification, COUNT(*) FROM documents GROUP BY classification")
            classification_counts = {row[0]: row[1] for row in cur.fetchall()}

            # Total chunks
            cur.execute("SELECT COUNT(*) FROM vector_chunks")
            total_chunks = cur.fetchone()[0]

            # Query stats
            cur.execute("""
                SELECT llm_provider, classification, COUNT(*)
                FROM audit_log
                WHERE action = 'rag_query'
                GROUP BY llm_provider, classification
            """)
            query_stats = [{'llm': row[0], 'class': row[1], 'count': row[2]} for row in cur.fetchall()]

            cur.close()

            return {
                'documents_by_status': status_counts,
                'documents_by_classification': classification_counts,
                'total_vector_chunks': total_chunks,
                'query_statistics': query_stats
            }

    # ---------------------------------------------------------------------
    # Authentication helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _map_user_row(row: Any) -> Dict[str, Any]:
        return {
            "id": str(row[0]),
            "email": row[1],
            "password_hash": row[2],
            "created_at": row[3],
            "updated_at": row[4],
        }

    def create_user(self, email: str, password_hash: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO users (email, password_hash)
                    VALUES (%s, %s)
                    RETURNING id, email, password_hash, created_at, updated_at
                    """,
                    (email, password_hash),
                )
                row = cur.fetchone()
                conn.commit()
                cur.close()
                return self._map_user_row(row)
            except psycopg2.Error as exc:
                conn.rollback()
                cur.close()
                if exc.pgcode == "23505":
                    return None
                raise

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, email, password_hash, created_at, updated_at
                FROM users
                WHERE email = %s
                LIMIT 1
                """,
                (email,),
            )
            row = cur.fetchone()
            cur.close()
            return self._map_user_row(row) if row else None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, email, password_hash, created_at, updated_at
                FROM users
                WHERE id = %s
                LIMIT 1
                """,
                (user_id,),
            )
            row = cur.fetchone()
            cur.close()
            return self._map_user_row(row) if row else None

    def create_refresh_token(self, user_id: str, token_hash: str, expires_at: datetime) -> None:
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
                VALUES (%s, %s, %s)
                """,
                (user_id, token_hash, expires_at),
            )
            conn.commit()
            cur.close()

    def get_valid_refresh_token(self, token_hash: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    rt.id,
                    rt.user_id,
                    rt.token_hash,
                    rt.expires_at,
                    rt.created_at,
                    u.id,
                    u.email,
                    u.password_hash,
                    u.created_at,
                    u.updated_at
                FROM refresh_tokens rt
                JOIN users u ON u.id = rt.user_id
                WHERE rt.token_hash = %s
                  AND rt.revoked_at IS NULL
                  AND rt.expires_at > NOW()
                LIMIT 1
                """,
                (token_hash,),
            )
            row = cur.fetchone()
            cur.close()
            if not row:
                return None
            return {
                "id": str(row[0]),
                "user_id": str(row[1]),
                "token_hash": row[2],
                "expires_at": row[3],
                "created_at": row[4],
                "user": self._map_user_row((row[5], row[6], row[7], row[8], row[9])),
            }

    def rotate_refresh_token(self, old_token_hash: str, new_token_hash: str, expires_at: datetime) -> None:
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE refresh_tokens
                SET revoked_at = NOW()
                WHERE token_hash = %s
                  AND revoked_at IS NULL
                RETURNING user_id
                """,
                (old_token_hash,),
            )
            row = cur.fetchone()
            if not row:
                conn.rollback()
                cur.close()
                raise ValueError("Refresh token is invalid or already rotated.")

            user_id = row[0]
            cur.execute(
                """
                INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
                VALUES (%s, %s, %s)
                """,
                (user_id, new_token_hash, expires_at),
            )
            conn.commit()
            cur.close()

    def revoke_refresh_token(self, token_hash: str) -> None:
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE refresh_tokens
                SET revoked_at = NOW()
                WHERE token_hash = %s
                  AND revoked_at IS NULL
                """,
                (token_hash,),
            )
            conn.commit()
            cur.close()

    # ---------------------------------------------------------------------
    # Chat session helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _map_chat_session_row(row: Any) -> Dict[str, Any]:
        messages = row[6]
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except json.JSONDecodeError:
                messages = []
        if not isinstance(messages, list):
            messages = []
        return {
            "id": str(row[0]),
            "title": row[2],
            "llmProvider": row[3],
            "createdAt": int(row[4]),
            "updatedAt": int(row[5]),
            "messages": messages,
        }

    @staticmethod
    def _map_chat_session_summary_row(row: Any) -> Dict[str, Any]:
        return {
            "id": str(row[0]),
            "title": row[1],
            "llmProvider": row[2],
            "createdAt": int(row[3]),
            "updatedAt": int(row[4]),
        }

    def list_chat_sessions(
        self, user_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cur = conn.cursor()
            query = """
                SELECT id, user_id, title, llm_provider, created_at, updated_at, messages
                FROM chat_sessions
                WHERE user_id = %s
                ORDER BY updated_at DESC
                """
            params: List[Any] = [user_id]
            if limit is not None:
                query += " LIMIT %s"
                params.append(limit)
            if offset > 0:
                query += " OFFSET %s"
                params.append(offset)

            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            cur.close()
            return [self._map_chat_session_row(row) for row in rows]

    def list_chat_session_summaries(
        self, user_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cur = conn.cursor()
            query = """
                SELECT id, title, llm_provider, created_at, updated_at
                FROM chat_sessions
                WHERE user_id = %s
                ORDER BY updated_at DESC
                """
            params: List[Any] = [user_id]
            if limit is not None:
                query += " LIMIT %s"
                params.append(limit)
            if offset > 0:
                query += " OFFSET %s"
                params.append(offset)

            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            cur.close()
            return [self._map_chat_session_summary_row(row) for row in rows]

    def get_chat_session(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, user_id, title, llm_provider, created_at, updated_at, messages
                FROM chat_sessions
                WHERE user_id = %s AND id = %s
                LIMIT 1
                """,
                (user_id, session_id),
            )
            row = cur.fetchone()
            cur.close()
            return self._map_chat_session_row(row) if row else None

    def save_chat_session(self, user_id: str, session: Dict[str, Any]) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cur = conn.cursor()
            llm_provider = session["llmProvider"]
            if hasattr(llm_provider, "value"):
                llm_provider = llm_provider.value
            cur.execute(
                """
                INSERT INTO chat_sessions (
                    id, user_id, title, llm_provider, created_at, updated_at, messages
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET title = EXCLUDED.title,
                    llm_provider = EXCLUDED.llm_provider,
                    updated_at = EXCLUDED.updated_at,
                    messages = EXCLUDED.messages
                WHERE chat_sessions.user_id = EXCLUDED.user_id
                RETURNING id, user_id, title, llm_provider, created_at, updated_at, messages
                """,
                (
                    session["id"],
                    user_id,
                    session["title"],
                    llm_provider,
                    int(session["createdAt"]),
                    int(session["updatedAt"]),
                    json.dumps(session.get("messages", [])),
                ),
            )
            row = cur.fetchone()
            if not row:
                conn.rollback()
                cur.close()
                raise ValueError("Session id is already used by another user.")
            conn.commit()
            cur.close()
            return self._map_chat_session_row(row)

    def delete_chat_session(self, user_id: str, session_id: str) -> bool:
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM chat_sessions
                WHERE user_id = %s AND id = %s
                """,
                (user_id, session_id),
            )
            deleted = cur.rowcount > 0
            conn.commit()
            cur.close()
            return deleted
