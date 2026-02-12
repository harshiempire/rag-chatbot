"""
Vector database layer using PostgreSQL + pgvector.

Manages connection pooling, schema initialization, document storage,
vector similarity search, and audit logging.
"""

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg2
from psycopg2 import pool
from pgvector.psycopg2 import register_vector

logger = logging.getLogger(__name__)


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

            # Create indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_classification ON documents(classification)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_vector_chunks_doc ON vector_chunks(document_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user_id ON refresh_tokens(user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_refresh_tokens_expires_at ON refresh_tokens(expires_at)")

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

    def similarity_search(self, query_embedding: List[float],
                         classifications: List[str], top_k: int = 5, min_similarity: float = 0.7,
                         source_id: Optional[str] = None):
        """Perform vector similarity search"""
        with self.get_connection() as conn:
            cur = conn.cursor()

            probes_raw = os.getenv("RAG_IVFFLAT_PROBES", "10")
            try:
                probes = max(int(probes_raw), 1)
                cur.execute(f"SET LOCAL ivfflat.probes = {probes}")
            except ValueError:
                logger.warning(f"Invalid RAG_IVFFLAT_PROBES='{probes_raw}', using PostgreSQL default.")

            cur.execute("""
                SELECT 
                    vc.id,
                    vc.content,
                    vc.metadata,
                    d.metadata as doc_metadata,
                    d.source_id,
                    vc.classification,
                    1 - (vc.embedding <=> %s::vector) as similarity
                FROM vector_chunks vc
                JOIN documents d ON vc.document_id = d.id
                WHERE vc.classification = ANY(%s)
                  AND d.status = 'published'
                  AND (%s IS NULL OR d.source_id = %s)
                  AND 1 - (vc.embedding <=> %s::vector) >= %s
                ORDER BY vc.embedding <=> %s::vector
                LIMIT %s
            """, (
                np.array(query_embedding),
                classifications,
                source_id,
                source_id,
                np.array(query_embedding),
                min_similarity,
                np.array(query_embedding),
                top_k
            ))

            results = []
            for row in cur.fetchall():
                results.append({
                    'chunk_id': str(row[0]),
                    'content': row[1],
                    'chunk_metadata': row[2] if isinstance(row[2], dict) else (json.loads(row[2]) if row[2] else {}),
                    'doc_metadata': row[3] if isinstance(row[3], dict) else (json.loads(row[3]) if row[3] else {}),
                    'source_id': row[4],
                    'classification': row[5],
                    'similarity': float(row[6])
                })

            cur.close()
            return results

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
