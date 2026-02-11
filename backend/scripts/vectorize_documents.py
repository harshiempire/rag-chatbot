import argparse
import json
import os

import psycopg2
from dotenv import load_dotenv

from app.vectorization.engine import (
    DEFAULT_EMBEDDING_MODEL,
    VectorizationEngine,
    is_regulatory_document,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Vectorize documents into vector_chunks")
    parser.add_argument("--source-id", required=True, help="documents.source_id to vectorize")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--min-chunk-size", type=int, default=200)
    parser.add_argument(
        "--chunking-strategy",
        choices=["auto", "fixed", "structure_aware"],
        default="auto",
        help="Chunking mode for vectorization",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete existing chunks for selected docs before inserting new vectors",
    )
    parser.add_argument("--doc-limit", type=int, default=None, help="Limit documents to process")
    parser.add_argument("--doc-offset", type=int, default=0, help="Offset for document batching")
    return parser.parse_args()


def parse_jsonb(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, bytes, bytearray)):
        return json.loads(value)
    return {}


def should_use_structure_aware(strategy: str, source_id: str, metadata: dict) -> bool:
    if strategy == "structure_aware":
        return True
    if strategy == "fixed":
        return False
    return is_regulatory_document(source_id=source_id, metadata=metadata)


def main():
    args = parse_args()

    load_dotenv(dotenv_path=".env")
    conn_str = os.getenv("DATABASE_URL")
    if not conn_str:
        raise SystemExit("DATABASE_URL not set")

    vectorization = VectorizationEngine(DEFAULT_EMBEDDING_MODEL)

    inserted = 0
    deleted = 0
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cur:
            if args.doc_limit is None:
                cur.execute(
                    """
                    SELECT id, source_id, content, metadata, classification
                    FROM documents
                    WHERE source_id = %s
                    ORDER BY id
                    """,
                    (args.source_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, source_id, content, metadata, classification
                    FROM documents
                    WHERE source_id = %s
                    ORDER BY id
                    LIMIT %s OFFSET %s
                    """,
                    (args.source_id, args.doc_limit, args.doc_offset),
                )
            rows = cur.fetchall()
            print("Documents fetched:", len(rows))

            for doc_id, source_id, content, metadata_raw, classification in rows:
                metadata = parse_jsonb(metadata_raw)
                use_structure_aware = should_use_structure_aware(
                    args.chunking_strategy, source_id, metadata
                )

                if args.replace_existing:
                    cur.execute("DELETE FROM vector_chunks WHERE document_id = %s", (doc_id,))
                    deleted += cur.rowcount

                if use_structure_aware:
                    vector_chunks = vectorization.vectorize_regulation(
                        content,
                        metadata,
                        use_structure_aware=True,
                        max_chunk_size=args.chunk_size,
                        min_chunk_size=args.min_chunk_size,
                    )
                else:
                    vector_chunks = vectorization.vectorize_document(
                        content,
                        metadata,
                        {"chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap},
                    )

                for chunk in vector_chunks:
                    cur.execute(
                        """
                        INSERT INTO vector_chunks (id, document_id, content, embedding, metadata, chunk_index, classification)
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
                        """,
                        (
                            chunk["id"],
                            doc_id,
                            chunk["content"],
                            [float(x) for x in chunk["embedding"]],
                            json.dumps(chunk["metadata"]),
                            chunk["chunk_index"],
                            classification,
                        ),
                    )
                    inserted += 1

        conn.commit()

    print("Inserted vector chunks:", inserted)
    if args.replace_existing:
        print("Deleted old chunks:", deleted)


if __name__ == "__main__":
    main()
