import argparse
import json
import os
import uuid
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        RecursiveCharacterTextSplitter = None


def main():
    parser = argparse.ArgumentParser(description='Vectorize documents into vector_chunks')
    parser.add_argument('--source-id', required=True, help='documents.source_id to vectorize')
    parser.add_argument('--chunk-size', type=int, default=1000)
    parser.add_argument('--chunk-overlap', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--doc-limit', type=int, default=None, help='Limit number of documents to process')
    parser.add_argument('--doc-offset', type=int, default=0, help='Offset for document batching')
    args = parser.parse_args()

    load_dotenv(dotenv_path='.env')
    conn_str = os.getenv('DATABASE_URL')
    if not conn_str:
        raise SystemExit('DATABASE_URL not set')

    # This script uses the local sentence-transformers model with 384-dim embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    def simple_split(text, chunk_size, chunk_overlap):
        if chunk_size <= 0:
            return [text]
        chunks = []
        start = 0
        step = max(chunk_size - chunk_overlap, 1)
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += step
        return chunks

    if RecursiveCharacterTextSplitter:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    else:
        splitter = None

    inserted = 0
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cur:
            # Fetch documents to vectorize
            if args.doc_limit is None:
                cur.execute('''
                    SELECT id, content, metadata, classification
                    FROM documents
                    WHERE source_id = %s
                    ORDER BY id
                ''', (args.source_id,))
            else:
                cur.execute('''
                    SELECT id, content, metadata, classification
                    FROM documents
                    WHERE source_id = %s
                    ORDER BY id
                    LIMIT %s OFFSET %s
                ''', (args.source_id, args.doc_limit, args.doc_offset))
            rows = cur.fetchall()
            print('Documents fetched:', len(rows))

            for doc_id, content, metadata, classification in rows:
                if splitter:
                    chunks = splitter.split_text(content)
                else:
                    chunks = simple_split(content, args.chunk_size, args.chunk_overlap)
                if not chunks:
                    continue

                embeddings = model.encode(chunks, batch_size=args.batch_size, show_progress_bar=False)

                for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_id = str(uuid.uuid4())
                    if metadata is None:
                        chunk_meta = {}
                    elif isinstance(metadata, (str, bytes, bytearray)):
                        chunk_meta = json.loads(metadata)
                    else:
                        chunk_meta = metadata
                    chunk_meta.update({
                        'chunk_size': len(chunk_text),
                        'total_chunks': len(chunks)
                    })

                    embedding_list = [float(x) for x in embedding]
                    cur.execute('''
                        INSERT INTO vector_chunks (id, document_id, content, embedding, metadata, chunk_index, classification)
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
                    ''', (
                        chunk_id,
                        doc_id,
                        chunk_text,
                        embedding_list,
                        json.dumps(chunk_meta),
                        idx,
                        classification
                    ))
                    inserted += 1

        conn.commit()

    print('Inserted vector chunks:', inserted)


if __name__ == '__main__':
    main()
