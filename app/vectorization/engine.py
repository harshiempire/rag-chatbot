"""
Vectorization engine for document embedding.

Handles text chunking (fixed-size and structure-aware) and embedding
generation using SentenceTransformers or OpenAI models.
"""

import logging
import os
import uuid
from typing import Dict, List

import openai
from sentence_transformers import SentenceTransformer

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        RecursiveCharacterTextSplitter = None

from app.vectorization.chunker import RegulationChunker

logger = logging.getLogger(__name__)

# Global singleton cache for embedding models (avoids reloading on each request)
_embedding_model_cache = {}


def get_cached_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Get cached embedding model (singleton pattern)"""
    global _embedding_model_cache
    if model_name not in _embedding_model_cache:
        logger.info(f"Loading embedding model: {model_name} (cached for future requests)")
        _embedding_model_cache[model_name] = SentenceTransformer(model_name)
    return _embedding_model_cache[model_name]


class VectorizationEngine:
    """Vectorize documents using embeddings"""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model

        if embedding_model == "all-MiniLM-L6-v2":
            # Use cached model instead of loading fresh
            self.model = get_cached_embedding_model(embedding_model)
            self.embedding_dim = 384
        elif embedding_model.startswith("text-embedding"):
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.model = "openai"
            self.embedding_dim = 1536
        else:
            raise ValueError(f"Unsupported embedding model: {embedding_model}")

    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into chunks"""
        if RecursiveCharacterTextSplitter:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            return splitter.split_text(text)

        # Fallback splitter if langchain is unavailable
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

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        if self.model == "openai":
            response = openai.Embedding.create(
                model=self.embedding_model,
                input=texts
            )
            return [item['embedding'] for item in response['data']]
        else:
            embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
            return embeddings.tolist()

    def vectorize_document(self, content: str, metadata: Dict, chunk_config: Dict) -> List[Dict]:
        """Vectorize entire document"""
        # Chunk text
        chunks = self.chunk_text(
            content,
            chunk_config.get('chunk_size', 1000),
            chunk_config.get('chunk_overlap', 200)
        )

        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)

        # Create vector chunks
        vector_chunks = []
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            vector_chunks.append({
                'id': str(uuid.uuid4()),
                'content': chunk_text,
                'embedding': embedding,
                'metadata': {**metadata, 'chunk_size': len(chunk_text), 'total_chunks': len(chunks)},
                'chunk_index': idx
            })

        return vector_chunks

    def vectorize_regulation(self, content: str, metadata: Dict,
                            use_structure_aware: bool = True,
                            max_chunk_size: int = 1500,
                            min_chunk_size: int = 200) -> List[Dict]:
        """
        Vectorize regulatory document with structure-aware chunking.

        This is the production-grade method for regulatory/legal documents.
        Uses RegulationChunker to respect section boundaries (§, paragraphs).

        Args:
            content: The regulatory text to vectorize
            metadata: Base metadata to attach to each chunk
            use_structure_aware: If True, use RegulationChunker; else fall back to fixed
            max_chunk_size: Maximum chunk size for structure-aware chunking
            min_chunk_size: Minimum chunk size (smaller chunks get merged)

        Returns:
            List of vector chunk dictionaries ready for database storage
        """
        if use_structure_aware:
            # Use production-grade structure-aware chunking
            chunker = RegulationChunker(
                max_chunk_size=max_chunk_size,
                min_chunk_size=min_chunk_size,
                include_parent_context=True
            )
            structured_chunks = chunker.chunk_regulation(content, metadata)

            # Extract just the content for embedding
            chunk_texts = [c['content'] for c in structured_chunks]
        else:
            # Fall back to fixed-size chunking
            chunk_texts = self.chunk_text(content, max_chunk_size, 200)
            structured_chunks = [
                {'content': t, 'metadata': {**metadata}, 'chunk_type': 'fixed'}
                for t in chunk_texts
            ]

        # Generate embeddings for all chunks
        embeddings = self.generate_embeddings(chunk_texts)

        # Create vector chunks with structure metadata
        vector_chunks = []
        for idx, (chunk_data, embedding) in enumerate(zip(structured_chunks, embeddings)):
            vector_chunks.append({
                'id': str(uuid.uuid4()),
                'content': chunk_data['content'],
                'embedding': embedding,
                'metadata': {
                    **chunk_data.get('metadata', {}),
                    'chunk_size': len(chunk_data['content']),
                    'total_chunks': len(structured_chunks),
                    'chunking_method': 'structure_aware' if use_structure_aware else 'fixed'
                },
                'chunk_index': idx,
                'chunk_type': chunk_data.get('chunk_type', 'unknown')
            })

        logger.info(f"Vectorized regulation: {len(vector_chunks)} chunks "
                   f"(method: {'structure_aware' if use_structure_aware else 'fixed'})")
        return vector_chunks
