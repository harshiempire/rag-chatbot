"""
Structure-aware chunking for regulatory documents.

Instead of splitting on arbitrary character counts, RegulationChunker
respects the natural structure of regulations: sections (§),
paragraphs (a), (b), (1), (2), etc.
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


class RegulationChunker:
    """
    Structure-aware chunking for regulatory documents (eCFR, CFR, legal texts).

    Benefits:
    - Preserves semantic boundaries
    - Each chunk is a complete regulatory unit
    - Better retrieval quality for legal/compliance QA
    """

    def __init__(self,
                 max_chunk_size: int = 1500,
                 min_chunk_size: int = 200,
                 include_parent_context: bool = True):
        """
        Args:
            max_chunk_size: Maximum characters per chunk (will split if exceeded)
            min_chunk_size: Minimum characters (will merge small sections)
            include_parent_context: Add section header to child chunks
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.include_parent_context = include_parent_context

        # Regex patterns for regulatory structure
        self.section_pattern = re.compile(
            r'(§\s*[\d.]+[a-z]?\s*[^\n]*)',  # Section markers: § 1277.3
            re.MULTILINE
        )
        self.paragraph_pattern = re.compile(
            r'^\s*(\([a-z0-9]+\))',  # Paragraph markers: (a), (1), (i)
            re.MULTILINE
        )
        self.subpart_pattern = re.compile(
            r'^(Subpart\s+[A-Z].*?)(?=\n)',  # Subpart headers
            re.MULTILINE | re.IGNORECASE
        )

    def chunk_regulation(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split regulatory text into structure-aware chunks.

        Returns:
            List of dicts with 'content', 'metadata', 'chunk_type'
        """
        metadata = metadata or {}
        chunks = []

        # First, split by major sections (§)
        sections = self._split_by_sections(text)

        for section in sections:
            section_header = section.get('header', '')
            section_content = section.get('content', '')

            # If section is small enough, keep as single chunk
            if len(section_content) <= self.max_chunk_size:
                if len(section_content) >= self.min_chunk_size:
                    chunks.append({
                        'content': section_content,
                        'metadata': {
                            **metadata,
                            'section_header': section_header,
                            'chunk_type': 'section',
                            'structure_level': 'section'
                        },
                        'chunk_type': 'section'
                    })
            else:
                # Split large sections by paragraphs
                paragraphs = self._split_by_paragraphs(section_content, section_header)
                chunks.extend(paragraphs)

        # Merge tiny chunks with neighbors
        chunks = self._merge_small_chunks(chunks)

        # Add chunk indices
        for idx, chunk in enumerate(chunks):
            chunk['metadata']['chunk_index'] = idx
            chunk['metadata']['total_chunks'] = len(chunks)

        logger.info(f"Created {len(chunks)} structure-aware chunks from regulation text")
        return chunks

    def _split_by_sections(self, text: str) -> List[Dict]:
        """Split text by § section markers"""
        sections = []

        # Find all section markers
        matches = list(self.section_pattern.finditer(text))

        if not matches:
            # No section markers found, treat entire text as one section
            return [{'header': '', 'content': text.strip()}]

        # Extract sections between markers
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            section_text = text[start:end].strip()
            header = match.group(1).strip()

            sections.append({
                'header': header,
                'content': section_text
            })

        # Include any text before first section
        if matches and matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if len(preamble) >= self.min_chunk_size:
                sections.insert(0, {'header': 'Preamble', 'content': preamble})

        return sections

    def _split_by_paragraphs(self, content: str, parent_header: str) -> List[Dict]:
        """Split section content by paragraph markers (a), (b), (1), (2)"""
        chunks = []

        # Find paragraph markers
        matches = list(self.paragraph_pattern.finditer(content))

        if not matches:
            # No paragraphs, split by sentence if still too large
            return self._split_by_sentences(content, parent_header)

        current_chunk = ""
        current_para = ""

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

            para_text = content[start:end].strip()
            para_marker = match.group(1)

            # Check if adding this paragraph exceeds max size
            if len(current_chunk) + len(para_text) > self.max_chunk_size:
                # Save current chunk and start new one
                if current_chunk:
                    chunk_content = current_chunk
                    if self.include_parent_context and parent_header:
                        chunk_content = f"{parent_header}\n\n{current_chunk}"

                    chunks.append({
                        'content': chunk_content.strip(),
                        'metadata': {
                            'section_header': parent_header,
                            'paragraph': current_para,
                            'chunk_type': 'paragraph_group',
                            'structure_level': 'paragraph'
                        },
                        'chunk_type': 'paragraph_group'
                    })

                current_chunk = para_text
                current_para = para_marker
            else:
                current_chunk += "\n\n" + para_text if current_chunk else para_text
                if not current_para:
                    current_para = para_marker

        # Don't forget the last chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk_content = current_chunk
            if self.include_parent_context and parent_header:
                chunk_content = f"{parent_header}\n\n{current_chunk}"

            chunks.append({
                'content': chunk_content.strip(),
                'metadata': {
                    'section_header': parent_header,
                    'paragraph': current_para,
                    'chunk_type': 'paragraph_group',
                    'structure_level': 'paragraph'
                },
                'chunk_type': 'paragraph_group'
            })

        return chunks

    def _split_by_sentences(self, content: str, parent_header: str) -> List[Dict]:
        """Fallback: split by sentences when no structure found"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', content)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.max_chunk_size:
                if current_chunk:
                    chunk_content = current_chunk
                    if self.include_parent_context and parent_header:
                        chunk_content = f"{parent_header}\n\n{current_chunk}"

                    chunks.append({
                        'content': chunk_content.strip(),
                        'metadata': {
                            'section_header': parent_header,
                            'chunk_type': 'sentence_group',
                            'structure_level': 'sentence'
                        },
                        'chunk_type': 'sentence_group'
                    })
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk_content = current_chunk
            if self.include_parent_context and parent_header:
                chunk_content = f"{parent_header}\n\n{current_chunk}"

            chunks.append({
                'content': chunk_content.strip(),
                'metadata': {
                    'section_header': parent_header,
                    'chunk_type': 'sentence_group',
                    'structure_level': 'sentence'
                },
                'chunk_type': 'sentence_group'
            })

        return chunks

    def _merge_small_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Merge chunks that are too small with their neighbors"""
        if len(chunks) <= 1:
            return chunks

        merged = []
        i = 0

        while i < len(chunks):
            chunk = chunks[i]

            # If chunk is too small and there's a next chunk, merge
            if len(chunk['content']) < self.min_chunk_size and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                merged_content = chunk['content'] + "\n\n" + next_chunk['content']

                merged.append({
                    'content': merged_content,
                    'metadata': {
                        **chunk['metadata'],
                        'merged': True,
                        'chunk_type': 'merged'
                    },
                    'chunk_type': 'merged'
                })
                i += 2  # Skip next chunk since we merged it
            else:
                merged.append(chunk)
                i += 1

        return merged
