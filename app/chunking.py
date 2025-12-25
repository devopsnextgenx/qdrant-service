# ============================================
# NEW FILE: app/chunking.py
# ============================================
"""
Text chunking utilities for splitting documents into overlapping chunks
"""
from typing import List, Dict, Any
import re
import uuid


class TextChunker:
    """
    Splits text into overlapping chunks to maintain context at boundaries.
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        """
        Args:
            chunk_size: Target size in tokens (approximate using words)
            overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 0.75 words for English)"""
        words = len(text.split())
        return int(words * 1.33)  # Convert words to approximate tokens
    
    def _split_into_words(self, text: str) -> List[str]:
        """Split text into words while preserving structure"""
        return text.split()
    
    def _generate_chunk_id(self, parent_id: str, chunk_index: int) -> str:
        """
        Generate a valid Qdrant point ID (must be UUID or unsigned integer).
        
        Creates a deterministic UUID based on parent_id and chunk_index.
        """
        # Create a deterministic seed from parent_id and chunk_index
        seed_string = f"{parent_id}_chunk_{chunk_index}"
        # Generate UUID v5 (deterministic, based on namespace + name)
        namespace = uuid.NAMESPACE_DNS
        chunk_uuid = uuid.uuid5(namespace, seed_string)
        return str(chunk_uuid)
    
    def chunk_text(self, text: str, doc_id: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Returns list of chunk documents with:
        - id: unique chunk ID (valid UUID)
        - text: chunk content
        - payload: metadata including chunk_index, parent_id, etc.
        """
        if not text or not text.strip():
            return []
        
        # Convert chunk_size and overlap from tokens to words (approximate)
        words_per_chunk = int(self.chunk_size / 1.33)
        words_overlap = int(self.overlap / 1.33)
        
        words = self._split_into_words(text)
        chunks = []
        
        start = 0
        chunk_index = 0
        
        while start < len(words):
            # Get chunk words
            end = start + words_per_chunk
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            # Generate valid UUID for Qdrant
            chunk_id = self._generate_chunk_id(doc_id, chunk_index)
            
            # Create chunk document
            chunk_doc = {
                'id': chunk_id,  # Valid UUID string
                'text': chunk_text,
                'payload': {
                    'chunk_index': chunk_index,
                    'chunk_content': chunk_text,
                    'parent_id': str(doc_id),  # Store original parent ID in payload
                    'total_chunks': None,  # Will be updated later
                    'start_word': start,
                    'end_word': min(end, len(words)),
                    **(metadata or {})
                }
            }
            chunks.append(chunk_doc)
            
            # Move to next chunk with overlap
            start += (words_per_chunk - words_overlap)
            chunk_index += 1
            
            # Break if we've processed all words
            if end >= len(words):
                break
        
        # Update total_chunks in all chunks
        for chunk in chunks:
            chunk['payload']['total_chunks'] = len(chunks)
        
        return chunks
