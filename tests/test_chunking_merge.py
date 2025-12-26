import os
import sys
import unittest

# Ensure project root is on sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.chunking import TextChunker

class TestChunkingMerge(unittest.TestCase):
    def test_no_merge_needed(self):
        # chunk_size = 30 tokens ~= 22 words. 0.5 * 22 = 11 words.
        # Text with 44 words should split into exactly 2 chunks of 22 words each (approx).
        chunker = TextChunker(chunk_size=40, overlap=0)
        # 40 tokens / 1.33 = 30 words per chunk.
        # Text with 60 words:
        text = "word " * 60
        chunks = chunker.chunk_text(text, doc_id="doc1")
        # Should have exactly 2 chunks of 30 words.
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]['text'].split()), 30)
        self.assertEqual(len(chunks[1]['text'].split()), 30)

    def test_merge_occurred(self):
        # 40 tokens / 1.33 = 30 words per chunk. 0.5 * 30 = 15 words.
        chunker = TextChunker(chunk_size=40, overlap=0)
        # Text with 40 words:
        # C0: 0-30
        # C1: 30-40 (10 words)
        # 10 < 15, so C1 should be merged into C0.
        text = "word " * 40
        chunks = chunker.chunk_text(text, doc_id="doc1")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]['text'].split()), 40)

    def test_no_merge_if_greater_than_50_percent(self):
        # 40 tokens / 1.33 = 30 words per chunk. 0.5 * 30 = 15 words.
        chunker = TextChunker(chunk_size=40, overlap=0)
        # Text with 50 words:
        # C0: 0-30
        # C1: 30-50 (20 words)
        # 20 > 15, so no merge.
        text = "word " * 50
        chunks = chunker.chunk_text(text, doc_id="doc1")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]['text'].split()), 30)
        self.assertEqual(len(chunks[1]['text'].split()), 20)

    def test_overlap_merge(self):
        # 40 tokens / 1.33 = 30 words per chunk. overlap 13 tokens ~= 10 words.
        # step = 30 - 10 = 20 words.
        chunker = TextChunker(chunk_size=40, overlap=13)
        # Text with 57 words:
        # C0: 0-30
        # step to 21
        # C1: 21-51
        # step to 42
        # C2: 42-57 (15 words)
        # 0.5 * 30 = 15. 15 is NOT LESS THAN 15. So no merge.
        text = "word " * 57
        chunks = chunker.chunk_text(text, doc_id="doc1")
        self.assertEqual(len(chunks), 3)

        # Text with 54 words:
        # C2: 40-54 (14 words)
        # 14 < 15. Merge C2 into C1.
        text = "word " * 54
        chunks = chunker.chunk_text(text, doc_id="doc1")
        # In current code, C1 would be 20-50.
        # After merge, C1 should be 20-54.
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[1]['text'].split()), 33)

if __name__ == '__main__':
    unittest.main()
