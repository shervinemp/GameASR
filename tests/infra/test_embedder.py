"""Tests for embedding generation."""
import unittest


class TestEmbedder(unittest.TestCase):
    """Embedding generation."""

    def test_embedder_encode(self):
        from voice_control.rag.embeddings import Embedder
        emb = Embedder()
        result = emb.encode(["hello world"])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], list)
        # default dimension should be non-zero
        self.assertGreater(len(result[0]), 0)

    def test_embedder_multiple_queries(self):
        from voice_control.rag.embeddings import Embedder
        emb = Embedder()
        result = emb.encode(["hello", "world"])
        self.assertEqual(len(result), 2)
