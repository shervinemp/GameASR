"""Tests for data loading utilities."""
import unittest


class TestDataLoader(unittest.TestCase):
    """Data loading utilities."""

    def test_data_loader_imports(self):
        from voice_control.rag.data import DataLoader, CodexDataLoader
        self.assertTrue(callable(DataLoader))
        self.assertTrue(callable(CodexDataLoader))
