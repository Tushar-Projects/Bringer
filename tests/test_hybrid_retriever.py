"""
Tests for HybridRetriever merge logic, specifically:
- BM25-only hits that clear the hybrid threshold are included.
- BM25-only hits below the hybrid threshold are excluded.
- Semantic pool size scales with the caller's semantic_top_k parameter.
"""

import unittest
from unittest.mock import Mock, patch

import config
from src.modules.hybrid_retriever import HybridRetriever


class HybridRetrieverMergeTests(unittest.TestCase):
    """Tests for the retrieve() merge step in HybridRetriever."""

    def _make_retriever(self, semantic_results, keyword_results):
        """
        Build a HybridRetriever with mocked internals.

        Args:
            semantic_results: list of dicts the semantic retriever returns.
            keyword_results: dict of {chunk_id: {...}} the keyword search returns.
        """
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.semantic_weight = config.SEMANTIC_WEIGHT  # 0.7
        retriever.keyword_weight = config.KEYWORD_WEIGHT    # 0.3
        retriever.semantic_retriever = Mock()
        retriever.semantic_retriever.retrieve.return_value = semantic_results
        retriever.keyword_search = Mock(return_value=keyword_results)
        return retriever

    # ------------------------------------------------------------------
    # (a) BM25-only hit surfaces when keyword score clears threshold
    # ------------------------------------------------------------------
    def test_bm25_only_hit_included_when_above_hybrid_threshold(self):
        """A chunk matched only by BM25 should appear in results if its
        keyword-derived final_score clears min_hybrid_score."""
        semantic_results = [
            {"content": "semantic hit", "metadata": {"chunk_id": "sem_1"}, "score": 0.8}
        ]
        keyword_results = {
            "sem_1": {"content": "semantic hit", "metadata": {"chunk_id": "sem_1"}, "keyword_score": 0.5, "raw_bm25": 3.0},
            "bm25_only": {"content": "exact part number XJ-4200", "metadata": {"chunk_id": "bm25_only"}, "keyword_score": 0.9, "raw_bm25": 8.0},
        }

        retriever = self._make_retriever(semantic_results, keyword_results)

        # keyword_weight=0.3, keyword_score=0.9 → keyword_only_final=0.27
        # Set threshold at 0.2 so this clears it.
        results = retriever.retrieve("XJ-4200", k=5, min_hybrid_score=0.2)

        chunk_ids = [r["chunk_id"] for r in results]
        self.assertIn("bm25_only", chunk_ids)
        self.assertIn("sem_1", chunk_ids)

        bm25_chunk = next(r for r in results if r["chunk_id"] == "bm25_only")
        self.assertEqual(bm25_chunk["semantic_score"], 0.0)
        self.assertGreater(bm25_chunk["final_score"], 0.0)

    # ------------------------------------------------------------------
    # (b) BM25-only hit excluded when below threshold
    # ------------------------------------------------------------------
    def test_bm25_only_hit_excluded_when_below_hybrid_threshold(self):
        """A chunk matched only by BM25 must be excluded if its
        keyword-derived final_score is below min_hybrid_score."""
        semantic_results = [
            {"content": "semantic hit", "metadata": {"chunk_id": "sem_1"}, "score": 0.8}
        ]
        keyword_results = {
            "bm25_weak": {"content": "weakly matched doc", "metadata": {"chunk_id": "bm25_weak"}, "keyword_score": 0.3, "raw_bm25": 1.5},
        }

        retriever = self._make_retriever(semantic_results, keyword_results)

        # keyword_weight=0.3, keyword_score=0.3 → keyword_only_final=0.09
        # Threshold 0.2 → should be excluded.
        results = retriever.retrieve("some query", k=5, min_hybrid_score=0.2)

        chunk_ids = [r["chunk_id"] for r in results]
        self.assertNotIn("bm25_weak", chunk_ids)
        self.assertIn("sem_1", chunk_ids)

    # ------------------------------------------------------------------
    # (c) Semantic pool size scales with semantic_top_k parameter
    # ------------------------------------------------------------------
    def test_semantic_pool_size_uses_semantic_top_k_param(self):
        """The semantic retriever should be called with the semantic_top_k
        value passed by the caller, not a hardcoded constant."""
        semantic_results = []
        keyword_results = {}

        retriever = self._make_retriever(semantic_results, keyword_results)

        # Call with semantic_top_k=10
        retriever.retrieve("test", k=5, semantic_top_k=10)

        call_kwargs = retriever.semantic_retriever.retrieve.call_args
        self.assertEqual(call_kwargs.kwargs.get("k", call_kwargs[1].get("k")), 10)

    def test_semantic_pool_defaults_to_config_when_not_specified(self):
        """When semantic_top_k is not passed, it should fall back to
        config.SEMANTIC_TOP_K."""
        semantic_results = []
        keyword_results = {}

        retriever = self._make_retriever(semantic_results, keyword_results)

        retriever.retrieve("test", k=5)

        call_kwargs = retriever.semantic_retriever.retrieve.call_args
        self.assertEqual(call_kwargs.kwargs.get("k", call_kwargs[1].get("k")), config.SEMANTIC_TOP_K)


if __name__ == "__main__":
    unittest.main()
