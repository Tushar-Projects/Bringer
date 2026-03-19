import unittest
from unittest.mock import Mock

import config
from src.modules.prompt_builder import PromptBuilder
from src.modules.rag_pipeline import RAGPipeline
from src.modules.reranker import Reranker


class PromptBuilderTests(unittest.TestCase):
    def test_system_prompt_supports_exact_and_closest_relevant_answering(self):
        builder = PromptBuilder()

        self.assertIn("You are a precise document extractor.", builder.system_prompt)
        self.assertIn("exact sentence or exact lines", builder.system_prompt)
        self.assertIn("closest relevant sentence", builder.system_prompt)
        self.assertIn("I could not find the exact answer in the documents.", builder.system_prompt)

    def test_prompt_uses_source_and_page_without_chunk_ids(self):
        builder = PromptBuilder()
        messages, _ = builder.build_prompt(
            "What is a watchdog timer?",
            [
                {
                    "content": "A watchdog timer is an electronic or software timer used to detect system failures.",
                    "metadata": {"source_file": "Module_2.pdf", "page_number": 17, "chunk_index": 3},
                }
            ],
            confidence_mode="moderate",
        )

        user_prompt = messages[1]["content"]
        self.assertIn("[Source: Module_2.pdf | page 17]", user_prompt)
        self.assertNotIn("chunk 3", user_prompt)
        self.assertIn("closest relevant sentence", user_prompt)


class RerankerTests(unittest.TestCase):
    def test_reranker_applies_min_score_threshold(self):
        reranker = Reranker.__new__(Reranker)
        reranker.model = Mock()
        reranker.model.predict.return_value = [0.85, 0.62]
        reranker.min_score = 0.7

        chunks = [
            {"content": "exact answer", "metadata": {}},
            {"content": "weakly related answer", "metadata": {}},
        ]

        result = reranker.rerank("question", chunks, top_k=3)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "exact answer")


class RagPipelineTests(unittest.TestCase):
    def test_pipeline_retries_with_relaxed_retrieval_when_strict_mode_fails(self):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline.expander = Mock()
        pipeline.expander.expand_query.return_value = ["query"]
        pipeline.retriever = Mock()
        pipeline.retriever.retrieve.side_effect = [[], [{"content": "candidate", "metadata": {"source_file": "a.txt"}, "final_score": 0.4}]]
        pipeline.reranker = Mock()
        pipeline.reranker.rerank.return_value = [{"content": "candidate", "metadata": {"source_file": "a.txt"}, "final_score": 0.4}]
        pipeline.prompt_builder = Mock()
        pipeline.prompt_builder.build_prompt.return_value = ([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], 42)
        pipeline.llm_client = Mock()
        pipeline.llm_client.stream.return_value = iter(["Answer text"])

        result = list(pipeline.run_rag("query"))

        self.assertEqual(result, ["Answer text", "\n\nSources\n", "- a.txt\n"])
        self.assertEqual(pipeline.retriever.retrieve.call_count, 2)
        self.assertEqual(pipeline.retriever.retrieve.call_args_list[0].kwargs["min_score"], config.STRICT_MIN_SIMILARITY_SCORE)
        self.assertEqual(pipeline.retriever.retrieve.call_args_list[1].kwargs["min_score"], config.RELAXED_MIN_SIMILARITY_SCORE)
        self.assertEqual(pipeline.prompt_builder.build_prompt.call_args.kwargs["confidence_mode"], "moderate")

    def test_pipeline_uses_fallback_only_when_both_stages_fail(self):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline.expander = Mock()
        pipeline.expander.expand_query.return_value = ["query"]
        pipeline.retriever = Mock()
        pipeline.retriever.retrieve.side_effect = [[], []]
        pipeline.reranker = Mock()
        pipeline.prompt_builder = Mock()
        pipeline.llm_client = Mock()

        result = list(pipeline.run_rag("query"))

        self.assertEqual(result, ["\nI could not find the exact answer in the documents."])
        pipeline.reranker.rerank.assert_not_called()

    def test_extract_sources_prefers_page_level_citations(self):
        pipeline = RAGPipeline.__new__(RAGPipeline)

        sources = pipeline._extract_sources(
            [
                {"metadata": {"source_file": "Module_2.pdf", "page_number": 17}},
                {"metadata": {"source_file": "notes.txt"}},
            ]
        )

        self.assertEqual(sources, ["Module_2.pdf (page 17)", "notes.txt"])


if __name__ == "__main__":
    unittest.main()
