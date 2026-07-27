"""
Bringer RAG System - Orchestration Pipeline

The central brain of the application. It receives a user query, triggers the
hybrid retrieval engine, formats the resulting contexts into a strict prompt,
sends it to the LM Studio LLM client, and handles the token streaming and source citations.
"""

import os
import re
import sys
import time
from collections.abc import Generator
from typing import Any

# pyrefly: ignore [missing-import]
from rich.console import Console

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config
from src.modules.hybrid_retriever import HybridRetriever
from src.modules.llm_client import LLMClient
from src.modules.logging_utils import debug_print
from src.modules.prompt_builder import PromptBuilder
from src.modules.query_expander import QueryExpander
from src.modules.reranker import Reranker

console = Console()


class RAGPipeline:
    def __init__(self):
        """Initializes all downstream engines required for full text generation."""
        self.retriever = HybridRetriever()
        self.prompt_builder = PromptBuilder()
        self.llm_client = LLMClient()
        self.expander = QueryExpander()
        self.reranker = Reranker()

    def _retrieve_candidates(
        self,
        expanded_queries: list[str],
        similarity_threshold: float,
        top_k: int,
    ) -> list[dict[str, Any]]:
        raw_chunks = {}
        for q in expanded_queries:
            chunks = self.retriever.retrieve(
                q,
                k=top_k,
                semantic_top_k=top_k,
                min_score=similarity_threshold,
            )
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id", chunk["metadata"].get("chunk_id", str(hash(chunk["content"]))))
                if chunk_id not in raw_chunks:
                    raw_chunks[chunk_id] = chunk
        return list(raw_chunks.values())

    def _filter_context_chunks(self, chunks: list[dict[str, Any]], min_similarity: float) -> list[dict[str, Any]]:
        """Keeps only strongly related chunks before prompt construction."""
        filtered = []
        for chunk in chunks:
            score = chunk.get("final_score", chunk.get("score", 0.0))
            if score >= min_similarity:
                filtered.append(chunk)
        return filtered

    def _extract_sources(self, chunks: list[dict[str, Any]]) -> list[str]:
        """Deduplicates and formats source metadata into clean citation strings."""
        sources = set()
        for chunk in chunks:
            source_file = chunk["metadata"].get("source_file", "Unknown")
            page_number = chunk["metadata"].get("page_number")

            if page_number is not None:
                sources.add(f"{source_file} (page {page_number})")
            else:
                sources.add(source_file)

        return sorted(sources)

    def _filter_reasoning_stream(self, token_stream: Generator[str, None, None]) -> Generator[str, None, None]:
        buffer = ""
        found_answer = False
        
        for token in token_stream:
            if found_answer:
                yield token
                continue
                
            buffer += token
            
            # The model is instructed to always output "Answer:" before its final answer.
            # We suppress everything before that marker, which naturally hides all 
            # innate reasoning steps like "Draft:", "<think>", etc.
            match = re.search(r"(?i)(?:final )?answer:\s*(?:\*\*)?", buffer)
            if match:
                found_answer = True
                yield buffer[match.end():]
                buffer = ""
                
        # Fallback: if the model completely ignored the instruction and never output "Answer:",
        # we yield whatever was buffered so the user doesn't get a blank response.
        if not found_answer and buffer:
            yield buffer

    def run_rag(self, query: str) -> Generator[str, None, None]:
        """Orchestrates the entire RAG flow: retrieve -> prompt -> stream LLM."""
        debug_print(f"\n[bold cyan]Query:[/bold cyan] [italic]{query}[/italic]")

        t0 = time.perf_counter()
        expanded_queries = self.expander.expand_query(query)
        time.perf_counter() - t0

        retrieval_modes = [
            ("high", config.STRICT_MIN_SIMILARITY_SCORE, config.STRICT_RERANK_MIN_SCORE, config.STRICT_FINAL_TOP_K),
            ("moderate", config.RELAXED_MIN_SIMILARITY_SCORE, config.RELAXED_RERANK_MIN_SCORE, config.RELAXED_FINAL_TOP_K),
        ]

        final_chunks = []
        confidence_mode = "high"
        t_retrieval = 0.0

        for idx, (mode, similarity_threshold, rerank_threshold, top_k) in enumerate(retrieval_modes):
            if idx == 1:
                debug_print("Strict mode failed, using relaxed retrieval...")

            t0 = time.perf_counter()
            unique_chunks = self._retrieve_candidates(expanded_queries, similarity_threshold, top_k)
            t_retrieval = time.perf_counter() - t0
            unique_chunks = self._filter_context_chunks(unique_chunks, similarity_threshold)

            if not unique_chunks:
                continue

            debug_print(
                f"[dim]Hybrid retrieval returned {len(unique_chunks)} unique chunks ({t_retrieval*1000:.1f}ms)[/dim]"
            )

            t0 = time.perf_counter()
            final_chunks = self.reranker.rerank(
                query,
                unique_chunks,
                top_k=top_k,
                min_score=rerank_threshold,
            )
            time.perf_counter() - t0
            final_chunks = self._filter_context_chunks(final_chunks, similarity_threshold)

            if final_chunks:
                confidence_mode = mode
                break

        if not final_chunks:
            yield "\nI could not find the answer in the provided documents."
            return

        t0 = time.perf_counter()
        messages, token_estimate = self.prompt_builder.build_prompt(
            query,
            final_chunks,
            confidence_mode=confidence_mode,
        )
        t_prompt = time.perf_counter() - t0

        debug_print(f"[dim]Prompt tokens: ~{token_estimate} ({t_prompt*1000:.1f}ms)[/dim]")
        debug_print("[dim]LLM generation started...[/dim]\n")

        try:
            generator = self.llm_client.stream(messages)
            if config.DEBUG_MODE:
                for token in generator:
                    yield token
            else:
                for token in self._filter_reasoning_stream(generator):
                    yield token
        except Exception as e:  # noqa: BLE001
            yield f"\n\n[Error during LLM generation: {e!s}]"
            return

        sources = self._extract_sources(final_chunks)
        yield "\n\nSources\n"
        for source in sources:
            yield f"- {source}\n"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    else:
        test_query = "What is a wired LAN?"

    console.print("\n[bold magenta]--- RAG Pipeline Test ---[/bold magenta]")

    pipeline = RAGPipeline()

    print("\nAnswer: ", end="", flush=True)
    for chunk in pipeline.run_rag(test_query):
        print(chunk, end="", flush=True)
    print("\n\n[bold green]Test Complete![/bold green]")
