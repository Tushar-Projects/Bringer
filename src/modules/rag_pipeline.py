"""
Bringer RAG System - Orchestration Pipeline

The central brain of the application. It receives a user query, triggers the
hybrid retrieval engine, formats the resulting contexts into a strict prompt,
sends it to the LLM client, and handles the token streaming and source citations.
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
from src.modules.config_manager import get_config_manager
from src.modules.hardware_detector import HardwareDetector
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

    def _retrieve_and_merge(
        self,
        queries: list[str],
        semantic_threshold: float,
        top_k: int,
    ) -> list[dict[str, Any]]:
        raw_chunks = {}
        for q in queries:
            chunks = self.retriever.retrieve(
                q,
                k=top_k,
                semantic_top_k=config.SEMANTIC_TOP_K,
                min_score=semantic_threshold,
            )
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id", chunk["metadata"].get("chunk_id", str(hash(chunk["content"]))))
                # Merge logic: if duplicate, keep the one with higher final_score
                if chunk_id not in raw_chunks or chunk["final_score"] > raw_chunks[chunk_id]["final_score"]:
                    raw_chunks[chunk_id] = chunk
        
        # Sort merged candidates by final_score descending
        merged = list(raw_chunks.values())
        merged.sort(key=lambda x: x["final_score"], reverse=True)
        return merged

    def _evaluate_confidence(self, candidates: list[dict[str, Any]], reranked: list[dict[str, Any]]) -> tuple[bool, str]:
        """
        Evaluates retrieval confidence using multiple signals.
        Returns a tuple of (is_confident, reason).
        """
        if not candidates:
            return False, "No candidates retrieved"
            
        if not reranked:
            return False, "No candidates survived reranking"
            
        best_semantic = max((c.get("semantic_score", 0.0) for c in candidates), default=0.0)
        best_rerank = max((c.get("rerank_score", 0.0) for c in reranked), default=0.0)
        
        if best_semantic < config.STRICT_MIN_SIMILARITY_SCORE:
            return False, f"Low semantic similarity ({best_semantic:.2f} < {config.STRICT_MIN_SIMILARITY_SCORE})"
            
        if best_rerank < config.STRICT_RERANK_MIN_SCORE:
            return False, f"Low reranker confidence ({best_rerank:.2f} < {config.STRICT_RERANK_MIN_SCORE})"
            
        return True, "High confidence"

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
            
            match = re.search(r"(?i)(?:final )?answer:\s*(?:\*\*)?", buffer)
            if match:
                found_answer = True
                yield buffer[match.end():]
                buffer = ""
                
        if not found_answer and buffer:
            yield buffer

    def run_rag(self, query: str) -> Generator[str, None, None]:
        """Orchestrates the entire RAG flow: retrieve -> prompt -> stream LLM."""
        if config.DEBUG_MODE:
            console.print(f"\n[bold cyan]Query:[/bold cyan] [italic]{query}[/italic]")

        # 1. Determine active profile
        config_manager = get_config_manager()
        active_mode = config_manager.get_active_mode()
        profile = HardwareDetector().select_profile() if active_mode == "auto" else active_mode
        
        if config.DEBUG_MODE:
            console.print(f"[dim]Power Profile:\n{profile}\n[/dim]")
            console.print(f"[dim]Query Expansion Enabled:\n{config.ENABLE_QUERY_EXPANSION}\n[/dim]")

        # 2. Initial Retrieval Pass
        semantic_threshold = config.RELAXED_MIN_SIMILARITY_SCORE # Use relaxed as base gatekeeper
        top_k = config.RELAXED_FINAL_TOP_K
        rerank_threshold = config.RELAXED_RERANK_MIN_SCORE
        
        candidates = self._retrieve_and_merge([query], semantic_threshold, top_k)
        retained = candidates # Semantic filter happens inside retrieve
        
        reranked = self.reranker.rerank(
            query,
            retained,
            top_k=top_k,
            min_score=rerank_threshold,
        )
        
        # 3. Evaluate Confidence
        is_confident, reason = self._evaluate_confidence(retained, reranked)
        
        expanded = False
        expansion_time = 0.0
        
        # 4. Conditional Query Expansion
        if not is_confident and config.ENABLE_QUERY_EXPANSION and profile != "low_power":
            t0 = time.perf_counter()
            expanded_queries = self.expander.expand_query(query)
            expansion_time = (time.perf_counter() - t0) * 1000
            expanded = True
            
            # Remove original query from expanded_queries since we already searched it,
            # actually expand_query returns the original query as well.
            new_queries = [q for q in expanded_queries if q != query]
            
            if new_queries:
                new_candidates = self._retrieve_and_merge(new_queries, semantic_threshold, top_k)
                
                # Merge and rescore/sort
                raw_chunks = {c.get("chunk_id", str(hash(c["content"]))): c for c in candidates}
                for chunk in new_candidates:
                    chunk_id = chunk.get("chunk_id", str(hash(chunk["content"])))
                    if chunk_id not in raw_chunks or chunk["final_score"] > raw_chunks[chunk_id]["final_score"]:
                        raw_chunks[chunk_id] = chunk
                        
                retained = list(raw_chunks.values())
                retained.sort(key=lambda x: x["final_score"], reverse=True)
                retained = retained[:top_k]
                
                reranked = self.reranker.rerank(
                    query,
                    retained,
                    top_k=top_k,
                    min_score=rerank_threshold,
                )

        final_chunks = reranked

        # 5. Debug Diagnostics
        if config.DEBUG_MODE:
            console.print(f"[dim]Semantic Search:\n{len(candidates)} candidates\n[/dim]")
            console.print(f"[dim]Semantic Filter:\n{len(retained)} retained\n[/dim]")
            console.print(f"[dim]Hybrid Ranking:\n{len(retained)} ranked\n[/dim]")
            console.print(f"[dim]Top-K:\n{min(len(retained), top_k)} selected\n[/dim]")
            console.print(f"[dim]Reranker:\n{len(reranked)} reranked\n[/dim]")
            console.print(f"[dim]Context:\n{len(final_chunks)} chunks\n[/dim]")
            
            if expanded:
                console.print("[dim]Query Expansion:\nTriggered\nReason:\n" + reason + "\n[/dim]")
                console.print(f"[dim]Expansion Time:\n{expansion_time:.1f} ms\n[/dim]")
            else:
                console.print("[dim]Query Expansion:\nSkipped\n[/dim]")
                if not is_confident:
                    console.print(f"[dim]Reason (not expanded):\n{reason}\nProfile={profile}, Enabled={config.ENABLE_QUERY_EXPANSION}[/dim]")

        if not final_chunks:
            yield "\nI could not find the answer in the provided documents."
            return

        t0 = time.perf_counter()
        messages, token_estimate = self.prompt_builder.build_prompt(
            query,
            final_chunks,
            confidence_mode="moderate" if not is_confident else "high",
        )
        t_prompt = time.perf_counter() - t0

        if config.DEBUG_MODE:
            console.print(f"[dim]Prompt tokens: ~{token_estimate} ({t_prompt*1000:.1f}ms)[/dim]")
            console.print("[dim]LLM generation started...[/dim]\n")

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
