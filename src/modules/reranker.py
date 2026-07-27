"""
Bringer RAG System — Lightweight Reranker

Uses a cross-encoder model to reorder retrieved chunks based on their
actual relevance to the user's query, improving precision over standard
cosine similarity limits.
"""

import os
import sys
import time
from typing import Any

import torch
from rich.console import Console
from sentence_transformers import CrossEncoder

# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config
from src.modules.config_manager import get_config_manager
from src.modules.hardware_detector import HardwareDetector
from src.modules.logging_utils import debug_print

console = Console()

# Global Singleton to avoid reloading overhead
_RERANKER_MODEL = None
_CURRENT_RERANKER_NAME = None

def get_reranker_model() -> CrossEncoder:
    """Lazy loader for the cross-encoder model."""
    global _RERANKER_MODEL, _CURRENT_RERANKER_NAME
    
    config_manager = get_config_manager()
    active_mode = config_manager.get_active_mode()
    
    if active_mode == "auto":
        profile_name = HardwareDetector().select_profile()
    else:
        profile_name = active_mode
        
    profile_config = config_manager.get_profile_config(profile_name)
    model_name = profile_config.get("reranker", {}).get("name", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    if _RERANKER_MODEL is None or _CURRENT_RERANKER_NAME != model_name:
        debug_print(f"[dim]Loading reranker model '{model_name}' onto {config.DEVICE}...[/dim]")
        t0 = time.perf_counter()
        _RERANKER_MODEL = CrossEncoder(model_name, device=config.DEVICE)
        _CURRENT_RERANKER_NAME = model_name
        load_time = time.perf_counter() - t0
        debug_print(f"[green]Reranker loaded in {load_time:.2f}s.[/green]")
    return _RERANKER_MODEL


class Reranker:
    def __init__(self):
        self.model = get_reranker_model()
        
        config_manager = get_config_manager()
        active_mode = config_manager.get_active_mode()
        if active_mode == "auto":
            profile_name = HardwareDetector().select_profile()
        else:
            profile_name = active_mode
            
        profile_config = config_manager.get_profile_config(profile_name)
        rerank_config = profile_config.get("reranker", {})
        self.min_score = rerank_config.get("min_score", 0.4)

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = config.FINAL_TOP_K,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Takes a list of candidate chunks and scores them against the query.
        
        Args:
            query: The user's text query.
            chunks: A list of dictionary objects containing 'content' and 'metadata'.
            top_k: The number of best chunks to return.
            
        Returns:
            The top_k sorted chunks with updated 'rerank_score'.
        """
        if not chunks:
            return []

        if len(chunks) == 1:
            chunks[0]["rerank_score"] = 1.0 # arbitrary default for single chunks
            return chunks

        debug_print(f"[dim]Reranking {len(chunks)} chunks...[/dim]")
        t0 = time.perf_counter()
        
        # Build pairs of (query, document) for the cross-encoder
        pairs = [[query, chunk["content"]] for chunk in chunks]
        
        # Generates logits indicating relevance
        scores = self.model.predict(pairs)
        
        # Attach scores and sort
        for chunk, score in zip(chunks, scores):
            # Normalizing/casting score strictly for ordering
            chunk["rerank_score"] = float(score)
            
        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Apply min_score threshold
        threshold = min_score if min_score is not None else getattr(self, "min_score", 0.0)
        chunks = [c for c in chunks if c["rerank_score"] >= threshold]
        
        top_chunks = chunks[:top_k]
        
        t_rank = time.perf_counter() - t0
        debug_print(f"[dim]Reranker reduced {len(chunks)} down to Top {len(top_chunks)} ({t_rank*1000:.1f}ms).[/dim]")
        
        return top_chunks


# Quick test trigger block (only runs if executed directly)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    else:
        test_query = "What is a wired LAN?"
        
    console.print("\n[bold magenta]--- Reranker Test ---[/bold magenta]")
    
    mock_chunks = [
        {"content": "Wi-Fi is a popular wireless networking technology that uses radio waves.", "metadata": {}},
        {"content": "A wired local area network (LAN) usually employs Ethernet cables to connect switches and computers.", "metadata": {}},
        {"content": "Bluetooth is primarily used for short-range personal area networks.", "metadata": {}},
        {"content": "In a wired architecture, data packets are routed physically via copper or fiber links.", "metadata": {}}
    ]
    
    reranker = Reranker()
    ranked = reranker.rerank(test_query, mock_chunks, top_k=2)
    
    console.print("\n[bold green]Top 2 Reranked Results:[/bold green]")
    for i, res in enumerate(ranked):
        console.print(f"\n{i+1}. Score: {res['rerank_score']:.3f}\n   {res['content']}")
