"""
Bringer RAG System — Lightweight Reranker

Uses a cross-encoder model to reorder retrieved chunks based on their
actual relevance to the user's query, improving precision over standard
cosine similarity limits.
"""

import time
import torch
from typing import List, Dict, Any
from rich.console import Console
from sentence_transformers import CrossEncoder

import sys
import os
# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config
from src.modules.logging_utils import debug_print

console = Console()

# Global Singleton to avoid reloading overhead
_RERANKER_MODEL = None

def get_reranker_model() -> CrossEncoder:
    """Lazy loader for the cross-encoder model."""
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        debug_print(f"[dim]Loading reranker model '{config.RERANKER_MODEL_NAME}' onto {config.DEVICE}...[/dim]")
        t0 = time.perf_counter()
        _RERANKER_MODEL = CrossEncoder(config.RERANKER_MODEL_NAME, device=config.DEVICE)
        load_time = time.perf_counter() - t0
        debug_print(f"[green]Reranker loaded in {load_time:.2f}s.[/green]")
    return _RERANKER_MODEL


class Reranker:
    def __init__(self):
        self.model = get_reranker_model()
        self.min_score = config.RERANK_MIN_SCORE

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = config.FINAL_TOP_K,
        min_score: float | None = None,
    ) -> List[Dict[str, Any]]:
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
        top_chunks = chunks[:top_k]
        if len(top_chunks) == 0:
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
        
    console.print(f"\n[bold magenta]--- Reranker Test ---[/bold magenta]")
    
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
