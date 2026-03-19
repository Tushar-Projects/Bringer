"""
Bringer RAG System — Semantic Retriever

Handles searching the persistent vector database for semantic matches
to a user's query.

Uses the global embedding model from Phase 4 to avoid reloading overhead.
Converts ChromaDB distances (cosine/L2) into an intuitive 0-1 similarity score
and discards chunks below a configurable threshold.
"""

import time
from typing import List, Dict, Any
from rich.console import Console

import sys
import os
# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import config
from src.modules.embedding_engine import get_embedding_model
from src.modules.vector_store import VectorStore
from src.modules.logging_utils import debug_print

console = Console()

class Retriever:
    def __init__(self):
        """Initializes the retriever, connecting to the vector store and embedding model."""
        self.vector_store = VectorStore()
        # Grab the updated EmbeddingEngine to use its LRU cached embed_query method
        from src.modules.embedding_engine import EmbeddingEngine
        self.embedding_engine = EmbeddingEngine()
        self.min_score = config.MIN_SIMILARITY_SCORE

    def retrieve(self, query: str, k: int = config.SEMANTIC_TOP_K, min_score: float | None = None) -> List[Dict[str, Any]]:
        """
        Embeds a user query, searches ChromaDB, and returns scored and filtered chunks.
        
        Args:
            query: The user's text query.
            k: The maximum number of chunks to return.
            
        Returns:
            A list of dictionary objects containing 'content', 'metadata', and 'score'.
        """
        if not query or not query.strip():
            debug_print("[yellow]Empty query provided to retriever.[/yellow]")
            return []

        threshold = self.min_score if min_score is None else min_score

        debug_print(f"\n[bold cyan]Query:[/bold cyan] '{query}'")
        
        # 1. Embed the query (utilizes LRU cache to save time on identical sub-queries)
        t0 = time.perf_counter()
        query_embedding = self.embedding_engine.embed_query(query)
        t_embed = time.perf_counter() - t0
        
        # 2. Search Vector DB
        t0 = time.perf_counter()
        raw_results = self.vector_store.semantic_search(query_embedding, n_results=k)
        t_search = time.perf_counter() - t0
        
        # Handle empty DB case natively
        if not raw_results["ids"] or not raw_results["ids"][0]:
            return []
            
        # 3. Process, score, and filter results
        t0 = time.perf_counter()
        formatted_results = []
        
        # Chroma returns lists of lists because it supports batch queries;
        # Since we passed a single query, we access the [0] index.
        distances = raw_results["distances"][0]
        documents = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]
        
        for dist, doc, meta in zip(distances, documents, metadatas):
            # Convert distance to similarity score
            # By default ChromaDB uses squared L2 distance.
            # Since our embeddings are normalized (length 1):
            # Squared L2 ranges from 0 (identical) to 4 (perfect opposites).
            # We map this to a 0.0 -> 1.0 similarity score.
            # A completely orthogonal vector has distance 2.0 (score 0.5).
            # A perfect match has distance 0.0 (score 1.0).
            score = 1.0 - (dist / 4.0)
            
            # Floor out tiny negative rounding errors
            score = max(0.0, min(1.0, score))
            
            # 4. Apply threshold filter
            if score >= threshold:
                formatted_results.append({
                    "content": doc,
                    "metadata": meta,
                    "score": round(score, 4)
                })
                
        # Sort descending by score just to be safe
        formatted_results.sort(key=lambda x: x["score"], reverse=True)
        t_process = time.perf_counter() - t0
        
        # 5. Logging
        if formatted_results:
            top_score = formatted_results[0]["score"]
            avg_score = sum(r["score"] for r in formatted_results) / len(formatted_results)
            debug_print(f"[green]Retrieved {len(formatted_results)} chunks[/green] "
                        f"(Discarded {len(distances) - len(formatted_results)} below threshold {threshold})")
            debug_print(f"Top similarity score: [bold green]{top_score:.4f}[/bold green]")
            debug_print(f"Average score: [cyan]{avg_score:.4f}[/cyan]")
        else:
            debug_print(f"[yellow]No chunks met the minimum similarity threshold ({threshold}).[/yellow]")
            
        debug_print(f"[dim]Timing: Embed {t_embed*1000:.1f}ms | Search {t_search*1000:.1f}ms | Process {t_process*1000:.1f}ms[/dim]")
        
        return formatted_results


# Quick test trigger block (only runs if executed directly)
if __name__ == "__main__":
    import json
    
    if len(sys.argv) > 1:
        query_str = " ".join(sys.argv[1:])
        
        console.print(f"\n[bold magenta]--- Semantic Retrieval Test ---[/bold magenta]")
        
        retriever = Retriever()
        results = retriever.retrieve(query_str, k=5)
        
        if results:
            console.print("\n[bold green]Top Match Preview:[/bold green]")
            best = results[0]
            print(json.dumps({
                "score": best["score"],
                "source": best["metadata"]["source_file"],
                "chunk_id": best["metadata"].get("chunk_id", "N/A"),
                "content_preview": best["content"][:150] + "..."
            }, indent=2, ensure_ascii=False))
        else:
            console.print("[red]No results retrieved.[/red]")
    else:
        print("Usage: python src/modules/retriever.py \"your search query here\"")
