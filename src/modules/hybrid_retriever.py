"""
Bringer RAG System — Hybrid Retriever

Improves retrieval quality by fusing Semantic Vector Search 
with BM25 Lexical Keyword Search.

- Uses the same global models to avoid memory bloat.
- BM25 index is cached locally and rebuilt only when DB length changes.
"""

import time
from typing import List, Dict, Any, Optional
from rich.console import Console
from rank_bm25 import BM25Okapi

import sys
import os
# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import config
from src.modules.retriever import Retriever
from src.modules.vector_store import VectorStore

console = Console()

class HybridRetriever:
    def __init__(self):
        self.vector_store = VectorStore()
        # Underlying semantic retriever
        self.semantic_retriever = Retriever()
        self.semantic_weight = config.SEMANTIC_WEIGHT
        self.keyword_weight = config.KEYWORD_WEIGHT
        
        # BM25 State
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_doc_ids: List[str] = []
        self._bm25_docs: List[str] = []
        self._bm25_metadatas: List[Dict[str, Any]] = []
        self._last_doc_count = -1
        
        # Preload BM25
        self._load_bm25_index_if_needed()

    def _load_bm25_index_if_needed(self):
        """Loads or rebuilds the BM25 lexical index only if the vector database has changed."""
        current_count = self.vector_store.collection.count()
        if current_count == self._last_doc_count and self._bm25_index is not None:
            return  # Cache hit
            
        console.print(f"[dim]Rebuilding BM25 lexical index for {current_count} documents...[/dim]")
        t0 = time.perf_counter()
        
        if current_count == 0:
            self._bm25_index = None
            self._bm25_doc_ids = []
            self._bm25_docs = []
            self._bm25_metadatas = []
            self._last_doc_count = 0
            return
            
        # Fetch all documents to build the term frequency index
        all_data = self.vector_store.collection.get(include=["documents", "metadatas"])
        
        self._bm25_doc_ids = all_data["ids"]
        self._bm25_docs = all_data["documents"]
        self._bm25_metadatas = all_data["metadatas"]
        
        # Tokenize documents for BM25 (simple split on spaces/punctuation)
        tokenized_corpus = [doc.lower().split() for doc in self._bm25_docs]
        self._bm25_index = BM25Okapi(tokenized_corpus)
        
        self._last_doc_count = current_count
        t_build = time.perf_counter() - t0
        console.print(f"[green]Built BM25 index in {t_build*1000:.1f}ms.[/green]")

    def keyword_search(self, query: str, k: int = config.BM25_TOP_K) -> Dict[str, Any]:
        """
        Executes a BM25 lexical search and returns normalized scores.
        """
        self._load_bm25_index_if_needed()
        
        if self._bm25_index is None:
            return {}
            
        tokenized_query = query.lower().split()
        
        # Get raw BM25 scores for all documents in the corpus
        raw_scores = self._bm25_index.get_scores(tokenized_query)
        
        # Create a list of all documents with their scores to sort and filter
        scored_docs = []
        for i, score in enumerate(raw_scores):
            if score > 0:  # Only keep documents that actually match keywords
                scored_docs.append((i, score))
                
        # Sort by score descending and take Top K
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_k_results = scored_docs[:k]
        
        if not top_k_results:
            return {}
            
        # Normalize BM25 scores to 0-1 range
        max_score = top_k_results[0][1]
        
        # Build dictionary of {chunk_id: {"content", "metadata", "keyword_score"}}
        results_dict = {}
        for idx, r_score in top_k_results:
            normalized_score = r_score / max_score if max_score > 0 else 0
            chunk_id = self._bm25_doc_ids[idx]
            
            results_dict[chunk_id] = {
                "content": self._bm25_docs[idx],
                "metadata": self._bm25_metadatas[idx],
                "keyword_score": round(normalized_score, 4),
                "raw_bm25": r_score
            }
            
        return results_dict

    def retrieve(self, query: str, k: int = config.HYBRID_TOP_K) -> List[Dict[str, Any]]:
        """
        Executes both Semantic and Lexical searches, merges results, and scores them.
        """
        t_start = time.perf_counter()
        
        console.print(f"\n[bold cyan]Hybrid Query:[/bold cyan] '{query}'")
        
        # 1. Execute Keyword Search
        t0 = time.perf_counter()
        keyword_results = self.keyword_search(query, k=config.BM25_TOP_K)
        t_key = time.perf_counter() - t0
        
        # 2. Execute Semantic Search 
        # (Pass large k here so we get good candidate pools to merge, we trim at the end)
        t0 = time.perf_counter()
        semantic_results = self.semantic_retriever.retrieve(query, k=config.SEMANTIC_TOP_K)
        t_sem = time.perf_counter() - t0
        
        # 3. Merge and deduplicate
        t0 = time.perf_counter()
        merged_chunks = {}
        
        # Add Semantic Results
        for s_res in semantic_results:
            # We must fetch chunk_id from metadata or reconstruct it (assuming vector DB metadatas carry it, 
            # let's fallback to content hash if chunk_id isn't directly bound at the top level)
            chunk_id = s_res["metadata"].get("chunk_id", str(hash(s_res["content"])))
            
            merged_chunks[chunk_id] = {
                "chunk_id": chunk_id,
                "content": s_res["content"],
                "metadata": s_res["metadata"],
                "semantic_score": s_res["score"],
                "keyword_score": 0.0,
                "final_score": 0.0
            }
            
        # Add Keyword Results
        for chunk_id, k_res in keyword_results.items():
            if chunk_id in merged_chunks:
                merged_chunks[chunk_id]["keyword_score"] = k_res["keyword_score"]
            else:
                merged_chunks[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": k_res["content"],
                    "metadata": k_res["metadata"],
                    "semantic_score": 0.0,
                    "keyword_score": k_res["keyword_score"],
                    "final_score": 0.0
                }
                
        # 4. Calculate Final Weighted Scores
        for chunk in merged_chunks.values():
            sem = chunk["semantic_score"]
            key = chunk["keyword_score"]
            chunk["final_score"] = round((self.semantic_weight * sem) + (self.keyword_weight * key), 4)
            
        # 5. Sort by final score
        final_results = list(merged_chunks.values())
        final_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Trim to Top K
        final_results = final_results[:k]
        t_merge = time.perf_counter() - t0
        
        t_total = time.perf_counter() - t_start
        
        # 6. Logging
        num_semantic = len(semantic_results)
        num_keyword = len(keyword_results)
        num_merged = len(merged_chunks)
        
        console.print(f"Semantic results: {num_semantic}")
        console.print(f"Keyword results: {num_keyword}")
        console.print(f"Merged results: {num_merged}")
        
        if final_results:
            best = final_results[0]
            console.print("\n[bold]Top result:[/bold]")
            console.print(f"semantic_score: {best['semantic_score']:.2f}")
            console.print(f"keyword_score: {best['keyword_score']:.2f}")
            console.print(f"final_score: [bold green]{best['final_score']:.2f}[/bold green]")
        else:
            console.print("[yellow]No relevant chunks found in either search.[/yellow]")
            
        console.print(f"\n[dim]Timing: Semantic {t_sem*1000:.1f}ms | BM25 {t_key*1000:.1f}ms | Merge {t_merge*1000:.1f}ms | Total {t_total*1000:.1f}ms[/dim]")
        
        return final_results


# Quick test trigger block (only runs if executed directly)
if __name__ == "__main__":
    import json
    
    if len(sys.argv) > 1:
        query_str = " ".join(sys.argv[1:])
        
        console.print(f"\n[bold magenta]--- Hybrid Retrieval Test ---[/bold magenta]")
        
        retriever = HybridRetriever()
        results = retriever.retrieve(query_str)
        
        if results:
            console.print("\n[bold green]Top Match Payload Preview:[/bold green]")
            best = results[0]
            
            # Print cleanly
            cleaned = best.copy()
            cleaned["content"] = cleaned["content"][:150] + "..."
            print(json.dumps(cleaned, indent=2, ensure_ascii=False))
        else:
            console.print("[red]No results retrieved.[/red]")
    else:
        print("Usage: python src/modules/hybrid_retriever.py \"your search query here\"")
