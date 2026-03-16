"""
Bringer RAG System — Embedding Engine

Responsible for converting document chunks into dense vector representations.
Uses `sentence-transformers` with automatic GPU acceleration, batched processing,
and strict memory management suitable for an 8 GB VRAM budget.
"""

import time
import torch
from typing import List, Dict, Any
from rich.console import Console
from sentence_transformers import SentenceTransformer
from functools import lru_cache

import sys
import os
# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

console = Console()

# =============================================================================
# Global Initializations (Performance Optimization)
# Initialize the model once globally to avoid reloading overhead per function call.
# =============================================================================
_MODEL = None

def get_embedding_model() -> SentenceTransformer:
    """Lazy loader for the embedding model to ensure it is only loaded once."""
    global _MODEL
    if _MODEL is None:
        console.print(f"[dim]Loading embedding model '{config.EMBEDDING_MODEL_NAME}' onto {config.DEVICE}...[/dim]")
        t0 = time.perf_counter()
        
        # Load the model directly onto the target device (GPU/CPU)
        _MODEL = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.DEVICE)
        
        load_time = time.perf_counter() - t0
        console.print(f"[green]Model loaded in {load_time:.2f}s.[/green]")
    return _MODEL

class EmbeddingEngine:
    def __init__(self):
        """Initialize the Embedding Engine (loads model implicitly via singleton)."""
        self.model = get_embedding_model()
        self.batch_size = config.EMBEDDING_BATCH_SIZE

    @lru_cache(maxsize=128)
    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string. Cached using LRU to prevent repeated computation.
        
        Args:
            query: The user's text query.
            
        Returns:
            A list of floats representing the dense vector.
        """
        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            convert_to_tensor=False,
            normalize_embeddings=True 
        )
        return embedding.tolist()

    @torch.no_grad()
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes a list of chunk dictionaries and adds an 'embedding' vector to each.
        
        Args:
            chunks: List of dictionaries (from chunking_engine.py) containing 'content'.
            
        Returns:
            The identical list of chunks, but with a new 'embedding' key added.
        """
        if not chunks:
            return []

        # Extract just the text contents to embed
        texts = [chunk["content"] for chunk in chunks]
        
        console.print(f"[dim]Generating embeddings for {len(texts)} chunks in batches of {self.batch_size}...[/dim]")
        t0 = time.perf_counter()
        
        # sentence-transformers automatically handles batching, memory mapping, 
        # and optimized inference (tqdm progress bar disabled for cleaner logs)
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=False,  # Return standard numpy arrays/lists for storage
            normalize_embeddings=True # Normalization helps cosine similarity search
        )
        
        t_embed = time.perf_counter() - t0
        
        # Attach the embeddings back to the original dictionary structures
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist() # Convert numpy array to standard python list
            
        console.print(f"[green]Generated {len(texts)} embeddings in {t_embed:.2f}s "
                      f"({len(texts)/t_embed:.1f} chunks/sec).[/green]")
                      
        # Periodically clear CUDA cache if we processed a massive batch
        if config.DEVICE == "cuda" and len(texts) > self.batch_size * 5:
            torch.cuda.empty_cache()
            
        return chunks


# Quick test trigger block (only runs if executed directly)
if __name__ == "__main__":
    from document_loader import DocumentLoader
    from chunking_engine import ChunkingEngine
    import sys
    from pathlib import Path
    
    if len(sys.argv) > 1:
        test_path_str = sys.argv[1]
        test_path = Path(test_path_str)
        
        loader = DocumentLoader()
        engine = ChunkingEngine()
        embedder = EmbeddingEngine()
        
        console.print(f"\n[bold magenta]--- Embedding Pipeline Test ---[/bold magenta]")
        pages = loader.load_document(test_path)
        
        if pages:
            chunks = engine.chunk_documents(pages)
            
            if chunks:
                console.print(f"\n[bold cyan]Embedding {len(chunks)} chunks...[/bold cyan]")
                embedded_chunks = embedder.generate_embeddings(chunks)
                
                first_chunk = embedded_chunks[0]
                embedding_dim = len(first_chunk["embedding"])
                
                console.print("\n[bold green]Success![/bold green]")
                console.print(f"Chunks processed: {len(embedded_chunks)}")
                console.print(f"Embedding dimension: {embedding_dim}")
                console.print(f"Device used: {embedder.model.device}")
                
                if embedding_dim == config.EMBEDDING_DIMENSIONS:
                    console.print(f"[green]Dimension matches expected config ({config.EMBEDDING_DIMENSIONS})[/green]")
                else:
                    console.print(f"[red]Warning: Expected {config.EMBEDDING_DIMENSIONS} dimensions, got {embedding_dim}[/red]")
        else:
            console.print("[red]Failed to load text for testing.[/red]")
    else:
        print("Usage: python src/modules/embedding_engine.py <path_to_file>")
