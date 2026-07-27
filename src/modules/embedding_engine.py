"""
Bringer RAG System — Embedding Engine

Responsible for converting document chunks into dense vector representations.
Uses `sentence-transformers` with automatic GPU acceleration, batched processing,
and strict memory management suitable for an 8 GB VRAM budget.
"""

import os
import sys
import time
from functools import lru_cache
from typing import Any

import torch
from rich.console import Console
from sentence_transformers import SentenceTransformer

# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config
from src.modules.config_manager import get_config_manager
from src.modules.hardware_detector import HardwareDetector
from src.modules.logging_utils import debug_print

console = Console()

# =============================================================================
# Global Initializations (Performance Optimization)
# Initialize the model once globally to avoid reloading overhead per function call.
# =============================================================================
_MODEL = None
_CURRENT_MODEL_NAME = None

def get_embedding_model() -> SentenceTransformer:
    """Lazy loader for the embedding model to ensure it is only loaded once per config change."""
    global _MODEL, _CURRENT_MODEL_NAME
    
    config_manager = get_config_manager()
    active_mode = config_manager.get_active_mode()
    
    if active_mode == "auto":
        profile_name = HardwareDetector().select_profile()
    else:
        profile_name = active_mode
        
    profile_config = config_manager.get_profile_config(profile_name)
    model_name = profile_config.get("embedding", {}).get("name", "sentence-transformers/all-MiniLM-L6-v2")
    
    if _MODEL is None or _CURRENT_MODEL_NAME != model_name:
        debug_print(f"[dim]Loading embedding model '{model_name}' onto {config.DEVICE}...[/dim]")
        t0 = time.perf_counter()
        
        # Load the model directly onto the target device (GPU/CPU)
        _MODEL = SentenceTransformer(model_name, device=config.DEVICE)
        _CURRENT_MODEL_NAME = model_name
        
        load_time = time.perf_counter() - t0
        debug_print(f"[green]Model loaded in {load_time:.2f}s.[/green]")
    return _MODEL

class EmbeddingEngine:
    def __init__(self):
        """Initialize the Embedding Engine (loads model implicitly via singleton)."""
        self.model = get_embedding_model()
        
        config_manager = get_config_manager()
        active_mode = config_manager.get_active_mode()
        if active_mode == "auto":
            profile_name = HardwareDetector().select_profile()
        else:
            profile_name = active_mode
            
        profile_config = config_manager.get_profile_config(profile_name)
        emb_config = profile_config.get("embedding", {})
        
        self.batch_size = emb_config.get("batch_size", 64)
        self.expected_dimensions = emb_config.get("dimensions", 384)

    @lru_cache(maxsize=128)  # noqa: B019
    def embed_query(self, query: str) -> list[float]:
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
    def generate_embeddings(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        
        debug_print(f"[dim]Generating embeddings for {len(texts)} chunks in batches of {self.batch_size}...[/dim]")
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
            
        debug_print(f"[green]Generated {len(texts)} embeddings in {t_embed:.2f}s "
                    f"({len(texts)/t_embed:.1f} chunks/sec).[/green]")
                      
        # Periodically clear CUDA cache if we processed a massive batch
        if config.DEVICE == "cuda" and len(texts) > self.batch_size * 5:
            torch.cuda.empty_cache()
            
        return chunks


# Quick test trigger block (only runs if executed directly)
if __name__ == "__main__":
    import sys
    from pathlib import Path

    from chunking_engine import ChunkingEngine
    from document_loader import DocumentLoader
    
    if len(sys.argv) > 1:
        test_path_str = sys.argv[1]
        test_path = Path(test_path_str)
        
        loader = DocumentLoader()
        engine = ChunkingEngine()
        embedder = EmbeddingEngine()
        
        console.print("\n[bold magenta]--- Embedding Pipeline Test ---[/bold magenta]")
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
                
                if embedding_dim == embedder.expected_dimensions:
                    console.print(f"[green]Dimension matches expected config ({embedder.expected_dimensions})[/green]")
                else:
                    console.print(f"[red]Warning: Expected {embedder.expected_dimensions} dimensions, got {embedding_dim}[/red]")
        else:
            console.print("[red]Failed to load text for testing.[/red]")
    else:
        print("Usage: python src/modules/embedding_engine.py <path_to_file>")
