"""
Bringer RAG System — Vector Store

Manages persistent storage of document embeddings using ChromaDB.
Handles adding, updating, and querying vectors, with built-in
support for deduplication via file hashing (incremental indexing).
"""

import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from rich.console import Console

import chromadb
from chromadb.config import Settings

import sys
import os
# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config
from src.modules.logging_utils import debug_print

console = Console()

class VectorStore:
    def __init__(self):
        """Initializes the persistent ChromaDB client and collection."""
        # Initialize persistent client pointing to our vector_db folder
        self.client = chromadb.PersistentClient(
            path=str(config.VECTOR_DB_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create the main collection.
        # We use a custom embedding function in hybrid_retriever, so we store raw vectors here.
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"description": "Bringer RAG Document Embeddings"}
        )

    def clear(self):
        """Clears the entire vector database collection."""
        console.print("Clearing vector database...")
        existing = self.collection.get(include=[])
        ids = existing.get("ids", []) if existing else []
        if ids:
            self.collection.delete(ids=ids)
        console.print("Database cleared.")

    def _get_file_hash(self, file_path: Path | str) -> str:
        """
        Calculates a SHA-256 hash of a file's contents to detect modifications.
        """
        path = Path(file_path)
        if not path.exists():
            return ""
            
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            # Read in chunks to handle potentially large files
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def is_file_indexed(self, file_path: Path | str) -> bool:
        """
        Checks if a file represents a new or modified document.
        Returns True if the file is already indexed and unmodified.
        """
        path = Path(file_path)
        current_hash = self._get_file_hash(path)
        
        # Query ChromaDB for ANY chunk belonging to this file to check its hash
        results = self.collection.get(
            where={"file_path": str(path.as_posix())},
            limit=1,
            include=["metadatas"]
        )
        
        if not results or not results["metadatas"]:
            return False
            
        # Compare the stored hash with the current file hash
        stored_hash = results["metadatas"][0].get("file_hash", "")
        return current_hash == stored_hash

    def remove_file(self, file_path: Path | str) -> int:
        """
        Deletes all chunks associated with a specific file.
        Used before re-indexing a modified file or when a file is deleted.
        
        Returns:
            Number of chunks deleted.
        """
        path_str = str(Path(file_path).as_posix())
        
        # First, count how many chunks exist for this file
        existing = self.collection.get(
            where={"file_path": path_str},
            include=["metadatas"]
        )
        
        count = len(existing["ids"]) if existing and "ids" in existing else 0
        
        if count > 0:
            self.collection.delete(where={"file_path": path_str})
            debug_print(f"[yellow]Removed {count} old chunks for {Path(file_path).name}[/yellow]")
            
        return count

    def add_chunks(self, chunks: List[Dict[str, Any]], original_file_path: Path | str) -> bool:
        """
        Stores chunks and their embeddings in ChromaDB.
        Automatically handles hashing for deduplication.
        
        Args:
            chunks: List of dictionaries from EmbeddingEngine (must have 'embedding').
            original_file_path: Path to the source file (to calculate hash).
            
        Returns:
            True if successful.
        """
        if not chunks:
            return False
            
        path = Path(original_file_path)
        file_hash = self._get_file_hash(path)
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        t0 = time.perf_counter()
        
        for chunk in chunks:
            if "embedding" not in chunk:
                console.print(f"[red]Error: Chunk {chunk.get('chunk_id')} is missing an embedding.[/red]")
                continue
                
            ids.append(chunk["chunk_id"])
            documents.append(chunk["content"])
            embeddings.append(chunk["embedding"])
            
            # Inject the file hash into the metadata for future deduplication checks
            meta = chunk["metadata"].copy()
            meta["file_hash"] = file_hash
            metadatas.append(meta)
            
        if not ids:
            return False
            
        # Add to ChromaDB in one batch
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        t_store = time.perf_counter() - t0
        debug_print(f"[green]Stored {len(ids)} vectors in ChromaDB ({t_store*1000:.1f}ms).[/green]")
        return True

    def process_file(self, file_path: Path | str) -> bool:
        """
        Runs the full indexing pipeline for a single file and stores its chunks.
        Returns True when indexing succeeds.
        """
        from src.modules.document_loader import DocumentLoader
        from src.modules.chunking_engine import ChunkingEngine
        from src.modules.embedding_engine import EmbeddingEngine

        path = Path(file_path)
        self.remove_file(path)

        loader = DocumentLoader()
        chunker = ChunkingEngine()
        embedder = EmbeddingEngine()

        pages = loader.load_document(path)
        if not pages:
            return False

        chunks = chunker.chunk_documents(pages)
        if not chunks:
            return False

        embedded_chunks = embedder.generate_embeddings(chunks)
        return self.add_chunks(embedded_chunks, path)

    def semantic_search(self, query_embedding: List[float], n_results: int = config.SEMANTIC_TOP_K) -> Dict[str, Any]:
        """
        Performs a pure vector similarity search.
        
        Args:
            query_embedding: The vector representation of the user's query.
            n_results: Number of results to return.
            
        Returns:
            ChromaDB query results dictionary.
        """
        if self.collection.count() == 0:
            debug_print("[yellow]Vector store is empty. No results found.[/yellow]")
            return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}
            
        # Chroma expects a batch of queries, so we wrap in a list
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count()),
            include=["metadatas", "documents", "distances"]
        )
        
        return results
        
    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about the current vector database."""
        total_chunks = self.collection.count()
        # To get total files, we'd need to fetch all metadata and count unique file_paths.
        # For huge DBs this is slow, but fine for local RAG.
        all_metadata = self.collection.get(include=["metadatas"])
        unique_files = set()
        if all_metadata and all_metadata["metadatas"]:
            unique_files = {m.get("file_path") for m in all_metadata["metadatas"] if m}
            
        return {
            "total_chunks": total_chunks,
            "total_documents": len(unique_files)
        }


# Quick test trigger block (only runs if executed directly)
if __name__ == "__main__":
    from document_loader import DocumentLoader
    from chunking_engine import ChunkingEngine
    from embedding_engine import EmbeddingEngine
    import sys
    
    if len(sys.argv) > 1:
        test_path_str = sys.argv[1]
        test_path = Path(test_path_str)
        
        console.print(f"\n[bold magenta]--- Vector Store Test ---[/bold magenta]")
        
        # 1. Init store
        store = VectorStore()
        initial_stats = store.get_stats()
        console.print(f"Initial DB State: {initial_stats['total_documents']} files, {initial_stats['total_chunks']} chunks.")
        
        # 2. Check Deduplication
        if store.is_file_indexed(test_path):
            console.print(f"[yellow]File '{test_path.name}' is already indexed and unmodified. Skipping injection.[/yellow]")
        else:
            console.print(f"File '{test_path.name}' is new or modified. Processing...")
            
            # Remove old chunks if it was modified
            store.remove_file(test_path)
            
            # 3. Process Pipeline
            loader = DocumentLoader()
            chunker = ChunkingEngine()
            embedder = EmbeddingEngine()
            
            pages = loader.load_document(test_path)
            if pages:
                chunks = chunker.chunk_documents(pages)
                embedded_chunks = embedder.generate_embeddings(chunks)
                
                # 4. Store
                store.add_chunks(embedded_chunks, test_path)
                
        # 5. Verify constraints
        final_stats = store.get_stats()
        console.print(f"\nFinal DB State: [green]{final_stats['total_documents']} files, {final_stats['total_chunks']} chunks.[/green]")
        
    else:
        print("Usage: python src/modules/vector_store.py <path_to_file>")
