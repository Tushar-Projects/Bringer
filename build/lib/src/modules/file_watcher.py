"""
Bringer RAG System — File Watcher

Monitors the documents/ directory for changes and automatically
indexes new or modified files, and removes deleted files from the vector store.
"""

import time
from pathlib import Path
from typing import Dict
from rich.console import Console

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import sys
import os
# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

from src.modules.document_loader import DocumentLoader
from src.modules.chunking_engine import ChunkingEngine
from src.modules.embedding_engine import EmbeddingEngine
from src.modules.vector_store import VectorStore

console = Console()

class DocumentIndexingHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.loader = DocumentLoader()
        self.chunker = ChunkingEngine()
        self.embedder = EmbeddingEngine()
        self.store = VectorStore()
        
        # Debounce dictionary: filepath -> last_processed_time
        self.last_processed: Dict[str, float] = {}
        self.debounce_seconds = 1.0

    def _is_valid_file(self, file_path: Path) -> bool:
        """Checks if the file should be processed."""
        if file_path.is_dir():
            return False
            
        name = file_path.name
        # Ignore temporary files
        if name.startswith('~$') or name.startswith('.') or name.endswith('.tmp') or name.endswith('.lock'):
            return False
            
        # Only process supported formats
        ext = file_path.suffix.lower()
        if ext not in ['.pdf', '.docx', '.pptx', '.txt', '.md']:
            return False
            
        return True

    def _should_debounce(self, file_path_str: str) -> bool:
        """Returns True if the event should be ignored due to debouncing."""
        now = time.time()
        if file_path_str in self.last_processed:
            if now - self.last_processed[file_path_str] < self.debounce_seconds:
                return True
        self.last_processed[file_path_str] = now
        return False

    def _index_document(self, file_path: Path):
        """Runs the full indexing pipeline for a single document."""
        console.print(f"\n[bold cyan]Indexing document...[/bold cyan]")
        
        t0 = time.perf_counter()
        
        # 1. Check if already indexed and unmodified
        if self.store.is_file_indexed(file_path):
            console.print(f"[dim]File '{file_path.name}' is already up-to-date.[/dim]")
            return
            
        # Remove old chunks if it's a modification
        self.store.remove_file(file_path)
        
        # 2. Extract Text
        pages = self.loader.load_document(file_path)
        if not pages:
            console.print(f"[yellow]Failed to extract text from {file_path.name}[/yellow]")
            return
            
        # 3. Chunk
        chunks = self.chunker.chunk_documents(pages)
        if not chunks:
            console.print(f"[yellow]No valid chunks generated from {file_path.name}[/yellow]")
            return
            
        # 4. Embed
        embedded_chunks = self.embedder.generate_embeddings(chunks)
        
        # 5. Store
        success = self.store.add_chunks(embedded_chunks, file_path)
        
        t_total = time.perf_counter() - t0
        
        if success:
            console.print(f"Chunks generated: {len(chunks)}")
            console.print(f"Embeddings created: {len(embedded_chunks)}")
            console.print(f"[bold green]Vector database updated ({t_total:.1f}s).[/bold green]\n")

    def _remove_document(self, file_path: Path):
        """Removes a document from the vector store."""
        console.print(f"\n[yellow]Removing {file_path.name} from vector database...[/yellow]")
        self.store.remove_file(file_path)

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._is_valid_file(path) and not self._should_debounce(str(path)):
            console.print(f"\n[cyan]File detected:[/cyan] {path.name}")
            self._index_document(path)

    def on_modified(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._is_valid_file(path) and not self._should_debounce(str(path)):
            console.print(f"\n[cyan]File modified:[/cyan] {path.name}")
            self._index_document(path)

    def on_deleted(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._is_valid_file(path):
            console.print(f"\n[red]File deleted:[/red] {path.name}")
            self._remove_document(path)


class DocumentWatcher:
    def __init__(self):
        self.observer = Observer()
        self.handler = DocumentIndexingHandler()
        self.watch_dir = str(config.DOCUMENTS_DIR)
        
    def start(self):
        """Starts the observer in a background thread."""
        config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.observer.schedule(self.handler, self.watch_dir, recursive=False)
        self.observer.start()
        console.print(f"[dim]Starting document watcher on {self.watch_dir}...[/dim]")
        
    def stop(self):
        """Stops the observer."""
        self.observer.stop()
        self.observer.join()

# For standalone testing
if __name__ == "__main__":
    console.print("[bold green]--- File Watcher Test ---[/bold green]")
    watcher = DocumentWatcher()
    watcher.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()
        console.print("\n[dim]Watcher stopped.[/dim]")
