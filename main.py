"""
Bringer RAG System — Main Entry Point

A simple Command-Line Interface (CLI) loop to interact with the RAG Orchestrator.
Allows the user to continuously ask questions without restarting the script,
streaming the answers back in real-time.
"""

import sys
import os
import time
from rich.console import Console

# Ensure our local modules can be found
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.modules.rag_pipeline import RAGPipeline
from src.modules.file_watcher import DocumentWatcher

console = Console()

def main():
    console.print("\n[bold green]--- Bringer RAG System ---[/bold green]")
    console.print("Booting up RAG Pipeline... (This may take a moment to load models into VRAM)\n")
    
    t0 = time.perf_counter()
    
    # Init and start background Watcher
    watcher = DocumentWatcher()
    watcher.start()
    
    # Init active pipeline
    pipeline = RAGPipeline()
    t_boot = time.perf_counter() - t0
    
    console.print(f"[bold green]RAG Assistant Ready[/bold green] [dim](Booted in {t_boot:.1f}s)[/dim]")
    console.print("Ask a question (type 'exit' or 'quit' to close)")
    
    while True:
        try:
            # Simple input prompt
            query = console.input("\n[bold yellow]>[/bold yellow] ").strip()
            
            if not query:
                continue
                
            if query.lower() in ("exit", "quit"):
                console.print("[dim]Stopping background watcher...[/dim]")
                watcher.stop()
                console.print("[dim]Shutting down Bringer. Goodbye![/dim]\n")
                break
                
            # We catch the generator outputs and print them to the terminal
            console.print("\n[bold cyan]Answer:[/bold cyan] ", end="")
            
            for token in pipeline.run_rag(query):
                # The generator yields raw strings. We print them without newlines to simulate streaming.
                print(token, end="", flush=True)
                
            print("\n")  # Ensure a clean line break after the response finishes
            
        except KeyboardInterrupt:
            # Handle Ctrl+C cleanly
            console.print("\n[dim]Stopping background watcher...[/dim]")
            watcher.stop()
            console.print("[dim]Shutting down Bringer. Goodbye![/dim]\n")
            break
        except Exception as e:
            console.print(f"\n[bold red]An unexpected error occurred:[/bold red] {str(e)}\n")

if __name__ == "__main__":
    main()
