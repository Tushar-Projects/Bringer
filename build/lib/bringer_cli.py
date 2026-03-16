"""
Bringer RAG System — CLI Launcher

Provides a single-command entry point to boot hardware detection,
server startup, model loading, file watching, and the interactive RAG pipeline.
"""

import sys
import time
from rich.console import Console

# Import config first
import config

# Import modules
from src.modules.hardware_detector import HardwareDetector
from src.modules.lmstudio_manager import LMStudioManager
from src.modules.file_watcher import DocumentWatcher
from src.modules.rag_pipeline import RAGPipeline

console = Console()

def launch_bringer():
    """Main entry point executed when the user runs the `Bringer` CLI command."""
    console.print("\n[bold magenta]--- Bringer RAG Assistant ---[/bold magenta]\n")
    
    # 1. Hardware Detection
    console.print("[dim]Detecting hardware...[/dim]")
    detector = HardwareDetector()
    hw_state = detector.detect_hardware()
    
    console.print(f"GPU detected: [cyan]{hw_state['gpu_name'] if hw_state['gpu_available'] else 'None'}[/cyan]")
    console.print(f"Power state: [cyan]{'Plugged in' if hw_state['plugged_in'] else 'On battery'}[/cyan]\n")
    
    # Select dynamically
    selected_model = detector.select_model()
    console.print(f"Selected model: [bold green]{selected_model}[/bold green]\n")
    
    # Overwrite config globally before anything else loads
    config.LLM_MODEL_NAME = selected_model
    
    # 2. LM Studio Check & Auto-Start
    console.print("[dim]Starting LM Studio server...[/dim]")
    lm_manager = LMStudioManager()
    lm_manager.ensure_ready(selected_model)
    console.print("[dim]Loading model...[/dim]")
    
    # 3. Boot Pipeline
    console.print("\n[dim]Booting RAG pipeline...[/dim]")
    t0 = time.perf_counter()
    
    # Spin up background watcher for documents/
    watcher = DocumentWatcher()
    watcher.start()
    
    # Spin up RAG generator (loads embedding engines to Device safely)
    pipeline = RAGPipeline()
    t_boot = time.perf_counter() - t0
    
    console.print(f"\n[bold green]RAG Assistant Ready[/bold green] [dim]({t_boot:.1f}s)[/dim]")
    console.print("Ask a question (type 'exit' or 'quit' to close)")
    
    # 4. Interactive Chat Loop
    try:
        while True:
            query = input("\n> ").strip()
            
            if not query:
                continue
                
            if query.lower() in ("exit", "quit"):
                console.print("\n[dim]Stopping background watcher...[/dim]")
                watcher.stop()
                console.print("[dim]Shutting down Bringer. Goodbye![/dim]\n")
                break
                
            console.print("\n[bold cyan]Answer:[/bold cyan] ", end="")
            
            # Generators stream response chunks natively from TTFB
            for token in pipeline.run_rag(query):
                print(token, end="", flush=True)
                
            print("\n")
            
    except KeyboardInterrupt:
        console.print("\n\n[dim]Stopping background watcher...[/dim]")
        watcher.stop()
        console.print("[dim]Shutting down Bringer. Goodbye![/dim]\n")
    except Exception as e:
        console.print(f"\n[bold red]Fatal System Error:[/bold red] {str(e)}\n")
        watcher.stop()


# Simple testing trigger
if __name__ == "__main__":
    launch_bringer()
