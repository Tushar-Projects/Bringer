"""
Bringer RAG System - CLI Launcher

Provides a single-command entry point to boot hardware detection,
server startup, model loading, file watching, and the interactive RAG pipeline.
"""

import argparse
import time

from rich.console import Console

import config
from src.modules.file_watcher import DocumentWatcher
from src.modules.hardware_detector import HardwareDetector
from src.modules.lmstudio_manager import LMStudioManager
from src.modules.logging_utils import configure_runtime_logging, debug_print
from src.modules.rag_pipeline import RAGPipeline
from src.modules.vector_store import VectorStore

console = Console()


def shutdown_lmstudio(lm_manager: LMStudioManager):
    """Stops LM Studio only when Bringer launched the server."""
    lm_manager.stop_server()


def shutdown_bringer(watcher, lm_manager: LMStudioManager):
    """Cleans up background services and shuts down Bringer."""
    console.print("Shutting down Bringer...")
    if watcher is not None:
        watcher.stop()

    if lm_manager is not None:
        shutdown_lmstudio(lm_manager)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_known_args(argv)


def _device_label(hw_state):
    return "GPU" if hw_state["gpu_available"] else "CPU"


def _documents_indexed() -> int:
    try:
        return VectorStore().get_stats()["total_documents"]
    except Exception:
        return 0


def launch_bringer(argv=None):
    """Main entry point executed when the user runs the `Bringer` CLI command."""
    args, _ = _parse_args(argv)
    configure_runtime_logging(args.debug)

    watcher = None
    lm_manager = None

    debug_print("[dim]Detecting hardware...[/dim]")
    detector = HardwareDetector()
    hw_state = detector.detect_hardware()

    debug_print(f"GPU detected: [cyan]{hw_state['gpu_name'] if hw_state['gpu_available'] else 'None'}[/cyan]")
    debug_print(f"Power state: [cyan]{'Plugged in' if hw_state['plugged_in'] else 'On battery'}[/cyan]\n")

    selected_model = detector.select_model()
    debug_print(f"Selected model: [bold green]{selected_model}[/bold green]\n")

    config.LLM_MODEL_NAME = selected_model

    debug_print("[dim]Starting LM Studio server...[/dim]")
    lm_manager = LMStudioManager()
    lm_manager.ensure_ready(selected_model)

    debug_print("\n[dim]Booting RAG pipeline...[/dim]")
    t0 = time.perf_counter()

    watcher = DocumentWatcher()
    watcher.start()

    pipeline = RAGPipeline()
    t_boot = time.perf_counter() - t0

    if config.DEBUG_MODE:
        console.print("\n[bold magenta]--- Bringer RAG Assistant ---[/bold magenta]\n")
        console.print(f"[bold green]RAG Assistant Ready[/bold green] [dim]({t_boot:.1f}s)[/dim]")
        console.print("Ask a question (type 'exit' or 'quit' to close)")
    else:
        console.print("Bringer AI Assistant\n")
        console.print(f"Model: {selected_model} ({_device_label(hw_state)})")
        console.print(f"Documents indexed: {_documents_indexed()}\n")
        console.print("Ready.")
        console.print("Ask a question or type 'exit'.")

    try:
        while True:
            query = input("\n> ").strip()

            if not query:
                continue

            if query.lower() in ("exit", "quit"):
                shutdown_bringer(watcher, lm_manager)
                break

            if config.DEBUG_MODE:
                console.print("\n[bold cyan]Answer:[/bold cyan] ", end="")
            else:
                console.print()

            for token in pipeline.run_rag(query):
                print(token, end="", flush=True)

            print("\n")

    except KeyboardInterrupt:
        shutdown_bringer(watcher, lm_manager)
    except Exception as e:
        console.print(f"\n[bold red]Fatal System Error:[/bold red] {str(e)}\n")
        shutdown_bringer(watcher, lm_manager)


if __name__ == "__main__":
    launch_bringer()
