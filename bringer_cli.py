"""
Bringer RAG System - CLI Launcher

Provides a single-command entry point to boot hardware detection,
server startup, model loading, file watching, and the interactive RAG pipeline.
"""

import argparse
import shutil
import time
from pathlib import Path

from rich.console import Console

import config
from src.modules.logging_utils import configure_runtime_logging, debug_print

console = Console()


def shutdown_lmstudio(lm_manager):
    """Stops LM Studio only when Bringer launched the server."""
    lm_manager.shutdown()


def shutdown_bringer(watcher, lm_manager):
    """Cleans up background services and shuts down Bringer."""
    console.print("Shutting down Bringer...")
    if watcher is not None:
        watcher.stop()

    if lm_manager is not None:
        shutdown_lmstudio(lm_manager)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--help", "-h", action="store_true")
    return parser.parse_known_args(argv)


def _device_label(hw_state):
    return "GPU" if hw_state["gpu_available"] else "CPU"


def _documents_indexed() -> int:
    try:
        from src.modules.vector_store import VectorStore
        return VectorStore().get_stats()["total_documents"]
    except Exception:
        return 0


def _load_runtime_modules():
    try:
        from src.modules.file_watcher import DocumentWatcher
        from src.modules.hardware_detector import HardwareDetector
        from src.modules.lmstudio_manager import LMStudioManager
        from src.modules.rag_pipeline import RAGPipeline
        return DocumentWatcher, HardwareDetector, LMStudioManager, RAGPipeline
    except ImportError as e:
        console.print("[bold red]Bringer is missing required Python dependencies.[/bold red]")
        console.print("Please run `pip install -e .` or use `install.bat`, then try again.")
        debug_print(f"[dim]Import error: {e}[/dim]")
        return None


def _check_lmstudio_cli() -> bool:
    if shutil.which("lms"):
        return True

    console.print("[bold red]LM Studio CLI was not found.[/bold red]")
    console.print("Install LM Studio, enable the `lms` command, and then run Bringer again.")
    return False


def run_reindex_mode():
    """Clears and rebuilds the local vector database from documents/."""
    from src.modules.hybrid_retriever import HybridRetriever
    from src.modules.vector_store import VectorStore

    console.print("Reindexing documents...")

    store = VectorStore()
    store.clear()

    docs_path = Path(config.DOCUMENTS_DIR)
    docs_path.mkdir(parents=True, exist_ok=True)

    supported_extensions = config.SUPPORTED_EXTENSIONS
    total_files = 0

    for file_path in sorted(docs_path.iterdir(), key=lambda path: path.name.lower()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in supported_extensions:
            continue

        console.print(f"Indexing {file_path.name}...")
        store.process_file(file_path)
        total_files += 1

    retriever = HybridRetriever()
    retriever.rebuild_bm25_index()

    console.print(f"Reindex complete. {total_files} files processed.")


def run_status():
    """Shows vector database and LM Studio status, then exits."""
    from src.modules.lmstudio_manager import LMStudioManager
    from src.modules.vector_store import VectorStore

    console.print("Bringer Status\n")

    store = VectorStore()
    total_chunks = store.collection.count()
    results = store.collection.get(include=["metadatas"])

    files = set()
    for meta in results.get("metadatas", []) if results else []:
        if meta and "source_file" in meta:
            files.add(meta["source_file"])

    console.print(f"Indexed files: {len(files)}")
    console.print(f"Total chunks: {total_chunks}\n")

    console.print("Files:")
    if files:
        for file_name in sorted(files):
            console.print(f"- {file_name}")
    else:
        console.print("- None")

    lm_manager = LMStudioManager()
    try:
        loaded_models = lm_manager.get_loaded_models()
        if loaded_models:
            console.print("\nActive model:")
            for model_name in loaded_models:
                console.print(f"- {model_name}")
        elif lm_manager.is_server_running():
            console.print("\nNo model currently loaded.")
        else:
            console.print("\nLM Studio not running.")
    except Exception:
        console.print("\nLM Studio not running.")


def show_help():
    """Displays the Bringer CLI help menu and exits."""
    console.print("[bold cyan]Bringer - Local AI Document Assistant[/bold cyan]\n")
    console.print("Usage:")
    console.print("  Bringer                 Start the assistant")
    console.print("  Bringer --debug         Run with detailed logs")
    console.print("  Bringer --status        Show indexed files and system status")
    console.print("  Bringer --reindex       Rebuild the document index")
    console.print("  Bringer --help          Show this help message\n")
    console.print("Description:")
    console.print("  Bringer is a CLI-based AI assistant that reads your local documents")
    console.print("  and answers questions using local language models.\n")
    console.print("Examples:")
    console.print("  Bringer")
    console.print("  Bringer --status")
    console.print("  Bringer --reindex")


def launch_bringer(argv=None):
    """Main entry point executed when the user runs the `Bringer` CLI command."""
    args, _ = _parse_args(argv)
    configure_runtime_logging(args.debug)

    if args.help:
        show_help()
        return

    if args.reindex:
        run_reindex_mode()
        return

    if args.status:
        run_status()
        return

    runtime_modules = _load_runtime_modules()
    if runtime_modules is None:
        return

    if not _check_lmstudio_cli():
        return

    DocumentWatcher, HardwareDetector, LMStudioManager, RAGPipeline = runtime_modules

    watcher = None
    lm_manager = None

    try:
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
            console.print("Ready.")

        while True:
            query = input("\n> ").strip()

            if not query:
                continue

            if query.lower() in ("exit", "quit"):
                break

            if config.DEBUG_MODE:
                console.print("\n[bold cyan]Answer:[/bold cyan] ", end="")
            else:
                console.print()

            for token in pipeline.run_rag(query):
                print(token, end="", flush=True)

            print("\n")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"\n[bold red]Fatal System Error:[/bold red] {str(e)}\n")
    finally:
        if watcher is not None or lm_manager is not None:
            shutdown_bringer(watcher, lm_manager)


if __name__ == "__main__":
    launch_bringer()
