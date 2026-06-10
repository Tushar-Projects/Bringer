"""
Bringer RAG System - CLI Launcher

Provides a single-command entry point to boot hardware detection,
server startup, model loading, file watching, and the interactive RAG pipeline.
"""

import argparse
import sys
import os
import shutil
import time
from pathlib import Path

from rich.console import Console

import config
from src.modules.logging_utils import configure_runtime_logging, debug_print
from src.modules.config_manager import get_config_manager

console = Console()

def shutdown_bringer(watcher):
    """Cleans up background services and shuts down Bringer."""
    console.print("Shutting down Bringer...")
    if watcher is not None:
        watcher.stop()
    
    try:
        from src.modules.llama_manager import get_llama_manager
        get_llama_manager().shutdown()
    except Exception:
        pass


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="bringer", add_help=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--help", "-h", action="store_true")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # models
    models_parser = subparsers.add_parser("models")
    models_subparsers = models_parser.add_subparsers(dest="models_command")
    models_subparsers.add_parser("show")
    models_subparsers.add_parser("reload")
    models_subparsers.add_parser("scan")
    
    models_profile = models_subparsers.add_parser("profile")
    models_profile.add_argument("name", choices=["low_power", "balanced", "high_performance"])
    
    models_set = models_subparsers.add_parser("set")
    models_set.add_argument("profile_name", choices=["low_power", "balanced", "high_performance"])
    models_set.add_argument("key", choices=["llm"])
    models_set.add_argument("path")
    
    # power
    power_parser = subparsers.add_parser("power")
    power_subparsers = power_parser.add_subparsers(dest="power_command")
    power_subparsers.add_parser("status")
    power_subparsers.add_parser("auto")
    power_subparsers.add_parser("low")
    power_subparsers.add_parser("balanced")
    power_subparsers.add_parser("high")
    
    # lifecycle
    subparsers.add_parser("install")
    subparsers.add_parser("update")
    
    uninstall_parser = subparsers.add_parser("uninstall")
    uninstall_parser.add_argument("--purge", action="store_true")
    
    subparsers.add_parser("doctor")
    subparsers.add_parser("init-config")

    return parser.parse_known_args(argv)


def _device_label(hw_state):
    return "GPU" if hw_state["gpu_available"] else "CPU"


def _load_runtime_modules():
    try:
        from src.modules.file_watcher import DocumentWatcher
        from src.modules.hardware_detector import HardwareDetector
        from src.modules.rag_pipeline import RAGPipeline
        from src.modules.llama_manager import get_llama_manager
        return DocumentWatcher, HardwareDetector, RAGPipeline, get_llama_manager
    except ImportError as e:
        console.print("[bold red]Bringer is missing required Python dependencies.[/bold red]")
        console.print("Please run `pip install -e .` or use `install.bat`, then try again.")
        debug_print(f"[dim]Import error: {e}[/dim]")
        return None


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
    """Shows vector database and system status, then exits."""
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

    try:
        from src.modules.llama_manager import get_llama_manager
        mgr = get_llama_manager()
        if mgr.is_model_loaded():
            console.print(f"\nActive model: {mgr.current_model_path}")
        else:
            console.print("\nNo LLM currently loaded in memory.")
    except Exception:
        console.print("\nLLM Manager not available.")


def show_help():
    """Displays the Bringer CLI help menu and exits."""
    console.print("[bold cyan]Bringer - Local AI Document Assistant[/bold cyan]\n")
    console.print("Usage:")
    console.print("  bringer                 Start the assistant")
    console.print("  bringer --debug         Run with detailed logs")
    console.print("  bringer --status        Show indexed files and system status")
    console.print("  bringer --reindex       Rebuild the document index")
    console.print("  bringer models ...      Manage models")
    console.print("  bringer power ...       Manage power profiles")
    console.print("  bringer doctor          Check system health")
    console.print("  bringer --help          Show this help message\n")


def cmd_models(args):
    config_manager = get_config_manager()
    if args.models_command == "show":
        console.print("[bold]Models Configuration:[/bold]")
        console.print(f"Active Mode: {config_manager.get_active_mode()}")
        for profile in ["low_power", "balanced", "high_performance"]:
            cfg = config_manager.get_profile_config(profile)
            console.print(f"\n[cyan]Profile: {profile}[/cyan]")
            llm_path = cfg.get("llm", {}).get("path", "Not configured")
            emb_name = cfg.get("embedding", {}).get("name", "Not configured")
            rerank_name = cfg.get("reranker", {}).get("name", "Not configured")
            console.print(f"  LLM: {llm_path}")
            console.print(f"  Embedding: {emb_name}")
            console.print(f"  Reranker: {rerank_name}")
            
    elif args.models_command == "reload":
        try:
            from src.modules.llama_manager import get_llama_manager
            from src.modules.hardware_detector import HardwareDetector
            mgr = get_llama_manager()
            mgr._unload_model()
            active_mode = config_manager.get_active_mode()
            if active_mode == "auto":
                profile_name = HardwareDetector().select_profile()
            else:
                profile_name = active_mode
            if mgr.load_model(profile_name):
                console.print(f"[green]Successfully reloaded model for profile '{profile_name}'.[/green]")
            else:
                console.print("[red]Failed to reload model.[/red]")
        except ImportError:
            console.print("[red]Required modules missing.[/red]")

    elif args.models_command == "scan":
        console.print("[cyan]Scanning for .gguf models...[/cyan]")
        found = False
        for root, _, files in os.walk(config.PROJECT_ROOT):
            for file in files:
                if file.endswith(".gguf"):
                    console.print(f"Found: {os.path.join(root, file)}")
                    found = True
        if not found:
            console.print("No .gguf models found.")
            
    elif args.models_command == "profile":
        config_manager.set_active_mode(args.name)
        console.print(f"[green]Active profile set to {args.name}[/green]")
        
    elif args.models_command == "set":
        if args.key == "llm":
            config_manager.set_profile_llm_path(args.profile_name, args.path)
            console.print(f"[green]Set LLM path for {args.profile_name} to {args.path}[/green]")

def cmd_power(args):
    config_manager = get_config_manager()
    if args.power_command == "status":
        from src.modules.hardware_detector import HardwareDetector
        hw_state = HardwareDetector().detect_hardware()
        console.print("[bold]Power Status:[/bold]")
        console.print(f"Plugged in: {hw_state['plugged_in']}")
        console.print(f"Power Saver: {hw_state['power_saver']}")
        console.print(f"GPU Available: {hw_state['gpu_available']} ({hw_state['gpu_name']})")
        console.print(f"Auto-selected Profile: {HardwareDetector().select_profile()}")
        console.print(f"Current Config Mode: {config_manager.get_active_mode()}")
        
    elif args.power_command in ["auto", "low", "balanced", "high"]:
        mapping = {
            "auto": "auto",
            "low": "low_power",
            "balanced": "balanced",
            "high": "high_performance"
        }
        config_manager.set_active_mode(mapping[args.power_command])
        console.print(f"[green]Power mode set to {args.power_command}[/green]")

def cmd_lifecycle(command):
    if command == "install":
        console.print("[bold]Bringer Installation Check[/bold]")
        
        # Check llama-cpp
        try:
            import llama_cpp
            console.print("[green]✓ llama-cpp installed[/green]")
        except ImportError:
            console.print("[yellow]⚠ llama-cpp not installed[/yellow]")
            
        # Check config
        config_manager = get_config_manager()
        if (config.PROJECT_ROOT / "models.json").exists():
            console.print("[green]✓ models.json found[/green]")
        else:
            console.print("[yellow]⚠ models.json not found[/yellow]")
            
        # Check profiles
        for profile in ["low_power", "balanced", "high_performance"]:
            cfg = config_manager.get_profile_config(profile)
            llm_path = cfg.get("llm", {}).get("path")
            if llm_path and os.path.exists(llm_path):
                console.print(f"[green]✓ LLM found for {profile}: {os.path.basename(llm_path)}[/green]")
            elif llm_path:
                console.print(f"[red]⚠ LLM path configured for {profile} but file not found: {llm_path}[/red]")
            else:
                console.print(f"[yellow]⚠ LLM not configured for {profile}[/yellow]")
                
    elif command == "update":
        console.print("[green]Update check complete. Existing configurations preserved.[/green]")
        
    elif command == "doctor":
        console.print("[bold]Bringer System Doctor[/bold]")
        cmd_lifecycle("install")
        from src.modules.hardware_detector import HardwareDetector
        hw = HardwareDetector().detect_hardware()
        console.print(f"[cyan]Hardware:[/cyan] GPU: {hw['gpu_available']}, Power: {'AC' if hw['plugged_in'] else 'Battery'}")
        console.print("[green]Doctor check finished.[/green]")
        
    elif command == "init-config":
        path = config.PROJECT_ROOT / "models.json"
        if not path.exists():
            get_config_manager().save_config(get_config_manager().raw_config)
            console.print(f"[green]Created {path}[/green]")
        else:
            console.print(f"[yellow]File {path} already exists. Skipping.[/yellow]")

def launch_bringer(argv=None):
    """Main entry point executed when the user runs the `bringer` CLI command."""
    args, _ = _parse_args(argv)
    configure_runtime_logging(args.debug)

    if args.help:
        show_help()
        return

    if getattr(args, "command", None):
        if args.command == "models":
            cmd_models(args)
            return
        elif args.command == "power":
            cmd_power(args)
            return
        elif args.command in ["install", "update", "doctor", "init-config"]:
            cmd_lifecycle(args.command)
            return
        elif args.command == "uninstall":
            if getattr(args, "purge", False):
                console.print("[bold red]Purge requested. This would delete models, DB, and config.[/bold red]")
                console.print("Please manually delete the Bringer directory.")
            else:
                console.print("Please use `pip uninstall bringer`.")
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

    DocumentWatcher, HardwareDetector, RAGPipeline, get_llama_manager_func = runtime_modules

    watcher = None

    try:
        debug_print("[dim]Detecting hardware...[/dim]")
        detector = HardwareDetector()
        hw_state = detector.detect_hardware()

        debug_print(f"GPU detected: [cyan]{hw_state['gpu_name'] if hw_state['gpu_available'] else 'None'}[/cyan]")
        debug_print(f"Power state: [cyan]{'Plugged in' if hw_state['plugged_in'] else 'On battery'}[/cyan]")
        debug_print(f"Power saver: [cyan]{'Enabled' if hw_state['power_saver'] else 'Disabled'}[/cyan]\n")

        config_manager = get_config_manager()
        active_mode = config_manager.get_active_mode()
        
        if active_mode == "auto":
            selected_profile = detector.select_profile()
        else:
            selected_profile = active_mode
            
        debug_print(f"Selected profile: [bold green]{selected_profile}[/bold green]\n")

        debug_print("\n[dim]Booting RAG pipeline...[/dim]")
        t0 = time.perf_counter()

        watcher = DocumentWatcher()
        watcher.start()

        pipeline = RAGPipeline()
        
        # Pre-load model to catch errors early
        mgr = get_llama_manager_func()
        if not mgr.load_model(selected_profile):
            console.print("[yellow]Bringer started, but no valid LLM is loaded. Vector search will work, but generation will fail.[/yellow]")
            console.print("[yellow]Use `bringer models set ...` to configure a model path.[/yellow]")

        t_boot = time.perf_counter() - t0

        if config.DEBUG_MODE:
            console.print("\n[bold magenta]--- Bringer RAG Assistant ---[/bold magenta]\n")
            console.print(f"[bold green]RAG Assistant Ready[/bold green] [dim]({t_boot:.1f}s)[/dim]")
            console.print("Ask a question (type 'exit' or 'quit' to close)")
        else:
            console.print("Bringer AI Assistant\n")
            console.print(f"Profile: {selected_profile} ({_device_label(hw_state)})")
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
        shutdown_bringer(watcher)


if __name__ == "__main__":
    launch_bringer()
