"""
Bringer RAG System - LM Studio Manager

Responsible for auto-starting the local LM Studio server if it is offline.
Checks loaded models via the API, unloads incorrect models, and
ensures the exact requested model is loaded before generation begins.
"""

import os
import subprocess
import sys
import time
from typing import List

import httpx
from rich.console import Console

# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config
from src.modules.logging_utils import debug_print

console = Console()


class LMStudioManager:
    def __init__(self):
        self.api_base = config.LLM_API_BASE
        self.timeout = 5.0
        self.model_poll_interval = 2.0
        self.model_load_timeout = 120.0
        self.unload_wait_seconds = 3.0
        self.readiness_timeout = 30.0
        self.started_by_bringer = False

    def is_server_running(self) -> bool:
        """Pings the /v1/models endpoint to check if LM Studio is responding."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.api_base}/models")
                return response.status_code == 200
        except httpx.ConnectError:
            return False
        except Exception:
            return False

    def start_server(self):
        """Launches LM Studio locally in server mode using the `lms` CLI."""
        debug_print("[yellow]LM Studio server is not running.[/yellow]")
        debug_print("[dim]Starting server via `lms server start`...[/dim]")

        try:
            subprocess.Popen(
                ["lms", "server", "start"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
            )
            self.started_by_bringer = True
        except Exception as e:
            console.print(f"[bold red]Critical error launching lms:[/bold red] {e}")
            console.print("[red]LM Studio CLI tools are not installed or not in PATH.[/red]")
            sys.exit(1)

        debug_print("[cyan]Waiting for server to become ready[/cyan]", end="")
        for _ in range(30):
            if self.is_server_running():
                debug_print("\n[green]LM Studio server is online.[/green]")
                return
            time.sleep(1)
            if config.DEBUG_MODE:
                print(".", end="", flush=True)

        console.print("\n[bold red]Error: LM Studio server failed to start or timed out.[/bold red]")
        sys.exit(1)

    def shutdown(self):
        """Stops the LM Studio server only if this process started it."""
        if not self.started_by_bringer:
            return

        console.print("Stopping LM Studio server...")
        try:
            subprocess.run(
                ["lms", "server", "stop"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                check=False,
            )
            console.print("LM Studio server stopped.")
        except Exception:
            console.print("Warning: Failed to stop LM Studio server.")

    def stop_server(self):
        """Backward-compatible alias for centralized shutdown."""
        self.shutdown()

    def get_loaded_models(self) -> List[str]:
        """Queries the API to return a list of currently loaded model IDs."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.api_base}/models")
                if response.status_code == 200:
                    data = response.json()
                    loaded = [model.get("id", "") for model in data.get("data", [])]
                    return [model_id for model_id in loaded if model_id]
        except Exception as e:
            debug_print(f"[dim]Failed checking loaded models: {e}[/dim]")
        return []

    def is_model_ready_for_generation(self, model_name: str) -> bool:
        """
        Verifies the model is ready by sending a minimal test completion request.
        """
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.api_base}/chat/completions", json=payload)
                return response.status_code == 200
        except Exception:
            return False

    def _log_loaded_models(self, loaded_models: List[str]):
        if loaded_models:
            debug_print("[cyan]Loaded models detected:[/cyan]")
            for model_name in loaded_models:
                debug_print(model_name)
        else:
            debug_print("[dim]No loaded models detected.[/dim]")

    def _run_lms_command(self, command: List[str], check: bool = True) -> bool:
        try:
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                check=check,
            )
            return True
        except Exception as e:
            console.print(f"[bold red]Failed to execute {' '.join(command)}:[/bold red] {e}")
            return False

    def _wait_for_model_registration(self, desired_model: str, timeout_seconds: float) -> bool:
        deadline = time.time() + timeout_seconds

        while time.time() < deadline:
            loaded_models = self.get_loaded_models()
            if desired_model in loaded_models:
                return True
            time.sleep(self.model_poll_interval)

        return False

    def wait_for_model_ready(self, desired_model: str) -> bool:
        """
        Only probes readiness after the exact model is visible in /v1/models.
        """
        if desired_model not in self.get_loaded_models():
            return False

        debug_print("[cyan]Running readiness probe...[/cyan]")
        deadline = time.time() + self.readiness_timeout

        while time.time() < deadline:
            if self.is_model_ready_for_generation(desired_model):
                debug_print("[bold green]Model ready.[/bold green]")
                return True
            time.sleep(self.model_poll_interval)

        console.print("[bold red]Error: Model appeared but failed readiness check.[/bold red]")
        return False

    def unload_all_models(self) -> bool:
        """Unloads all currently loaded models using the LMS CLI to free VRAM."""
        debug_print("[cyan]Unloading currently loaded models...[/cyan]")

        if not self._run_lms_command(["lms", "unload", "--all"]):
            return False

        time.sleep(self.unload_wait_seconds)
        return True

    def load_model(self, desired_model: str) -> bool:
        """Ensures the exact desired model is loaded and ready."""
        debug_print("\n[cyan]Checking LM Studio server...[/cyan]")

        loaded_models = self.get_loaded_models()
        self._log_loaded_models(loaded_models)

        if desired_model in loaded_models:
            debug_print(f"[green]Desired model already loaded:[/green] {desired_model}")
            return self.wait_for_model_ready(desired_model)

        debug_print("[yellow]Desired model not loaded.[/yellow]")

        if loaded_models and not self.unload_all_models():
            return False

        debug_print(f"[cyan]Loading model {desired_model} using preset Bringer_RAG...[/cyan]")
        if not self._run_lms_command(
            ["lms", "load", desired_model, "--preset", "Bringer_RAG"],
            check=True,
        ):
            return False

        debug_print("[cyan]Waiting for model to register...[/cyan]")
        if not self._wait_for_model_registration(
            desired_model,
            timeout_seconds=self.model_load_timeout,
        ):
            console.print("\n[bold red]Error: Timed out waiting for model to appear in LM Studio.[/bold red]")
            return False

        debug_print("[green]Model detected.[/green]")
        return self.wait_for_model_ready(desired_model)

    def ensure_ready(self, selected_model: str):
        """Master orchestration: server check -> model check -> load logic."""
        if not self.is_server_running():
            self.start_server()
        else:
            debug_print("[dim]Server running.[/dim]")

        if not self.load_model(selected_model):
            sys.exit(1)


if __name__ == "__main__":
    console.print("\n[bold magenta]--- LM Studio Manager Test ---[/bold magenta]")

    manager = LMStudioManager()
    test_model = "qwen2.5-coder-7b-instruct"
    manager.ensure_ready(test_model)

    console.print("\n[bold green]Manager test complete.[/bold green]")
