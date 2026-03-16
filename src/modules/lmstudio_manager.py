"""
Bringer RAG System — LM Studio Manager

Responsible for auto-starting the local LM Studio server if it is offline.
Checks currently loaded models via the API. Unloads incorrect models to save VRAM,
and ensures the correct language model (selected by hardware detection)
is loaded before generation begins.
"""

import time
import httpx
import subprocess
from typing import List
from rich.console import Console

import sys
import os
# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

console = Console()

class LMStudioManager:
    def __init__(self):
        self.api_base = config.LLM_API_BASE
        self.timeout = 5.0  # Quick timeout for health checks

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
        console.print("[yellow]LM Studio server is not running.[/yellow]")
        console.print("[dim]Starting server via `lms server start`...[/dim]")
        
        try:
            subprocess.Popen(
                ["lms", "server", "start"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                shell=True
            )
        except Exception as e:
            console.print(f"[bold red]Critical Error launching lms:[/bold red] {e}")
            console.print("[red]LM Studio CLI tools are not installed or not in PATH.[/red]")
            sys.exit(1)
            
        # Poll until the server stands up
        console.print("[cyan]Waiting for server to become ready[/cyan]", end="")
        max_retries = 30
        for _ in range(max_retries):
            if self.is_server_running():
                console.print("\n[green]LM Studio Server is online.[/green]")
                return
            time.sleep(1)
            print(".", end="", flush=True)
            
        console.print("\n[bold red]Error: LM Studio server failed to start or timed out.[/bold red]")
        sys.exit(1)

    def get_loaded_models(self) -> List[str]:
        """Queries the API to return a list of currently loaded model IDs."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.api_base}/models")
                if response.status_code == 200:
                    data = response.json()
                    loaded = [model.get("id", "") for model in data.get("data", [])]
                    return [m for m in loaded if m]
        except Exception as e:
            console.print(f"[dim]Failed checking loaded models: {e}[/dim]")
        return []
        
    def is_model_ready_for_generation(self, model_name: str) -> bool:
        """
        Actively verifies if the model is ready by sending a minimal test completion request.
        This proves the model is fully loaded into VRAM and accepting generation commands.
        """
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.api_base}/chat/completions", json=payload)
                return response.status_code == 200
        except Exception:
            return False

    def unload_all_models(self):
        """Unloads all currently loaded models using the LMS CLI to free VRAM."""
        console.print("[dim]Unloading currently loaded models from VRAM...[/dim]")
        try:
            subprocess.run(
                ["lms", "unload", "--all"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True
            )
            # Short wait for VRAM to clear
            time.sleep(2)
        except Exception as e:
            console.print(f"[dim]Warning: Failed to execute unload command: {e}[/dim]")

    def load_model(self, desired_model: str):
        """Intelligently loads the desired model, unloading others if needed."""
        console.print("\n[dim]Checking LM Studio server models...[/dim]")
        
        loaded_models = self.get_loaded_models()
        
        if not loaded_models:
            console.print("[dim]No model loaded.[/dim]")
            
        # 1. Exact model is already loaded
        if any(desired_model.lower() in m.lower() for m in loaded_models):
            console.print(f"[green]Detected loaded model:[/green] {desired_model}")
            console.print("[dim]Model already loaded — skipping load.[/dim]")
            return
            
        # 2. Other models are loaded — need to unload first to prevent VRAM overflow
        if loaded_models and not any(desired_model.lower() in m.lower() for m in loaded_models):
            console.print(f"[yellow]Different models currently loaded: {', '.join(loaded_models)}[/yellow]")
            self.unload_all_models()
            
        # 3. Load the target model
        console.print(f"\n[cyan]Loading model:[/cyan] {desired_model} using preset Bringer_RAG...")
        console.print("[dim](Please wait, this may take a moment)[/dim]")
        
        try:
            subprocess.Popen(
                ["lms", "load", desired_model, "--preset", "Bringer_RAG", "--gpu", "max"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True
            )
        except Exception as e:
            console.print(f"[bold red]Failed to execute model load sequence: {e}[/bold red]")
            return
            
        # 4. Poll until the model appears in /v1/models
        console.print("[cyan]Waiting for model to load[/cyan]", end="")
        max_load_seconds = 90
        t0 = time.time()
        
        model_loaded = False
        while time.time() - t0 < max_load_seconds:
            current_models = self.get_loaded_models()
            if any(desired_model.lower() in m.lower() for m in current_models):
                model_loaded = True
                break
            time.sleep(2)
            print(".", end="", flush=True)
            
        if not model_loaded:
            console.print("\n[bold red]Error: Timed out waiting for model to appear in LM Studio.[/bold red]")
            return

        console.print("\n[green]Model detected in /v1/models.[/green]")
        console.print("[dim]Running readiness test...[/dim]")

        # 5. Perform Readiness Check
        t1 = time.time()
        model_ready = False
        # Give it a short window after appearing to be fully ready for generation
        while time.time() - t1 < 30:
            if self.is_model_ready_for_generation(desired_model):
                model_ready = True
                break
            time.sleep(2)
            
        if model_ready:
            console.print("[bold green]Model ready.[/bold green]")
        else:
            console.print("[bold red]Error: Model appeared but failed readiness check.[/bold red]")

    def ensure_ready(self, selected_model: str):
        """Master orchestration: server check -> model check -> load logic."""
        if not self.is_server_running():
            self.start_server()
        else:
            console.print("[dim]Server running.[/dim]")
            
        self.load_model(selected_model)


# Quick standalone test
if __name__ == "__main__":
    console.print("\n[bold magenta]--- LM Studio Manager Test ---[/bold magenta]")
    
    manager = LMStudioManager()
    
    # Simulate hardware detection selecting a default
    test_model = "qwen2.5-7b-instruct"
    manager.ensure_ready(test_model)
    
    console.print("\n[bold green]Manager test complete.[/bold green]")
