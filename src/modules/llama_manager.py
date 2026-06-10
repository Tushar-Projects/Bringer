"""
Bringer RAG System - Llama-cpp Manager

Replaces LM Studio. Handles in-process loading, unloading, and streaming 
inference of llama-cpp GGUF models based on the centralized configuration.
"""

import sys
import os
import gc
from typing import Optional, Dict, Any, Generator, List

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config
from src.modules.logging_utils import debug_print
from src.modules.config_manager import get_config_manager

from rich.console import Console
console = Console()

class LlamaManager:
    def __init__(self):
        self.llm = None
        self.current_model_path = None
        self.config_manager = get_config_manager()

    def _unload_model(self):
        """Unloads the current model and frees memory cleanly."""
        if self.llm is not None:
            debug_print("[dim]Unloading current LLM from memory...[/dim]")
            del self.llm
            self.llm = None
            self.current_model_path = None
            
            # Force garbage collection to free RAM/VRAM
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            debug_print("[green]LLM memory freed.[/green]")

    def load_model(self, profile_name: str) -> bool:
        """Loads a model based on the specified profile name."""
        try:
            from llama_cpp import Llama
        except ImportError:
            console.print("[bold red]llama-cpp-python is not installed. Please run `pip install llama-cpp-python`.[/bold red]")
            return False

        profile_config = self.config_manager.get_profile_config(profile_name)
        llm_config = profile_config.get("llm", {})
        
        model_path = llm_config.get("path")
        if not model_path:
            debug_print(f"[yellow]Warning: No LLM path configured for profile '{profile_name}'.[/yellow]")
            return False
            
        absolute_path = os.path.abspath(model_path)
        if not os.path.exists(absolute_path):
            console.print(f"[bold red]Error: Model file not found at {absolute_path}[/bold red]")
            return False

        if self.current_model_path == absolute_path and self.llm is not None:
            debug_print(f"[dim]Model {absolute_path} is already loaded.[/dim]")
            return True

        self._unload_model()
        
        debug_print(f"[dim]Loading model: {absolute_path}...[/dim]")
        
        n_ctx = llm_config.get("n_ctx", 8192)
        n_gpu_layers = llm_config.get("n_gpu_layers", -1)
        n_batch = llm_config.get("batch_size", 512)

        try:
            self.llm = Llama(
                model_path=absolute_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_batch=n_batch,
                verbose=config.DEBUG_MODE
            )
            self.current_model_path = absolute_path
            debug_print("[green]Model loaded successfully.[/green]")
            return True
        except Exception as e:
            console.print(f"[bold red]Failed to load model:[/bold red] {e}")
            return False

    def is_model_loaded(self) -> bool:
        return self.llm is not None
        
    def generate(self, messages: List[Dict[str, str]], profile_name: str) -> Optional[str]:
        """Generates a complete response."""
        if not self.is_model_loaded():
            if not self.load_model(profile_name):
                return None
                
        profile_config = self.config_manager.get_profile_config(profile_name)
        llm_config = profile_config.get("llm", {})
        
        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=llm_config.get("temperature", 0.7),
                top_p=llm_config.get("top_p", 0.95),
                max_tokens=llm_config.get("max_tokens", 2048),
                repeat_penalty=llm_config.get("repeat_penalty", 1.1),
                stream=False
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            console.print(f"[bold red]Inference error:[/bold red] {e}")
            return None

    def stream(self, messages: List[Dict[str, str]], profile_name: str) -> Generator[str, None, None]:
        """Streams the response back token by token."""
        if not self.is_model_loaded():
            if not self.load_model(profile_name):
                yield "\n[Error: Could not load model for inference.]"
                return

        profile_config = self.config_manager.get_profile_config(profile_name)
        llm_config = profile_config.get("llm", {})
        
        try:
            response_generator = self.llm.create_chat_completion(
                messages=messages,
                temperature=llm_config.get("temperature", 0.7),
                top_p=llm_config.get("top_p", 0.95),
                max_tokens=llm_config.get("max_tokens", 2048),
                repeat_penalty=llm_config.get("repeat_penalty", 1.1),
                stream=True
            )
            for chunk in response_generator:
                if not isinstance(chunk, dict):
                    continue
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                if "content" in delta:
                    yield delta["content"]
        except Exception as e:
            yield f"\n[Streaming error: {e}]"
            
    def shutdown(self):
        """Cleans up the model before exit."""
        self._unload_model()

# Singleton for application lifetime
_llama_manager_instance = None

def get_llama_manager() -> LlamaManager:
    global _llama_manager_instance
    if _llama_manager_instance is None:
        _llama_manager_instance = LlamaManager()
    return _llama_manager_instance
