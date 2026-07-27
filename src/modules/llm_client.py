"""
Bringer RAG System — LLM Client

Client for interacting with the LlamaManager.
Supports generation and streaming of LLM responses based on the active profile.
"""

import os
import sys
from collections.abc import Generator

from rich.console import Console

# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.modules.config_manager import get_config_manager
from src.modules.hardware_detector import HardwareDetector
from src.modules.llama_manager import get_llama_manager

console = Console()

class LLMClient:
    def __init__(self):
        """Initializes the LLM client."""
        self.llama_manager = get_llama_manager()
        self.config_manager = get_config_manager()
        self.hardware_detector = HardwareDetector()

    def get_current_profile_name(self) -> str:
        """Determines the active profile name."""
        active_mode = self.config_manager.get_active_mode()
        if active_mode == "auto":
            return self.hardware_detector.select_profile()
        return active_mode

    def generate(self, messages: list[dict[str, str]]) -> str | None:
        """
        Generates a complete response string.
        """
        profile_name = self.get_current_profile_name()
        return self.llama_manager.generate(messages, profile_name)

    def stream(self, messages: list[dict[str, str]]) -> Generator[str, None, None]:
        """
        Streams the response back token by token.
        """
        profile_name = self.get_current_profile_name()
        yield from self.llama_manager.stream(messages, profile_name)

# Quick test trigger block (only runs if executed directly)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_prompt = " ".join(sys.argv[1:])
    else:
        test_prompt = "Explain RAG (Retrieval-Augmented Generation) in exactly two short sentences."
        
    console.print("\n[bold magenta]--- Llama Manager Test ---[/bold magenta]")
    console.print(f"[cyan]Prompt:[/cyan] {test_prompt}\n")
    
    client = LLMClient()
    messages = [
        {"role": "system", "content": "You are a helpful AI engineer."},
        {"role": "user", "content": test_prompt}
    ]
    
    console.print("[dim]Testing Streaming Response...[/dim]")
    
    print("Response: ", end="", flush=True)
    for chunk in client.stream(messages):
        print(chunk, end="", flush=True)
    print("\n\n[bold green]Test Complete![/bold green]")
