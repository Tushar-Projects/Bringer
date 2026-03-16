"""
Bringer RAG System — LLM Client

HTTP client for interacting with the LM Studio local inference API.
Follows the OpenAI Chat Completions API format and supports streaming.
"""

import json
import httpx
from typing import List, Dict, Any, Generator, Optional
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

import sys
import os
# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

console = Console()

class LLMClient:
    def __init__(self):
        """Initializes the LLM client using configurations from config.py."""
        self.api_url = config.LLM_API_CHAT_ENDPOINT
        self.model_name = config.LLM_MODEL_NAME
        self.temperature = config.LLM_TEMPERATURE
        self.top_p = getattr(config, 'LLM_TOP_P', 0.9)
        self.max_tokens = config.LLM_MAX_TOKENS
        self.timeout = config.LLM_TIMEOUT

    def _build_payload(self, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
        """Constructs the standard OpenAI-compatible payload."""
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": stream
        }

    def generate(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Sends a blocking request to LM Studio and returns the full response string.
        
        Args:
            messages: List of message dicts [{"role": "system"/"user", "content": "..."}]
            
        Returns:
            The generated text, or None if the request fails.
        """
        payload = self._build_payload(messages, stream=False)
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.api_url, json=payload)
                response.raise_for_status()
                
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                
                console.print("[red]Unexpected response format from LM Studio.[/red]")
                return None
                
        except httpx.ConnectError:
            console.print("[bold red]Error: Could not connect to LM Studio.[/bold red]")
            console.print(f"[yellow]Ensure LM Studio is running and the local server is started at {config.LLM_API_BASE}[/yellow]")
            return None
        except httpx.TimeoutException:
            console.print(f"[bold red]Error: Request timed out after {self.timeout}s.[/bold red]")
            return None
        except httpx.HTTPStatusError as e:
            console.print(f"[bold red]HTTP {e.response.status_code} Error from LM Studio:[/bold red] {e.response.text}")
            return None
        except Exception as e:
            console.print(f"[bold red]Unexpected error calling LM Studio:[/bold red] {str(e)}")
            return None

    def stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
        Streams the response back from LM Studio token by token using SSE.
        
        Args:
            messages: List of message dicts [{"role": "system"/"user", "content": "..."}]
            
        Yields:
            Tokens/string fragments as they are generated.
        """
        payload = self._build_payload(messages, stream=True)
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", self.api_url, json=payload) as response:
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                            
                        data_str = line[6:]  # Strip 'data: ' prefix
                        
                        if data_str.strip() == "[DONE]":
                            break
                            
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.ConnectError:
            yield "\n[Error: Could not connect to LM Studio. Ensure the local server is running.]"
        except httpx.TimeoutException:
            yield "\n[Error: LLM request timed out. The model may be under heavy load or processing a massive context window.]"
        except httpx.ReadError:
            yield "\n[Error: The connection was closed unexpectedly by the LM Studio server.]"
        except Exception as e:
            yield f"\n[Streaming error: {str(e)}]"


# Quick test trigger block (only runs if executed directly)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_prompt = " ".join(sys.argv[1:])
    else:
        test_prompt = "Explain RAG (Retrieval-Augmented Generation) in exactly two short sentences."
        
    console.print(f"\n[bold magenta]--- LM Studio API Test ---[/bold magenta]")
    console.print(f"[cyan]Endpoint:[/cyan] {config.LLM_API_CHAT_ENDPOINT}")
    console.print(f"[cyan]Prompt:[/cyan] {test_prompt}\n")
    
    client = LLMClient()
    messages = [
        {"role": "system", "content": "You are a helpful AI engineer."},
        {"role": "user", "content": test_prompt}
    ]
    
    console.print("[dim]Testing Streaming Response...[/dim]")
    
    # We use Rich's Live view to stream Markdown nicely (or just print chunks)
    # Simple print implementation for testing
    import sys as sys_lib
    
    print("Response: ", end="", flush=True)
    for chunk in client.stream(messages):
        print(chunk, end="", flush=True)
    print("\n\n[bold green]Test Complete![/bold green]")
