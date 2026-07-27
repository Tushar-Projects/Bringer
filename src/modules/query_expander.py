"""
Bringer RAG System — Query Expander

Uses the LLM to generate alternative phrasing for a user query.
Multiple queries increase the probability of matching the vocabulary used
in the retrieved documents, significantly improving recall.
"""

import json
import os
import sys
import time

from rich.console import Console

# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.modules.llm_client import LLMClient
from src.modules.logging_utils import debug_print

console = Console()

class QueryExpander:
    def __init__(self):
        self.llm_client = LLMClient()
        
        # System prompt engineered for fast, structured list outputs
        self.system_prompt = (
            "You are an AI assistant specialized in information retrieval. "
            "Your task is to generate exactly 3 alternative search queries based on the user's original query. "
            "These alternatives should use different synonyms, rephrase the concept, or expand acronyms to maximize the chance of finding relevant documents. "
            "Return ONLY a valid JSON array of strings, with no markdown formatting, no backticks, and no extra conversational text.\n"
            "Example output:\n[\"alternative 1\", \"alternative 2\", \"alternative 3\"]"
        )

    def expand_query(self, query: str) -> list[str]:
        """
        Generates alternative queries via the LLM API.
        
        Args:
            query: The original user query.
            
        Returns:
            A list containing the original query plus up to 3 generated alternatives.
        """
        t0 = time.perf_counter()
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Original query: {query}"}
        ]
        
        response_text = self.llm_client.generate(messages)
        t_gen = time.perf_counter() - t0
        
        queries = [query]  # Always include the exact original query first
        
        if not response_text:
            debug_print(f"[dim]Query expansion failed, returning original query ({t_gen*1000:.1f}ms).[/dim]")
            return queries
            
        try:
            # Clean up the output in case the LLM ignored rules and wrapped in markdown
            cleaned_text = response_text.strip()
            cleaned_text = cleaned_text.removeprefix("```json")
            cleaned_text = cleaned_text.removeprefix("```")
            cleaned_text = cleaned_text.removesuffix("```")
                
            alternatives = json.loads(cleaned_text.strip())
            
            if isinstance(alternatives, list):
                # Filter out pure duplicates and empty strings
                for alt in alternatives:
                    valid_alt = str(alt).strip()
                    if valid_alt and valid_alt.lower() != query.lower():
                        queries.append(valid_alt)
                        
            debug_print(f"[dim]Query expansion generated {len(queries)-1} variations ({t_gen*1000:.1f}ms).[/dim]")
            
        except json.JSONDecodeError:
            debug_print(f"[yellow]Failed to parse query expansion JSON. Raw output: {response_text}[/yellow]")
            debug_print(f"[dim]Returning original query only ({t_gen*1000:.1f}ms).[/dim]")
            
        return queries

# Quick test block
if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    else:
        test_query = "What is a wired LAN?"
        
    console.print("\n[bold magenta]--- Query Expander Test ---[/bold magenta]")
    console.print(f"Original: '{test_query}'")
    
    expander = QueryExpander()
    expanded = expander.expand_query(test_query)
    
    console.print("\n[bold green]Final Query Target List:[/bold green]")
    for i, q in enumerate(expanded):
        console.print(f"{i+1}. {q}")
