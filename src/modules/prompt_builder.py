"""
Bringer RAG System — Prompt Builder

Transforms raw retrieved chunks and a user query into a highly structured,
OpenAI-compatible prompt payload ready for the LLM. 

Enforces strict token limits to prevent context window overflow (OOM or crashes),
and grounds the LLM via strict system instructions to minimize hallucinations.
"""

from typing import List, Dict, Any, Tuple
from rich.console import Console
import tiktoken

import sys
import os
# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config
from src.modules.logging_utils import debug_print

console = Console()

# We use cl100k_base to quickly approximate tokens, even if Qwen uses a slightly different BPE.
# It is fast and close enough for context safety limits.
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

class PromptBuilder:
    def __init__(self):
        """Initializes the prompt builder with context limits from config."""
        self.max_context_tokens = config.LLM_MAX_CONTEXT_TOKENS
        
        # Base system instruction used for all RAG queries
        self.system_prompt = (
            "You are a precise document extractor.\n\n"
            "INSTRUCTIONS:\n"
            "1. You MUST extract the answer strictly from the provided context.\n"
            "2. Do NOT invent facts or use outside knowledge.\n"
            "3. If the context is a strong match, return the exact sentence or exact lines that answer the question.\n"
            "4. If the context is only a moderate match, return the closest relevant sentence without adding new facts.\n"
            "5. If the answer is not explicitly or closely present, respond exactly: \"I could not find the exact answer in the documents.\"\n"
            "6. Do not combine unrelated chunks or pages."
        )

    def _estimate_tokens(self, text: str) -> int:
        """Fast approximation of token count."""
        return len(_TOKENIZER.encode(text))

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        confidence_mode: str = "high",
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        Constructs the final prompt array and enforces token safety limits.
        
        Args:
            query: The user's question.
            retrieved_chunks: List of chunk dictionaries from the Retriever.
            
        Returns:
            Tuple of (messages_payload, estimated_total_tokens)
        """
        # 1. Base tokens calculation
        system_tokens = self._estimate_tokens(self.system_prompt)
        if confidence_mode == "moderate":
            answer_instruction = (
                "Return the closest relevant sentence(s) from the context that answer the question. "
                "Stay grounded in the provided text and do not add new facts. "
                "If no relevant text exists, return exactly: "
                "\"I could not find the exact answer in the documents.\""
            )
        else:
            answer_instruction = (
                "Return the exact sentence(s) from the context that answer the question. "
                "If no exact answer exists in the context, return exactly: "
                "\"I could not find the exact answer in the documents.\""
            )

        query_format = f"\n\n---\n\nQUESTION:\n{query}\n\n{answer_instruction}"
        query_tokens = self._estimate_tokens(query_format)
        
        # 2. Add chunks until we hit the context safety limit
        # Leave a 500 token safety buffer for system framing and the expected output generation
        available_budget = self.max_context_tokens - system_tokens - query_tokens - config.LLM_MAX_TOKENS - 500
        
        context_text = "CONTEXT:\n"
        chunks_used = 0
        current_tokens = 0
        
        for k, chunk in enumerate(retrieved_chunks):
            # Format the chunk with clear source attribution
            source_file = chunk["metadata"].get("source_file", "Unknown")
            chunk_index = chunk["metadata"].get("chunk_index", 0)
            page_number = chunk["metadata"].get("page_number")
            chunk_content = chunk.get("content", "").strip()
            
            if page_number is not None:
                chunk_block = f"\n[Source: {source_file} | page {page_number}]\n{chunk_content}\n"
            else:
                chunk_block = f"\n[Source: {source_file}]\n{chunk_content}\n"
                
            chunk_tokens = self._estimate_tokens(chunk_block)
            
            # If adding this chunk exceeds our remaining budget, stop adding chunks
            if current_tokens + chunk_tokens > available_budget:
                if chunks_used == 0:
                    debug_print("[yellow]Warning: The first retrieved chunk is too large for the context window![/yellow]")
                else:
                    debug_print(f"[dim]Truncated prompt at {chunks_used} chunks to stay within {self.max_context_tokens} token limit.[/dim]")
                break
                
            context_text += chunk_block
            current_tokens += chunk_tokens
            chunks_used += 1

        total_prompt_estimate = system_tokens + current_tokens + query_tokens
        
        # 3. Finalize User Prompt Assembly
        user_prompt = f"{context_text}{query_format}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        debug_print(f"[dim]Context chunks injected: {chunks_used}[/dim]")
        debug_print(f"[dim]Prompt token estimate: ~{total_prompt_estimate}[/dim]")
        
        return messages, total_prompt_estimate


# Quick test trigger block (only runs if executed directly)
if __name__ == "__main__":
    import json
    from hybrid_retriever import HybridRetriever
    
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    else:
        test_query = "What is a wired LAN?"
        
    console.print(f"\n[bold magenta]--- Prompt Builder Test ---[/bold magenta]")
    console.print(f"Query: \"{test_query}\"")
    
    # Use real chunks from the Hybrid Retriever
    console.print(f"[dim]Retrieving real chunks for query...[/dim]")
    retriever = HybridRetriever()
    real_chunks = retriever.retrieve(test_query)
    
    if not real_chunks:
        console.print("[yellow]No chunks found in the database. Please ingest some documents first![/yellow]")
    else:
        builder = PromptBuilder()
        messages, token_count = builder.build_prompt(test_query, real_chunks)
        
        console.print("\n[bold green]Generated OpenAI-Compatible Payload:[/bold green]")
        print(json.dumps({"messages": messages}, indent=2, ensure_ascii=False))
