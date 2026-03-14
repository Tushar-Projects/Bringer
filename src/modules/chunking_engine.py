"""
Bringer RAG System — Chunking Engine

Responsible for taking raw document text and splitting it into clean, 
token-aware semantic chunks.

Uses tiktoken for tokenization (cl100k_base, the standard for OpenAI models)
and LangChain's RecursiveCharacterTextSplitter for syntax-aware splitting
that respects sentence boundaries.
"""

import re
import time
from typing import List, Dict, Any
from pathlib import Path
from rich.console import Console

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
import os
# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

console = Console()

# =============================================================================
# Global Initializations (Performance Optimization)
# Initialize these once on module load, rather than per document
# =============================================================================
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

_TEXT_SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=config.CHUNK_SIZE_TOKENS,
    chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
    separators=config.CHUNK_SEPARATORS,
    keep_separator=True,
    is_separator_regex=False
)

class ChunkingEngine:
    def __init__(self):
        """Initialize the Token-Aware Chunking Engine."""
        self.chunk_size = config.CHUNK_SIZE_TOKENS
        self.chunk_overlap = config.CHUNK_OVERLAP_TOKENS
        self.tokenizer = _TOKENIZER
        self.text_splitter = _TEXT_SPLITTER

    def _normalize_text(self, text: str) -> str:
        """
        Cleans and normalizes text prior to chunking to improve embedding quality.
        """
        if not text:
            return ""
            
        # Remove zero-width characters and unusual spaces
        text = text.replace("\u200b", "").replace("\u00a0", " ")
        
        # Collapse multiple spaces into one
        text = re.sub(r' {2,}', ' ', text)
        
        # Collapse 3+ newlines into 2 (preserve paragraph breaks, eliminate huge gaps)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix PDF mid-sentence line breaks (heuristic: uncapitalized word on next line)
        # Note: We keep this simple to avoid breaking genuine list items.
        text = re.sub(r'(?<=[a-zA-Z,])\n(?=[a-z])', ' ', text)
        
        return text.strip()

    def chunk_document(self, document_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits a single document's text into token-aware chunks, inheriting metadata.
        
        Args:
            document_dict: Dictionary containing 'content' and 'metadata'.
            
        Returns:
            A list of dictionary chunks with 'content' and appended 'metadata'.
        """
        text = document_dict.get("content", "")
        base_meta = document_dict.get("metadata", {})
        
        if not text or not text.strip():
            console.print(f"[yellow]Skipping empty document: {base_meta.get('source_file', 'Unknown')}[/yellow]")
            return []

        # 1. Normalize text
        t0 = time.perf_counter()
        clean_text = self._normalize_text(text)
        t_norm = time.perf_counter() - t0
        
        # 2. Split recursively (token-aware)
        t0 = time.perf_counter()
        raw_chunks = self.text_splitter.split_text(clean_text)
        t_split = time.perf_counter() - t0
        total_chunks = len(raw_chunks)
        
        # 3. Assemble chunks with metadata
        t0 = time.perf_counter()
        structured_chunks = []
        
        for idx, chunk_text in enumerate(raw_chunks):
            # 4. Final validation: Skip extremely tiny chunks unless it's the only one
            chunk_tokens = len(self.tokenizer.encode(chunk_text))
            if chunk_tokens < 10 and total_chunks > 1:
                continue
                
            source_file_name = base_meta.get('source_file', 'unknown')
            file_type = base_meta.get('file_type', 'txt')
            page_number = base_meta.get('page_number')
            
            # Create a safe chunk ID from the filename and page number
            safe_name = source_file_name.replace(' ', '_').replace('.', '_')
            if page_number is not None:
                chunk_id = f"{safe_name}_p{page_number}_{idx}"
            else:
                chunk_id = f"{safe_name}_{idx}"
            
            # Inherit base metadata and append chunk-specific data
            meta = base_meta.copy()
            meta.update({
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "token_count": chunk_tokens
            })
            
            chunk_dict = {
                "chunk_id": chunk_id,
                "content": chunk_text,
                "metadata": meta
            }
            structured_chunks.append(chunk_dict)
            
        t_meta = time.perf_counter() - t0
        
        if structured_chunks:
            avg_tokens = sum(c['metadata']['token_count'] for c in structured_chunks) / len(structured_chunks)
            console.print(f"Generated {len(structured_chunks)} chunks for [cyan]{base_meta.get('source_file', 'Unknown')}[/cyan] (avg {avg_tokens:.1f} tokens)")
            console.print(f"  [dim]Timing: Norm {t_norm*1000:.1f}ms | Split {t_split*1000:.1f}ms | Meta/TokenCount {t_meta*1000:.1f}ms[/dim]")
        
        return structured_chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a batch of documents.
        
        Args:
            documents: List of dicts, each containing 'content' and 'metadata'.
                       
        Returns:
            A flattened list of all generated chunks.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            
        console.print(f"[green]Total chunks generated across {len(documents)} documents: {len(all_chunks)}[/green]")
        return all_chunks


# Quick test trigger block (only runs if executed directly)
if __name__ == "__main__":
    from document_loader import DocumentLoader
    import sys
    import json
    
    if len(sys.argv) > 1:
        test_path_str = sys.argv[1]
        test_path = Path(test_path_str)
        loader = DocumentLoader()
        engine = ChunkingEngine()
        
        console.print(f"\n[bold magenta]--- Chunking Test ---[/bold magenta]")
        console.print(f"1. Loading: {test_path.name}")
        pages = loader.load_document(test_path)
        
        if pages:
            console.print("2. Chunking...")
            chunks = engine.chunk_documents(pages)
            
            if chunks:
                console.print("\n[bold green]Success![/bold green] Showing first chunk:")
                # Print just the first chunk formatted nicely
                print(json.dumps(chunks[0], indent=2, ensure_ascii=False))
                
                if len(chunks) > 1:
                    console.print(f"\n[bold green]... and {len(chunks)-1} more chunks.[/bold green]")
        else:
            console.print("[red]Failed to load text for testing.[/red]")
    else:
        print("Usage: python src/modules/chunking_engine.py <path_to_file>")
