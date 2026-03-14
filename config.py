"""
Bringer RAG System — Central Configuration

All configurable settings are defined here.
Modify these values to tune the system for your hardware and use case.
"""

from pathlib import Path
import torch

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
DOCUMENTS_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# =============================================================================
# Device Selection (GPU with automatic CPU fallback)
# =============================================================================
def get_device() -> str:
    """
    Returns 'cuda' if a GPU is available, otherwise 'cpu'.
    For RTX 4070 Laptop (8 GB VRAM), we run embeddings and reranker on GPU
    alongside the LLM in LM Studio. If VRAM pressure occurs, set
    FORCE_CPU=True below to force CPU mode for embeddings/reranking.
    """
    if FORCE_CPU:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

FORCE_CPU = False  # Set True to force embeddings & reranker onto CPU

DEVICE = get_device()

# =============================================================================
# Embedding Model
# =============================================================================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64           # Chunks per batch for embedding generation
EMBEDDING_DIMENSIONS = 384          # Output dimensions of the embedding model

# =============================================================================
# Reranker Model
# =============================================================================
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K = 5                  # Number of results after reranking

# =============================================================================
# Chunking
# =============================================================================
CHUNK_SIZE_TOKENS = 600             # Target chunk size in tokens
CHUNK_OVERLAP_TOKENS = 80           # Overlap between consecutive chunks (tokens)
# Recursive splitting hierarchy (preserves paragraphs and sentences before words)
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

# =============================================================================
# Retrieval
# =============================================================================
SEMANTIC_TOP_K = 5                  # Default top-k results from vector search
MIN_SIMILARITY_SCORE = 0.55         # Minimum score threshold (0-1) to keep a chunk
BM25_TOP_K = 20                     # Top-k results from BM25 keyword search
HYBRID_TOP_K = 5                    # Merged results before reranking
FINAL_TOP_K = 5                     # Results sent to LLM after reranking

SEMANTIC_WEIGHT = 0.7               # Weight given to vector similarity
KEYWORD_WEIGHT = 0.3                # Weight given to BM25 lexical matches

# =============================================================================
# LM Studio / LLM
# =============================================================================
LLM_API_BASE = "http://localhost:1234/v1"
LLM_API_CHAT_ENDPOINT = f"{LLM_API_BASE}/chat/completions"
LLM_MODEL_NAME = "qwen2.5-7b-instruct"  # Must match model loaded in LM Studio
LLM_TEMPERATURE = 0.1                    # Low temp for factual RAG answers
LLM_MAX_TOKENS = 1024                    # Max output tokens per response
LLM_MAX_CONTEXT_TOKENS = 8192            # Max total prompt size context window
LLM_TIMEOUT = 120                        # Seconds before request timeout

# =============================================================================
# File Watcher
# =============================================================================
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md"}
WATCH_DEBOUNCE_SECONDS = 2          # Wait time before indexing after file event

# =============================================================================
# ChromaDB
# =============================================================================
CHROMA_COLLECTION_NAME = "bringer_documents"

# =============================================================================
# Logging
# =============================================================================
LOG_FILE = LOGS_DIR / "bringer.log"
LOG_LEVEL = "INFO"
