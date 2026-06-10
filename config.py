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
DEBUG_MODE = False

DEVICE = get_device()

# =============================================================================
# Model configuration is now managed centrally via models.json
# See src/modules/config_manager.py
# =============================================================================
LLM_TIMEOUT = 120                        # Seconds before request timeout

# =============================================================================
# Reranker / Retrieval Thresholds
# =============================================================================
RERANK_MIN_SCORE = 0.4
STRICT_RERANK_MIN_SCORE = 0.7
RELAXED_RERANK_MIN_SCORE = 0.4

SEMANTIC_TOP_K = 3
MIN_SIMILARITY_SCORE = 0.45
STRICT_MIN_SIMILARITY_SCORE = 0.6
RELAXED_MIN_SIMILARITY_SCORE = 0.3
BM25_TOP_K = 20
HYBRID_TOP_K = 3
FINAL_TOP_K = 3
STRICT_FINAL_TOP_K = 3
RELAXED_FINAL_TOP_K = 5

SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# =============================================================================
# Chunking
# =============================================================================
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

# =============================================================================
# Vector Database
# =============================================================================
CHROMA_COLLECTION_NAME = "bringer_documents"

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
