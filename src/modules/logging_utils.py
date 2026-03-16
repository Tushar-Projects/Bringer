"""
Runtime logging helpers for clean CLI output and optional debug verbosity.
"""

import logging
import os
import warnings

from rich.console import Console

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

console = Console()


def is_debug_mode() -> bool:
    return bool(getattr(config, "DEBUG_MODE", False))


def debug_print(*args, **kwargs):
    if is_debug_mode():
        console.print(*args, **kwargs)


def configure_runtime_logging(debug_mode: bool):
    config.DEBUG_MODE = debug_mode
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    third_party_level = logging.DEBUG if debug_mode else logging.ERROR
    for logger_name in (
        "transformers",
        "sentence_transformers",
        "huggingface_hub",
        "chromadb",
        "watchdog",
    ):
        logging.getLogger(logger_name).setLevel(third_party_level)

    if not debug_mode:
        warnings.filterwarnings("ignore")
