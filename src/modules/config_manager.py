"""
Bringer RAG System — Configuration Manager

Centralized manager for models.json. Handles parsing, profile inheritance, 
and graceful fallbacks. Guarantees a valid configuration dictionary is 
always returned even if the user's config is broken or missing.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config
from src.modules.logging_utils import debug_print

CONFIG_FILE_PATH = config.PROJECT_ROOT / "models.json"

BUILT_IN_DEFAULTS = {
    "active_mode": "auto",
    "defaults": {
        "llm": {
            "n_ctx": 8192,
            "temperature": 0.1,
            "top_p": 0.90,
            "max_tokens": 1024,
            "repeat_penalty": 1.1,
            "n_gpu_layers": -1,  # -1 means all layers by default in llama-cpp if possible
        },
        "embedding": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 64,
            "dimensions": 384
        },
        "reranker": {
            "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "min_score": 0.4
        }
    },
    "profiles": {
        "low_power": {},
        "balanced": {},
        "high_performance": {}
    }
}

class ConfigManager:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or CONFIG_FILE_PATH
        self.raw_config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Loads models.json gracefully. Returns built-ins if missing/broken."""
        if not self.config_path.exists():
            debug_print("[yellow]models.json not found. Using built-in defaults.[/yellow]")
            return BUILT_IN_DEFAULTS.copy()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except json.JSONDecodeError as e:
            debug_print(f"[bold red]Failed to parse models.json:[/bold red] {e}")
            debug_print("[yellow]Falling back to built-in defaults.[/yellow]")
            return BUILT_IN_DEFAULTS.copy()
        except Exception as e:
            debug_print(f"[bold red]Unexpected error reading models.json:[/bold red] {e}")
            return BUILT_IN_DEFAULTS.copy()

    def save_config(self, config_dict: Dict[str, Any]):
        """Saves the given dictionary to models.json."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2)
            self.raw_config = config_dict
        except Exception as e:
            debug_print(f"[bold red]Failed to save models.json:[/bold red] {e}")

    def get_active_mode(self) -> str:
        """Returns the configured active mode ('auto', 'low', 'balanced', 'high')."""
        return self.raw_config.get("active_mode", "auto")

    def set_active_mode(self, mode: str):
        """Sets the active mode and saves."""
        valid_modes = ["auto", "low_power", "balanced", "high_performance"]
        if mode in valid_modes:
            self.raw_config["active_mode"] = mode
            self.save_config(self.raw_config)

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merges `update` into `base`."""
        merged = base.copy()
        for k, v in update.items():
            if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                merged[k] = self._deep_merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    def _resolve_profile(self, profile_name: str, visited: set = None) -> Dict[str, Any]:
        """Resolves a profile, applying inheritance from parent -> defaults -> built-ins."""
        if visited is None:
            visited = set()

        if profile_name in visited:
            debug_print(f"[red]Circular inheritance detected in profile: {profile_name}[/red]")
            return {}

        visited.add(profile_name)

        profiles = self.raw_config.get("profiles", {})
        
        # Start with built-in defaults
        resolved = self._deep_merge({}, BUILT_IN_DEFAULTS["defaults"])
        
        # Merge user defaults
        user_defaults = self.raw_config.get("defaults", {})
        resolved = self._deep_merge(resolved, user_defaults)

        profile_data = profiles.get(profile_name, {})
        
        # Handle inheritance
        parent_name = profile_data.get("inherits")
        if parent_name:
            parent_resolved = self._resolve_profile(parent_name, visited)
            resolved = self._deep_merge(resolved, parent_resolved)

        # Merge the profile's own specific overrides
        resolved = self._deep_merge(resolved, profile_data)
        
        # Remove the inherits key from final resolved config
        if "inherits" in resolved:
            del resolved["inherits"]

        return resolved

    def get_profile_config(self, profile_name: str) -> Dict[str, Any]:
        """Gets fully resolved configuration for a profile."""
        profiles = self.raw_config.get("profiles", {})
        # If the requested profile strictly doesn't exist, we try falling back to balanced
        if profile_name not in profiles and profile_name != "balanced":
            debug_print(f"[yellow]Profile '{profile_name}' not found. Falling back to 'balanced'.[/yellow]")
            return self._resolve_profile("balanced")
            
        return self._resolve_profile(profile_name)

    def set_profile_llm_path(self, profile_name: str, path: str):
        """Updates the LLM path for a given profile."""
        if "profiles" not in self.raw_config:
            self.raw_config["profiles"] = {}
        if profile_name not in self.raw_config["profiles"]:
            self.raw_config["profiles"][profile_name] = {}
        if "llm" not in self.raw_config["profiles"][profile_name]:
            self.raw_config["profiles"][profile_name]["llm"] = {}
            
        self.raw_config["profiles"][profile_name]["llm"]["path"] = path
        self.save_config(self.raw_config)

# Singleton instance
config_manager = ConfigManager()

def get_config_manager() -> ConfigManager:
    return config_manager
