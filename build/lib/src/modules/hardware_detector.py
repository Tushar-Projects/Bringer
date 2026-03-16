"""
Bringer RAG System — Hardware Detector

Analyzes system resources (GPU availability, battery power state)
to dynamically select the most appropriate LLM tier for local inference.
"""

import psutil
import torch
from typing import Dict, Any, Tuple
from rich.console import Console

import sys
import os
# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

console = Console()

class HardwareDetector:
    def __init__(self):
        """Initializes the detector."""
        self.gpu_available = torch.cuda.is_available()
        # If FORCE_CPU is set in config, simulate no GPU
        if config.FORCE_CPU:
            self.gpu_available = False

    def is_plugged_in(self) -> bool:
        """
        Checks if the device is plugged into AC power.
        Defaults to True if battery information is unavailable (e.g., desktops).
        """
        battery = psutil.sensors_battery()
        if battery is None:
            # Desktop PCs or systems without battery sensors are considered "plugged in"
            return True
        return battery.power_plugged

    def detect_hardware(self) -> Dict[str, Any]:
        """
        Returns a dictionary of hardware states.
        """
        plugged_in = self.is_plugged_in()
        
        status = {
            "gpu_available": self.gpu_available,
            "gpu_name": torch.cuda.get_device_name(0) if self.gpu_available else "N/A",
            "plugged_in": plugged_in
        }
        return status

    def select_model(self) -> str:
        """
        Selects the best model based on the current hardware state.
        
        Logic:
        - GPU + AC Power: Large Model
        - GPU + Battery: Medium Model
        - CPU only: Small Model
        """
        if self.gpu_available:
            if self.is_plugged_in():
                return config.LLM_MODEL_LARGE
            else:
                return config.LLM_MODEL_MEDIUM
        else:
            return config.LLM_MODEL_SMALL


# Quick test trigger
if __name__ == "__main__":
    console.print("\n[bold magenta]--- Hardware Detector Test ---[/bold magenta]")
    
    detector = HardwareDetector()
    hw_state = detector.detect_hardware()
    
    console.print(f"[cyan]GPU Detected:[/cyan] {hw_state['gpu_available']}", end="")
    if hw_state['gpu_available']:
        console.print(f" ({hw_state['gpu_name']})")
    else:
        console.print()
        
    console.print(f"[cyan]Power state:[/cyan] {'Plugged in' if hw_state['plugged_in'] else 'On battery'}")
    
    selected_model = detector.select_model()
    console.print(f"\n[bold green]Selected Model:[/bold green] {selected_model}")
