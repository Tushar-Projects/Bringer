"""
Bringer RAG System — Hardware Detector

Analyzes system resources (GPU availability, battery power state)
to dynamically select the most appropriate LLM tier for local inference.
"""

import subprocess
from typing import Any

import psutil
from rich.console import Console

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

import os
import sys

# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

console = Console()

class HardwareDetector:
    def __init__(self):
        """Initializes the detector."""
        if config.FORCE_CPU:
            self.gpu_available = False
            self.gpu_name = "Forced CPU"
        else:
            self.gpu_available, self.gpu_name = self.detect_gpu()

    def _detect_gpu_via_torch(self) -> tuple[bool, str]:
        """Checks whether CUDA is available via PyTorch."""
        if torch is None:
            return False, "N/A"

        try:
            if torch.cuda.is_available():
                return True, torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001, S110
            pass

        return False, "N/A"

    def _detect_gpu_via_nvidia_smi(self) -> tuple[bool, str]:
        """Checks whether an NVIDIA GPU is present using nvidia-smi."""
        query_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if query_result.returncode == 0:
            gpu_name = query_result.stdout.strip().splitlines()[0] if query_result.stdout.strip() else "NVIDIA GPU available"
            return True, gpu_name

        probe_result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe_result.returncode == 0:
            return True, "NVIDIA GPU available"

        return False, "N/A"

    def detect_gpu(self) -> tuple[bool, str]:
        """
        Detects GPU availability using PyTorch first, then nvidia-smi as a fallback.
        """
        gpu_available, gpu_name = self._detect_gpu_via_torch()
        if gpu_available:
            return gpu_available, gpu_name

        return self._detect_gpu_via_nvidia_smi()

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

    def is_power_saver_enabled(self) -> bool:
        """
        Attempts to detect if OS power saver is enabled.
        On Windows, we can use ctypes. For other OS, we default to False unless battery < 20%.
        """
        import platform
        if platform.system() == "Windows":
            try:
                import ctypes
                from ctypes import wintypes
                
                class SYSTEM_POWER_STATUS(ctypes.Structure):
                    _fields_ = [
                        ('ACLineStatus', wintypes.BYTE),
                        ('BatteryFlag', wintypes.BYTE),
                        ('BatteryLifePercent', wintypes.BYTE),
                        ('SystemStatusFlag', wintypes.BYTE),
                        ('BatteryLifeTime', wintypes.DWORD),
                        ('BatteryFullLifeTime', wintypes.DWORD),
                    ]
                
                status = SYSTEM_POWER_STATUS()
                if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
                    # SystemStatusFlag bit 0 indicates Battery Saver is on (value 1)
                    return status.SystemStatusFlag == 1
            except Exception:  # noqa: BLE001, S110
                pass
                
        # Fallback for non-Windows or if ctypes fails: 
        # assume power saver if on battery and < 20%
        battery = psutil.sensors_battery()
        return bool(battery and not battery.power_plugged and battery.percent < 20)

    def detect_hardware(self) -> dict[str, Any]:
        """
        Returns a dictionary of hardware states.
        """
        plugged_in = self.is_plugged_in()
        power_saver = self.is_power_saver_enabled()
        
        status = {
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name if self.gpu_available else "N/A",
            "plugged_in": plugged_in,
            "power_saver": power_saver
        }
        return status

    def select_profile(self) -> str:
        """
        Selects the best power profile based on the current hardware state.
        """
        plugged_in = self.is_plugged_in()
        power_saver = self.is_power_saver_enabled()
        
        if power_saver:
            return "low_power"
        elif not plugged_in:
            return "balanced"
        else:
            return "high_performance"

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
    console.print(f"[cyan]Power Saver:[/cyan] {'Enabled' if hw_state['power_saver'] else 'Disabled'}")
    
    selected_profile = detector.select_profile()
    console.print(f"\n[bold green]Selected Profile:[/bold green] {selected_profile}")
