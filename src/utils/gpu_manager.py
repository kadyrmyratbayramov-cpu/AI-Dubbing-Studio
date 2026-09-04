"""GPU/VRAM management helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DeviceInfo:
    device: str
    available: bool
    total_vram_mb: int
    allocated_vram_mb: int


class GPUManager:
    """Simple VRAM-aware device manager."""

    def __init__(self, preferred_device: str = "auto", max_vram_mb: int = 7600):
        self.preferred_device = preferred_device
        self.max_vram_mb = max_vram_mb

    def inspect(self) -> DeviceInfo:
        try:
            import torch
        except ImportError:
            return DeviceInfo(device="cpu", available=False, total_vram_mb=0, allocated_vram_mb=0)

        if not torch.cuda.is_available() or self.preferred_device == "cpu":
            return DeviceInfo(device="cpu", available=False, total_vram_mb=0, allocated_vram_mb=0)

        props = torch.cuda.get_device_properties(0)
        total = int(props.total_memory / 1024 / 1024)
        allocated = int(torch.cuda.memory_allocated(0) / 1024 / 1024)
        return DeviceInfo(device="cuda", available=True, total_vram_mb=total, allocated_vram_mb=allocated)

    def select_device(self) -> str:
        info = self.inspect()
        if info.device == "cuda" and info.total_vram_mb >= min(self.max_vram_mb, 2048):
            return "cuda"
        return "cpu"

    def cleanup(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    def snapshot(self) -> Dict[str, Any]:
        info = self.inspect()
        return {
            "device": info.device,
            "cuda_available": info.available,
            "total_vram_mb": info.total_vram_mb,
            "allocated_vram_mb": info.allocated_vram_mb,
            "max_vram_mb": self.max_vram_mb,
        }
