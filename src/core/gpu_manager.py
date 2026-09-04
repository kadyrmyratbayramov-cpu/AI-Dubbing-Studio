"""GPU and VRAM lifecycle utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class GPUStatus:
    available: bool
    device_name: str = "CPU"
    total_vram_gb: float = 0.0
    used_vram_gb: float = 0.0


class GPUManager:
    def __init__(self, force_cpu: bool = False) -> None:
        self.force_cpu = force_cpu
        self._torch = None
        try:
            import torch

            self._torch = torch
        except Exception:
            self._torch = None

    def cuda_available(self) -> bool:
        return bool(self._torch and not self.force_cpu and self._torch.cuda.is_available())

    def preferred_device(self) -> str:
        return "cuda" if self.cuda_available() else "cpu"

    def get_status(self) -> GPUStatus:
        if not self.cuda_available():
            return GPUStatus(available=False)
        assert self._torch is not None
        props = self._torch.cuda.get_device_properties(0)
        used = self._torch.cuda.memory_allocated(0) / (1024**3)
        total = props.total_memory / (1024**3)
        return GPUStatus(True, props.name, total, used)

    def under_vram_limit(self, max_vram_gb: float) -> bool:
        status = self.get_status()
        return not status.available or status.used_vram_gb <= max_vram_gb

    def clear_cuda_cache(self) -> None:
        if self._torch and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def unload_model(self, model_obj: object) -> None:
        del model_obj
        self.clear_cuda_cache()

    def as_dict(self) -> Dict[str, str | float | bool]:
        status = self.get_status()
        return {
            "available": status.available,
            "device_name": status.device_name,
            "total_vram_gb": round(status.total_vram_gb, 3),
            "used_vram_gb": round(status.used_vram_gb, 3),
        }
