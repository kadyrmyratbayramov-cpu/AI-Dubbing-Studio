"""Lip-sync engine interface and honest availability reporting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LipSyncStatus:
    available: bool
    mode: str
    message: str


class LipSyncEngine:
    def __init__(self) -> None:
        self.status = LipSyncStatus(
            available=False,
            mode="NOT AVAILABLE",
            message="Basic/advanced lip-sync is not enabled in this build. Audio timing sync is used instead.",
        )

    def get_status(self) -> LipSyncStatus:
        return self.status

    def apply(self, *_args, **_kwargs) -> None:
        """No-op placeholder with explicit status."""
        return None
