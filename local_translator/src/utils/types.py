from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float


