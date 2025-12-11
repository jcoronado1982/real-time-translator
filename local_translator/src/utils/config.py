from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    sample_rate: int = 16_000
    block_size: int = 480  # ~30 ms blocks at 16kHz
    channels: int = 1
    vad_threshold: float = 0.5
    max_silence_after_speech: float = 0.8  # seconds
    whisper_model_size: str = "small"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "int8"
    translation_model_name: str = "Helsinki-NLP/opus-mt-es-en"
    translation_device: str = "cuda"  # -1 for CPU in HF pipeline
    models_dir: Path = Path(__file__).resolve().parents[2] / "models"


settings = Settings()


