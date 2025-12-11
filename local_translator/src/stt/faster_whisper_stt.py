from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

# Asumo que esta ruta es correcta
from local_translator.src.utils.config import settings
from local_translator.src.utils.logger import get_logger
from local_translator.src.utils.types import TranscriptionResult


class FasterWhisperSTT:
    """
    Faster-Whisper wrapper configured for low-latency, small Spanish model.
    """

    def __init__(
        self,
        # Forzamos los valores para la GTX 1660, ignorando settings.py en el init
        model_size: str = "small",  
        device: str = "cuda",
        compute_type: str = "float16",
        model_dir: Optional[Path] = None,
    ):
        self._log = get_logger(__name__)
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model_dir = model_dir or settings.models_dir
        
        # El modelo se carga en la GPU (cuda)
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            download_root=str(self.model_dir),
        )
        self._log.info(
            "Loaded Faster-Whisper (size=%s, device=%s, compute=%s)",
            self.model_size,
            self.device,
            self.compute_type,
        )

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Run transcription on a mono float32 audio array (16 kHz).
        """
        segments, info = self._model.transcribe(
            audio=audio,
            language="es",
            beam_size=1,
            vad_filter=False,
        )
        # For low latency we concatenate text from all returned segments.
        text = " ".join(segment.text.strip() for segment in segments)
        return TranscriptionResult(
            text=text.strip(),
            language=info.language,
            duration=info.duration,
        )