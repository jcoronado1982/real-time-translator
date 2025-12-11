from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from faster_whisper import WhisperModel

from local_translator.src.utils.logger import get_logger

AudioInput = Union[str, Path, np.ndarray, list[Any]]


class WhisperSTT:
    """
    High-performance wrapper around Faster-Whisper for Spanish transcription.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "int8",
    ) -> None:
        self._log = get_logger(__name__)

        # Fallback to CPU if CUDA not available.
        resolved_device = device
        if device == "cuda":
            try:
                import torch  # local import to avoid hard dependency if unused

                if not torch.cuda.is_available():
                    self._log.warning("CUDA not available; falling back to CPU")
                    resolved_device = "cpu"
            except Exception:  # pragma: no cover - defensive
                self._log.warning("CUDA check failed; falling back to CPU")
                resolved_device = "cpu"

        self.model = WhisperModel(
            model_size,
            device=resolved_device,
            compute_type=compute_type,
        )
        self.device = resolved_device
        self._log.info("Whisper Model loaded on %s", self.device)

    def transcribe(self, audio_segment: AudioInput) -> str:
        """
        Transcribe a given audio input to Spanish text.
        Accepts file paths or numpy arrays supported by faster-whisper.
        """
        if audio_segment is None:
            return ""

        segments_iter, _ = self.model.transcribe(
            audio_segment,
            language="es",
            beam_size=1,
            vad_filter=False,
            temperature=0.0,
        )

        segments = []
        for segment in segments_iter:
            print(
                f"DEBUG RAW SEGMENT: '{segment.text}' (Start: {segment.start}, End: {segment.end})"
            )
            segments.append(segment)
            print(f"DEBUG Segment: {segment.text} (Log-prob: {segment.avg_logprob})")

        text = " ".join(segment.text.strip() for segment in segments).strip()
        if not text:
            self._log.warning(
                "Audio processed but no text detected. Check microphone volume."
            )
        return text


if __name__ == "__main__":
    stt = WhisperSTT()

    # Optional: provide a wav path as the first argument.
    audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if audio_path and audio_path.exists():
        start = time.time()
        text = stt.transcribe(str(audio_path))
        elapsed = time.time() - start
        print(f"Transcription: {text}")
        print(f"Inference time: {elapsed:.3f}s")
    else:
        print("Model loaded. Provide a wav path as argument to test transcription.")


