from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from local_translator.src.utils.logger import get_logger


class SileroVAD:
    """
    Thin wrapper around the silero-vad ONNX runtime model.
    The package provides a callable model that returns speech probabilities.
    """

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._log = get_logger(__name__)
        self._model = None
        self._lock = threading.Lock()
        self._load_model()

    def _load_model(self) -> None:
        try:
            # The silero-vad package may export either SileroVad or SileroVAD.
            try:
                from silero_vad import SileroVad  # type: ignore
            except ImportError:  # pragma: no cover - depends on package version
                from silero_vad import SileroVAD as SileroVad  # type: ignore

            self._model = SileroVad()
            self._log.info("Silero VAD model loaded (onnx)")
        except Exception as exc:  # pragma: no cover - defensive
            self._log.error("Failed to load Silero VAD: %s", exc)
            raise

    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Returns True if the audio frame contains speech with probability > threshold.
        """
        if self._model is None:
            raise RuntimeError("Silero VAD model not initialized")
        with self._lock:
            try:
                # Common interfaces: callable or .predict
                if hasattr(self._model, "predict"):
                    prob = float(self._model.predict(audio, self.sample_rate))
                else:
                    prob = float(self._model(audio, self.sample_rate))
            except Exception as exc:  # pragma: no cover - defensive
                self._log.error("VAD inference failed: %s", exc)
                prob = 0.0
        return prob >= self.threshold


