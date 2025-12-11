from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from local_translator.src.utils.logger import get_logger


class MicrophoneStream:
    """
    Non-blocking microphone capture that pushes audio frames into a queue.
    """

    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        channels: int = 1,
        dtype: str = "float32",
        audio_queue: Optional[queue.Queue] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels
        self.dtype = dtype
        self.audio_queue = audio_queue or queue.Queue(maxsize=100)
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self._log = get_logger(__name__)

    def _callback(self, indata, frames, time, status) -> None:  # type: ignore[override]
        if status:
            self._log.warning("Sounddevice status: %s", status)
        with self._lock:
            if self._stream is None:
                return
        # Convert to mono if needed and copy to avoid referencing input buffer.
        data = np.copy(indata)
        if self.channels > 1:
            data = np.mean(data, axis=1, keepdims=True)
        data = data.astype(self.dtype, copy=False).flatten()
        try:
            self.audio_queue.put_nowait(data)
        except queue.Full:
            self._log.warning("Audio queue is full; dropping frame")

    def start(self) -> None:
        """
        Start microphone capture.
        """
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._callback,
        )
        self._stream.start()
        self._log.info(
            "Microphone stream started (sr=%d, block=%d)", self.sample_rate, self.block_size
        )

    def stop(self) -> None:
        """
        Stop microphone capture and close the stream.
        """
        with self._lock:
            if self._stream is None:
                return
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._log.info("Microphone stream stopped")


