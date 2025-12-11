from __future__ import annotations

import queue
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

from local_translator.src.audio.microphone_stream import MicrophoneStream
from local_translator.src.stt.faster_whisper_stt import FasterWhisperSTT
from local_translator.src.translation.helsinki_translator import HelsinkiTranslator
from local_translator.src.utils.config import settings
from local_translator.src.utils.logger import get_logger
from local_translator.src.vad.silero_vad import SileroVAD

log = get_logger("main")


class InputPipeline:
    """
    Microphone -> VAD -> STT -> Translation pipeline.
    """

    def __init__(self) -> None:
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        models_dir = settings.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)

        self.microphone = MicrophoneStream(
            sample_rate=settings.sample_rate,
            block_size=settings.block_size,
            channels=settings.channels,
            audio_queue=self.audio_queue,
        )
        self.vad = SileroVAD(
            sample_rate=settings.sample_rate,
            threshold=settings.vad_threshold,
        )
        self.stt = FasterWhisperSTT(model_dir=models_dir)
        self.translator = HelsinkiTranslator(
            model_dir=str(models_dir),
            device=settings.translation_device,
        )

        self._processing_thread: threading.Thread | None = None
        self._running = threading.Event()
        self._frame_duration = settings.block_size / settings.sample_rate

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self.microphone.start()
        self._processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._processing_thread.start()
        log.info("Pipeline started")

    def stop(self) -> None:
        self._running.clear()
        self.microphone.stop()
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2)
        log.info("Pipeline stopped")

    def _process_loop(self) -> None:
        speech_buffer: list[np.ndarray] = []
        speech_active = False
        silence_accum = 0.0

        while self._running.is_set():
            try:
                frame = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            speech_detected = self.vad.is_speech(frame)
            if speech_detected:
                speech_buffer.append(frame)
                speech_active = True
                silence_accum = 0.0
            else:
                if speech_active:
                    silence_accum += self._frame_duration
                    if silence_accum >= settings.max_silence_after_speech:
                        self._flush_segment(speech_buffer)
                        speech_buffer = []
                        speech_active = False
                        silence_accum = 0.0

        # Flush remaining buffered speech when stopping.
        if speech_buffer:
            self._flush_segment(speech_buffer)

    def _flush_segment(self, speech_buffer: list[np.ndarray]) -> None:
        if not speech_buffer:
            return
        audio = np.concatenate(speech_buffer)
        try:
            transcription = self.stt.transcribe(audio)
            translation = self.translator.translate(transcription.text)
            log.info(
                "ES: %s | EN: %s",
                transcription.text,
                translation,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error("Failed to process segment: %s", exc)


def main(run_seconds: int = 60) -> None:
    pipeline = InputPipeline()
    stop_event = threading.Event()

    def _handle_sigint(signum, frame):
        log.info("Received interrupt, stopping...")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    pipeline.start()
    start_time = time.time()

    try:
        while not stop_event.is_set():
            if run_seconds and (time.time() - start_time) >= run_seconds:
                log.info("Run duration reached (%ss); stopping.", run_seconds)
                break
            time.sleep(0.1)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    duration = 120 if len(sys.argv) < 2 else int(sys.argv[1])
    main(run_seconds=duration)


