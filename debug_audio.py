from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path
from typing import Tuple

import numpy as np


def read_wav(path: Path) -> Tuple[int, np.ndarray]:
    """
    Read a WAV file, preferring scipy.io.wavfile if available, otherwise using
    the standard library wave module.
    """
    try:
        from scipy.io import wavfile  # type: ignore

        sample_rate, data = wavfile.read(path)
        return int(sample_rate), np.array(data)
    except Exception:
        with wave.open(str(path), "rb") as wf:
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            audio_bytes = wf.readframes(frames)
            # Determine dtype from sample width
            sampwidth = wf.getsampwidth()
            if sampwidth == 1:
                dtype = np.uint8  # 8-bit PCM unsigned
            elif sampwidth == 2:
                dtype = np.int16
            elif sampwidth == 4:
                dtype = np.int32
            else:
                raise ValueError(f"Unsupported sample width: {sampwidth}")

            data = np.frombuffer(audio_bytes, dtype=dtype)
            channels = wf.getnchannels()
            if channels > 1:
                data = data.reshape(-1, channels)
        return sample_rate, data


def analyze(path: Path) -> None:
    sample_rate, data = read_wav(path)
    duration = len(data) / float(sample_rate)
    max_amplitude = float(np.max(np.abs(data))) if data.size else 0.0

    print(f"File: {path}")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Max Amplitude: {max_amplitude}")
    print(f"Duration: {duration:.3f} seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug WAV file properties.")
    parser.add_argument(
        "wav_path",
        nargs="?",
        default="prueba.wav",
        help="Path to WAV file (default: prueba.wav)",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav_path)
    if not wav_path.exists():
        print(f"File not found: {wav_path}")
        sys.exit(1)

    analyze(wav_path)


if __name__ == "__main__":
    main()


