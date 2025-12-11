from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from local_translator.src.utils.logger import get_logger


class PiperTTS:
    """
    Wrapper for local Piper TTS using pre-downloaded binaries/models.
    Streams generated audio directly to aplay for low latency playback.
    """
    # model_name: str = "en_US-ryan-medium.onnx",
    def __init__(
        self,
        models_root: Optional[Path] = None,
        binary_name: str = "piper",   
        model_name: str = "en_US-ryan-high.onnx",
    ) -> None:
        self._log = get_logger(__name__)
        base_dir = models_root or Path(__file__).resolve().parents[2] / "models" / "piper"
        self.base_dir = base_dir
        self.piper_bin = (base_dir / binary_name).resolve()
        self.model_path = (base_dir / model_name).resolve()

        if not self.piper_bin.is_file():
            raise FileNotFoundError(f"Piper binary not found at {self.piper_bin}")
        if not os.access(self.piper_bin, os.X_OK):
            raise PermissionError(f"Piper binary is not executable: {self.piper_bin}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Piper model not found at {self.model_path}")

        self._log.info("Piper TTS initialized (bin=%s, model=%s)", self.piper_bin, self.model_path)

    def speak(self, text: str) -> None:
        if not text:
            return

        # Launch Piper and pipe its raw audio directly to aplay.
        try:
            piper_proc = subprocess.Popen(
                [str(self.piper_bin), "--model", str(self.model_path), "--output_raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            aplay_proc = subprocess.Popen(
                ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
                stdin=piper_proc.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            # Allow aplay to receive EOF correctly.
            if piper_proc.stdout:
                piper_proc.stdout.close()

            stdout_data, stderr_data = piper_proc.communicate(input=text.encode("utf-8"), timeout=30)
            aplay_stdout, aplay_stderr = aplay_proc.communicate(timeout=30)

            if piper_proc.returncode != 0:
                self._log.error("Piper failed (code=%s): %s", piper_proc.returncode, stderr_data.decode())
            if aplay_proc.returncode not in (0, None):
                self._log.error("aplay failed (code=%s): %s", aplay_proc.returncode, (aplay_stderr or b"").decode())
        except subprocess.TimeoutExpired:
            self._log.error("TTS playback timed out")
        except Exception as exc:  # pragma: no cover - defensive
            self._log.error("TTS playback error: %s", exc)


