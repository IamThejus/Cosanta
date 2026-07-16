"""Speech-to-text via whisper.cpp.

Public API is unchanged: ``Transcriber(settings).transcribe(audio) -> str``.
Only the implementation changed — from Faster Whisper to whisper.cpp.

Why whisper.cpp instead of Faster Whisper
-----------------------------------------
Faster Whisper depends on **ctranslate2** (C++/CMake, no aarch64 cp314 wheel)
and **tokenizers** (Rust/maturin). Both fail to install on Termux. whisper.cpp
is a self-contained C/C++ program with **no Python native dependency**: it is
built once (from the Termux package or a one-time cmake build) and we drive its
CLI as a subprocess. That means Python needs zero compiled STT packages, which
is exactly what makes the install reliable on-device.

Trade-offs
----------
* whisper.cpp CLI reloads the model on each call, so there is a per-utterance
  load cost (a second or two for the ``base.en`` ggml model on an SD636). We
  accept this because reliability and maintainability rank above raw latency
  here, and the subprocess approach has nothing to keep warm or crash. For
  lower latency you can run ``whisper-server`` and point Cosanta at it — see
  README — without changing this module's public API.
* Models are GGML ``.bin`` files (e.g. ``ggml-base.en.bin``), downloaded once
  into ``models/``; there is no Python-side model download.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from errors import TranscriptionError
from settings import Settings

logger = logging.getLogger("cosanta.stt")


class Transcriber:
    """Transcribes recorded audio by invoking the whisper.cpp binary."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._binary: str | None = None  # resolved lazily on first use

    def _resolve_binary(self) -> str:
        """Find the whisper.cpp executable, trying modern then legacy names."""
        if self._binary is not None:
            return self._binary
        candidates = [self._settings.whisper_binary, "whisper-cli", "whisper", "main"]
        for name in candidates:
            found = shutil.which(name) or (name if Path(name).exists() else None)
            if found:
                self._binary = found
                logger.info("[Whisper] Using binary: %s", found)
                return found
        raise TranscriptionError(
            "whisper.cpp binary not found. Install it (pkg install whisper.cpp) "
            "or build it and set COSANTA_WHISPER_BINARY to its path."
        )

    def _write_wav(self, samples: np.ndarray) -> Path:
        """Write int16 PCM to a temporary 16 kHz mono WAV for whisper.cpp."""
        fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        fd.close()
        path = Path(fd.name)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self._settings.sample_rate)
            wav.writeframes(samples.astype(np.int16).tobytes())
        return path

    def transcribe(self, audio: np.ndarray | str | Path) -> str:
        """Transcribe int16 PCM samples (or a WAV path) to plain text."""
        s = self._settings
        binary = self._resolve_binary()

        if not Path(s.whisper_model_path).exists():
            raise TranscriptionError(
                f"Whisper model not found: {s.whisper_model_path}. Download a "
                "ggml model into models/ (see README) or set COSANTA_WHISPER_MODEL."
            )

        # Normalise input to a WAV file on disk (whisper.cpp reads files).
        cleanup: Path | None = None
        if isinstance(audio, np.ndarray):
            if audio.size == 0:
                return ""
            wav_path = self._write_wav(audio)
            cleanup = wav_path
        else:
            wav_path = Path(audio)

        cmd = [
            binary,
            "-m", s.whisper_model_path,
            "-f", str(wav_path),
            "-l", s.whisper_language,
            "-t", str(s.whisper_threads),
            "-nt",           # no timestamps: stdout is just the transcript
            "-np",           # no progress prints
        ]

        logger.info("[Whisper] Transcribing...")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=s.whisper_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TranscriptionError(f"whisper.cpp timed out: {exc}") from exc
        except OSError as exc:
            raise TranscriptionError(f"Could not run whisper.cpp: {exc}") from exc
        finally:
            if cleanup is not None:
                cleanup.unlink(missing_ok=True)

        if proc.returncode != 0:
            raise TranscriptionError(
                f"whisper.cpp failed (exit {proc.returncode}): {proc.stderr.strip()[:200]}"
            )

        text = proc.stdout.strip()
        logger.info("[Whisper] Heard: %r", text)
        return text
