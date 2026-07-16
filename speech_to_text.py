"""Speech-to-text using Faster Whisper.

Single responsibility: given recorded audio, return plain text. The model is
loaded lazily on first use so start-up is fast and a wake-word-only test run
does not pay the model-load cost.

Faster Whisper accepts either a file path or a float32 numpy array normalised to
[-1, 1]. We feed it the in-memory array to avoid a disk round-trip, but the
method also accepts a path for flexibility.

On Termux, ``faster-whisper`` pulls in ``ctranslate2``; use small models
(``tiny.en`` / ``base.en``) and ``compute_type="int8"`` to stay within a
phone's CPU/RAM budget.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from errors import TranscriptionError
from settings import Settings

logger = logging.getLogger("cosanta.stt")

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - platform dependent
    WhisperModel = None  # type: ignore[assignment]


class Transcriber:
    """Wraps a lazily-loaded Faster Whisper model."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: WhisperModel | None = None

    def _ensure_model(self) -> "WhisperModel":
        if self._model is not None:
            return self._model
        if WhisperModel is None:
            raise TranscriptionError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            )
        s = self._settings
        logger.info("[Whisper] Loading model '%s' (%s)...", s.whisper_model, s.whisper_compute_type)
        try:
            self._model = WhisperModel(
                s.whisper_model,
                device=s.whisper_device,
                compute_type=s.whisper_compute_type,
                download_root=str(s.MODELS_DIR),
            )
        except Exception as exc:
            raise TranscriptionError(f"Could not load Whisper model: {exc}") from exc
        return self._model

    def transcribe(self, audio: np.ndarray | str | Path) -> str:
        """Transcribe int16 PCM samples (or a WAV path) to plain text."""
        model = self._ensure_model()

        if isinstance(audio, np.ndarray):
            if audio.size == 0:
                return ""
            # Whisper wants float32 in [-1, 1]; our capture is int16.
            source: np.ndarray | str = (audio.astype(np.float32) / 32768.0)
        else:
            source = str(audio)

        logger.info("[Whisper] Transcribing...")
        try:
            segments, _info = model.transcribe(
                source,
                language=self._settings.whisper_language,
                vad_filter=True,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            raise TranscriptionError(f"Transcription failed: {exc}") from exc

        logger.info("[Whisper] Heard: %r", text)
        return text
