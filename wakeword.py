"""Wake-word detection using Picovoice Porcupine.

Single responsibility: turn a stream of microphone frames into a "wake word
detected" event. It knows nothing about recording speech, transcription, or the
LLM. It reads frames from an :class:`audio.AudioRecorder` so the microphone is
owned in exactly one place.

Porcupine dictates the audio format: 16 kHz, 16-bit mono, and a fixed frame
length (``porcupine.frame_length``, typically 512 samples). We expose that
frame length so the recorder can be configured to match.
"""

from __future__ import annotations

import logging
import threading

from audio import AudioRecorder
from errors import WakeWordError
from settings import Settings

logger = logging.getLogger("cosanta.wakeword")

try:
    import pvporcupine
except Exception:  # pragma: no cover - platform dependent
    pvporcupine = None  # type: ignore[assignment]


class WakeWordDetector:
    """Thin wrapper around a Porcupine handle."""

    def __init__(self, settings: Settings) -> None:
        if pvporcupine is None:
            raise WakeWordError(
                "pvporcupine is not installed. Run: pip install pvporcupine"
            )
        if not settings.porcupine_access_key:
            raise WakeWordError(
                "PORCUPINE_ACCESS_KEY is not set. Get a free key at "
                "https://console.picovoice.ai/ and add it to your .env."
            )

        self._settings = settings
        try:
            self._porcupine = self._create_handle(settings)
        except Exception as exc:
            raise WakeWordError(f"Failed to initialise Porcupine: {exc}") from exc

        logger.info(
            "Porcupine ready (frame_length=%d, sample_rate=%d)",
            self._porcupine.frame_length,
            self._porcupine.sample_rate,
        )

    @staticmethod
    def _create_handle(settings: Settings):
        """Build a Porcupine handle from either a custom or built-in keyword."""
        kwargs: dict = {
            "access_key": settings.porcupine_access_key,
            "sensitivities": [settings.wakeword_sensitivity],
        }
        if settings.wakeword_keyword_path:
            kwargs["keyword_paths"] = [settings.wakeword_keyword_path]
        else:
            kwargs["keywords"] = [settings.wakeword_builtin]
        if settings.wakeword_model_path:
            kwargs["model_path"] = settings.wakeword_model_path
        return pvporcupine.create(**kwargs)

    # -- properties used to configure the recorder -------------------------- #
    @property
    def frame_length(self) -> int:
        return self._porcupine.frame_length

    @property
    def sample_rate(self) -> int:
        return self._porcupine.sample_rate

    # -- detection ---------------------------------------------------------- #
    def process(self, frame) -> int:
        """Process one frame; returns the keyword index or -1 if none."""
        return self._porcupine.process(frame)

    def listen(
        self,
        recorder: AudioRecorder,
        stop_event: threading.Event | None = None,
    ) -> bool:
        """Block until the wake word is heard.

        Returns ``True`` when detected, or ``False`` if ``stop_event`` is set
        first (used for graceful shutdown). Raises :class:`WakeWordError` on an
        unrecoverable engine failure.
        """
        logger.info("[WakeWord] Listening...")
        while stop_event is None or not stop_event.is_set():
            try:
                frame = recorder.read(self.frame_length)
                if self._porcupine.process(frame) >= 0:
                    logger.info("[WakeWord] Detected")
                    return True
            except WakeWordError:
                raise
            except Exception as exc:  # pragma: no cover - hardware dependent
                raise WakeWordError(f"Wake-word processing failed: {exc}") from exc
        return False

    def close(self) -> None:
        try:
            self._porcupine.delete()
        except Exception:  # pragma: no cover
            pass
