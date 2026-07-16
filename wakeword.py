"""Wake-word detection for Cosanta.

The rest of the application depends only on the abstract :class:`WakeWordEngine`
interface — it never imports a concrete engine directly. This is what lets us
swap Porcupine out for OpenWakeWord (and, later, anything else) without touching
``conversation.py``.

    class WakeWordEngine:
        def start(self):              # load the model / acquire resources
        def stop(self):               # signal the listen loop to stop
        def wait_for_wake_word(self): # block until the wake word is heard

Only :class:`OpenWakeWordEngine` is shipped here.

Why OpenWakeWord
----------------
OpenWakeWord is fully open-source, needs no access key or cloud account, ships
pretrained models ("hey_jarvis", "alexa", ...), and runs its inference through
ONNX Runtime — which already ships in this project for Piper and has good
aarch64/Termux wheels. It works on raw 16 kHz int16 audio in 80 ms (1280-sample)
chunks, so it slots straight into the existing shared microphone stream.

Microphone ownership is unchanged: the engine does *not* open the mic itself. It
reads frames from the shared :class:`audio.AudioRecorder` that ``main.py`` wires
in, so there is still exactly one owner of the audio device.
"""

from __future__ import annotations

import abc
import logging
import os
import threading
from pathlib import Path

from audio import AudioRecorder
from errors import WakeWordError
from settings import Settings

logger = logging.getLogger("cosanta.wakeword")

try:
    import openwakeword
    from openwakeword.model import Model as _OWWModel
except Exception:  # pragma: no cover - platform dependent
    openwakeword = None  # type: ignore[assignment]
    _OWWModel = None  # type: ignore[assignment]


class WakeWordEngine(abc.ABC):
    """Abstract wake-word engine.

    Implementations read audio from an injected recorder and block until their
    configured wake word is detected. ``stop()`` provides cooperative,
    thread-safe shutdown so a signal handler can unblock ``wait_for_wake_word``.
    """

    @abc.abstractmethod
    def start(self) -> None:
        """Load the model and prepare for listening. Idempotent."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Ask any in-progress ``wait_for_wake_word`` call to return."""

    @abc.abstractmethod
    def wait_for_wake_word(self) -> bool:
        """Block until the wake word is heard.

        Returns ``True`` when detected, or ``False`` if :meth:`stop` was called
        first (used for graceful shutdown). Raises :class:`WakeWordError` on an
        unrecoverable engine failure.
        """


class OpenWakeWordEngine(WakeWordEngine):
    """Wake-word engine backed by OpenWakeWord."""

    def __init__(self, settings: Settings, recorder: AudioRecorder) -> None:
        self._settings = settings
        self._recorder = recorder
        self._model = None  # created lazily in start()
        self._stopped = threading.Event()

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _looks_like_path(spec: str) -> bool:
        """Distinguish a custom model file from a pretrained model name."""
        return spec.endswith((".onnx", ".tflite")) or os.sep in spec

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> None:
        if self._model is not None:
            return
        if openwakeword is None or _OWWModel is None:
            raise WakeWordError(
                "openwakeword is not installed. Run: pip install openwakeword"
            )

        s = self._settings
        model_spec = s.wakeword_model
        is_path = self._looks_like_path(model_spec)

        # A missing *custom* model is a hard error; pretrained names are fetched
        # on demand below.
        if is_path and not Path(model_spec).exists():
            raise WakeWordError(
                f"Wake-word model not found: {model_spec!r}. Place your "
                ".onnx/.tflite model in models/ and set COSANTA_WAKEWORD_MODEL."
            )

        logger.info("[WakeWord] Loading OpenWakeWord model '%s'...", model_spec)

        # Ensure the shared feature extractors (and the pretrained model, if we
        # were given a name) are present. This is a no-op once cached, so a
        # download failure is only fatal if the files are genuinely missing.
        try:
            names = [] if is_path else [model_spec]
            openwakeword.utils.download_models(model_names=names)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning(
                "[WakeWord] Could not download models (%s); assuming cached.", exc
            )

        try:
            self._model = _OWWModel(
                wakeword_models=[model_spec],
                inference_framework=s.wakeword_inference_framework,
            )
        except Exception as exc:
            raise WakeWordError(
                f"Failed to initialise OpenWakeWord: {exc}"
            ) from exc

        self._stopped.clear()
        logger.info(
            "[WakeWord] Ready (framework=%s, threshold=%.2f, frame=%d).",
            s.wakeword_inference_framework,
            s.wakeword_threshold,
            s.wakeword_frame_length,
        )

    def stop(self) -> None:
        self._stopped.set()

    # -- detection ---------------------------------------------------------- #
    def wait_for_wake_word(self) -> bool:
        if self._model is None:
            self.start()

        # Reset the streaming buffers so audio captured during the previous turn
        # (e.g. the tail of the user's speech) cannot trigger a false detection.
        reset = getattr(self._model, "reset", None)
        if callable(reset):
            reset()

        s = self._settings
        frame_length = s.wakeword_frame_length
        threshold = s.wakeword_threshold

        logger.info("[WakeWord] Listening...")
        while not self._stopped.is_set():
            try:
                frame = self._recorder.read(frame_length)
            except WakeWordError:
                raise
            except Exception as exc:
                # Audio-stream hiccups are recoverable; surface as WakeWordError
                # only if the read cannot be retried by the caller.
                raise WakeWordError(f"Audio stream failure: {exc}") from exc

            try:
                scores = self._model.predict(frame)
            except Exception as exc:  # pragma: no cover - runtime dependent
                raise WakeWordError(f"OpenWakeWord inference failed: {exc}") from exc

            if scores and max(scores.values()) >= threshold:
                logger.info("[WakeWord] Wake word detected.")
                return True

        return False
