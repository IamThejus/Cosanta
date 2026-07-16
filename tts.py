"""Text-to-speech for Cosanta.

The rest of the app calls a single method: ``tts.speak(text)``. That contract is
defined by the abstract :class:`TextToSpeech` so the engine can be swapped
freely. Two implementations ship here:

* :class:`AndroidTTS` (default) — uses the phone's built-in TTS via
  ``termux-tts-speak`` (from termux-api). No models, no ONNX Runtime, no native
  build: it offloads synthesis to Android itself, which is the most reliable
  option on Termux.
* :class:`PiperTTS` — the previous Piper engine, kept behind the same interface
  so it can be swapped back once Piper installs cleanly (set
  ``COSANTA_TTS_BACKEND=piper``).

No LLM logic lives here.
"""

from __future__ import annotations

import abc
import logging
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from audio import AudioPlayer
from errors import TTSError
from settings import Settings

logger = logging.getLogger("cosanta.tts")


class TextToSpeech(abc.ABC):
    """Abstract speech synthesiser. The whole public surface is ``speak``."""

    @abc.abstractmethod
    def speak(self, text: str) -> None:
        """Synthesise and play ``text``. Empty text is a no-op."""


class AndroidTTS(TextToSpeech):
    """Speak through Android's system TTS via ``termux-tts-speak``.

    Android performs synthesis and playback, so Cosanta needs no voice model
    and no audio-output library on this path.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if shutil.which("termux-tts-speak") is None:
            raise TTSError(
                "termux-tts-speak not found. Install Termux:API: "
                "pkg install termux-api (and the Termux:API app)."
            )

    def speak(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        s = self._settings
        cmd = ["termux-tts-speak"]
        if s.android_tts_engine:
            cmd += ["-e", s.android_tts_engine]
        if s.android_tts_language:
            cmd += ["-l", s.android_tts_language]
        cmd += [
            "-p", str(s.android_tts_pitch),
            "-r", str(s.android_tts_rate),
            "-s", s.android_tts_stream,
            text,
        ]
        logger.info("[TTS] Speaking (Android)...")
        try:
            # Blocks until Android finishes speaking, keeping the loop in order.
            subprocess.run(cmd, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise TTSError(f"Android TTS failed: {exc}") from exc


class PiperTTS(TextToSpeech):
    """Piper voice model synthesis, played via an injected AudioPlayer.

    Retained for parity/swap-back; not the default on Termux because Piper pulls
    ONNX Runtime + espeak-ng phonemisation that are awkward to install there.
    """

    def __init__(self, settings: Settings, player: AudioPlayer) -> None:
        self._settings = settings
        self._player = player
        self._voice = None

    def _ensure_voice(self):
        if self._voice is not None:
            return self._voice
        try:
            from piper import PiperVoice
        except Exception as exc:  # pragma: no cover - optional
            raise TTSError("piper-tts is not installed.") from exc
        voice_path = self._settings.piper_voice_path
        if not voice_path or not Path(voice_path).exists():
            raise TTSError(f"Piper voice model not found: {voice_path!r}.")
        logger.info("[TTS] Loading Piper voice '%s'...", voice_path)
        try:
            self._voice = PiperVoice.load(voice_path)
        except Exception as exc:
            raise TTSError(f"Could not load Piper voice: {exc}") from exc
        return self._voice

    def speak(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        voice = self._ensure_voice()
        fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        fd.close()
        wav_path = Path(fd.name)
        logger.info("[TTS] Speaking (Piper)...")
        try:
            with wave.open(str(wav_path), "wb") as wav:
                voice.synthesize(text, wav)
            self._player.play_wav(wav_path)
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"Piper synthesis failed: {exc}") from exc
        finally:
            wav_path.unlink(missing_ok=True)


def build_tts(settings: Settings, player: AudioPlayer) -> TextToSpeech:
    """Factory returning the configured TTS engine."""
    backend = settings.tts_backend.lower()
    if backend == "android":
        return AndroidTTS(settings)
    if backend == "piper":
        return PiperTTS(settings, player)
    raise TTSError(
        f"Unknown tts_backend '{settings.tts_backend}'. Use 'android' or 'piper'."
    )
