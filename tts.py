"""Text-to-speech using Piper.

Single responsibility: turn text into spoken audio. It synthesises a WAV file
with Piper and hands it to an injected :class:`audio.AudioPlayer` for playback.
Injecting the player (rather than importing/duplicating playback logic) keeps
all speaker code in ``audio.py`` and makes ``tts`` easy to unit-test with a fake
player. There is no LLM logic here.

Piper needs a voice model (``.onnx``) and its matching config (``.onnx.json``)
next to it. Download voices from:
https://huggingface.co/rhasspy/piper-voices — place them under ``voices/`` and
point ``COSANTA_PIPER_VOICE`` at the ``.onnx`` file.
"""

from __future__ import annotations

import logging
import tempfile
import wave
from pathlib import Path

from audio import AudioPlayer
from errors import TTSError
from settings import Settings

logger = logging.getLogger("cosanta.tts")

try:
    from piper import PiperVoice
except Exception:  # pragma: no cover - platform dependent
    PiperVoice = None  # type: ignore[assignment]


class PiperTTS:
    """Wraps a lazily-loaded Piper voice and plays its output."""

    def __init__(self, settings: Settings, player: AudioPlayer) -> None:
        self._settings = settings
        self._player = player
        self._voice: PiperVoice | None = None

    def _ensure_voice(self) -> "PiperVoice":
        if self._voice is not None:
            return self._voice
        if PiperVoice is None:
            raise TTSError(
                "piper-tts is not installed. Run: pip install piper-tts"
            )
        voice_path = self._settings.piper_voice_path
        if not voice_path or not Path(voice_path).exists():
            raise TTSError(
                f"Piper voice model not found: {voice_path!r}. Download a voice "
                "into voices/ and set COSANTA_PIPER_VOICE."
            )
        logger.info("[Piper] Loading voice '%s'...", voice_path)
        try:
            # Piper looks for the matching <model>.onnx.json config alongside
            # the model automatically.
            self._voice = PiperVoice.load(voice_path)
        except Exception as exc:
            raise TTSError(f"Could not load Piper voice: {exc}") from exc
        return self._voice

    def synthesize(self, text: str, path: str | Path | None = None) -> Path:
        """Synthesise ``text`` to a WAV file and return its path."""
        voice = self._ensure_voice()
        if path is None:
            fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            fd.close()
            path = fd.name
        path = Path(path)

        s = self._settings
        try:
            with wave.open(str(path), "wb") as wav:
                # Newer Piper accepts synthesis options via a SynthesisConfig;
                # older versions take keyword args. Try the modern path first.
                try:
                    from piper import SynthesisConfig

                    voice.synthesize_wav(
                        text,
                        wav,
                        syn_config=SynthesisConfig(
                            length_scale=s.piper_length_scale,
                            noise_scale=s.piper_noise_scale,
                            noise_w_scale=s.piper_noise_w,
                        ),
                    )
                except Exception:
                    # Fallback for older piper-tts APIs.
                    voice.synthesize(
                        text,
                        wav,
                        length_scale=s.piper_length_scale,
                        noise_scale=s.piper_noise_scale,
                        noise_w=s.piper_noise_w,
                    )
        except Exception as exc:
            raise TTSError(f"Piper synthesis failed: {exc}") from exc
        return path

    def speak(self, text: str) -> None:
        """Synthesise and immediately play ``text``. Empty text is a no-op."""
        text = (text or "").strip()
        if not text:
            return
        logger.info("[Piper] Speaking...")
        wav_path = self.synthesize(text)
        try:
            self._player.play_wav(wav_path)
        finally:
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:  # pragma: no cover
                pass
