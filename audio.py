"""Audio capture and playback for Cosanta.

This module is intentionally free of any AI logic. It does three things:

1. Owns the microphone via a single persistent input stream. Both wake-word
   detection and speech recording read frames from the *same* stream, which
   avoids device-contention problems that appear on Android/Termux when two
   parts of the program try to open the mic at once.
2. Records speech with simple energy-based endpointing (stop after a period of
   trailing silence) so the user does not have to talk for a fixed duration.
3. Plays WAV files, preferring ``sounddevice`` but falling back to Termux's
   command-line players if PortAudio output is unavailable.

The primary backend is ``sounddevice`` (PortAudio). On Termux install it with::

    pkg install portaudio
    pip install sounddevice numpy
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from errors import AudioError
from settings import Settings

logger = logging.getLogger("cosanta.audio")

try:  # sounddevice is optional at import time so the module can be inspected.
    import sounddevice as sd
except Exception:  # pragma: no cover - platform dependent
    sd = None  # type: ignore[assignment]


class AudioRecorder:
    """Persistent microphone stream producing 16-bit PCM frames.

    Use it as a context manager so the underlying stream is always closed::

        with AudioRecorder(settings) as rec:
            frame = rec.read(512)          # for the wake-word engine
            speech = rec.record_speech()   # for speech-to-text
    """

    def __init__(self, settings: Settings, frame_length: int = 512) -> None:
        self._settings = settings
        self._frame_length = frame_length
        self._stream = None  # created in open()

    # -- lifecycle ---------------------------------------------------------- #
    def open(self) -> "AudioRecorder":
        if sd is None:
            raise AudioError(
                "sounddevice is not available. Install PortAudio and the "
                "sounddevice package (see module docstring)."
            )
        try:
            self._stream = sd.RawInputStream(
                samplerate=self._settings.sample_rate,
                blocksize=self._frame_length,
                device=self._settings.input_device,
                channels=self._settings.channels,
                dtype="int16",
            )
            self._stream.start()
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise AudioError(f"Could not open microphone: {exc}") from exc
        logger.debug("Microphone stream opened (%d Hz)", self._settings.sample_rate)
        return self

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
                logger.debug("Microphone stream closed")

    def __enter__(self) -> "AudioRecorder":
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- frame access ------------------------------------------------------- #
    @property
    def frame_length(self) -> int:
        return self._frame_length

    def read(self, num_frames: int | None = None) -> np.ndarray:
        """Read exactly ``num_frames`` samples as an ``int16`` numpy array.

        Defaults to ``frame_length`` samples, which is what Porcupine expects.
        """
        if self._stream is None:
            raise AudioError("Recorder is not open; use it as a context manager.")
        count = num_frames or self._frame_length
        data, overflowed = self._stream.read(count)
        if overflowed:
            logger.warning("Microphone input overflow (dropped samples)")
        return np.frombuffer(data, dtype=np.int16)

    # -- speech recording --------------------------------------------------- #
    def record_speech(self) -> np.ndarray:
        """Record until the user stops talking, returning int16 PCM.

        Endpointing is done with a running RMS energy measure: once we have
        recorded at least ``record_min_seconds`` and then observe
        ``silence_duration`` seconds below ``silence_threshold``, recording
        stops. A hard cap of ``record_max_seconds`` guarantees termination.
        """
        s = self._settings
        frames: list[np.ndarray] = []
        frames_per_second = s.sample_rate / self._frame_length
        max_frames = int(s.record_max_seconds * frames_per_second)
        min_frames = int(s.record_min_seconds * frames_per_second)
        silence_frames_needed = int(s.silence_duration * frames_per_second)

        silent_run = 0
        for i in range(max_frames):
            frame = self.read(self._frame_length)
            frames.append(frame)

            rms = _rms(frame)
            if rms < s.silence_threshold:
                silent_run += 1
            else:
                silent_run = 0

            if i >= min_frames and silent_run >= silence_frames_needed:
                break

        if not frames:
            return np.empty(0, dtype=np.int16)
        return np.concatenate(frames)

    # -- persistence -------------------------------------------------------- #
    def save_wav(self, samples: np.ndarray, path: str | Path) -> Path:
        """Write int16 PCM samples to a mono 16-bit WAV file."""
        path = Path(path)
        try:
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(self._settings.channels)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(self._settings.sample_rate)
                wav.writeframes(samples.astype(np.int16).tobytes())
        except Exception as exc:
            raise AudioError(f"Failed to write WAV file {path}: {exc}") from exc
        return path


class AudioPlayer:
    """Play WAV files through the phone speaker.

    Tries ``sounddevice`` first (lowest latency, no external process). If that
    is unavailable or fails, it falls back to whichever Termux CLI player is on
    ``PATH`` (``play-audio`` from termux-api, or ``termux-media-player``).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def play_wav(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise AudioError(f"Cannot play missing audio file: {path}")

        if sd is not None:
            try:
                self._play_with_sounddevice(path)
                return
            except Exception as exc:  # pragma: no cover - hardware dependent
                logger.warning("sounddevice playback failed (%s); trying Termux", exc)

        self._play_with_termux(path)

    def _play_with_sounddevice(self, path: Path) -> None:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        data = np.frombuffer(frames, dtype=np.int16)
        sd.play(data, samplerate=rate, device=self._settings.output_device)
        sd.wait()

    def _play_with_termux(self, path: Path) -> None:
        for player in ("play-audio", "termux-media-player"):
            binary = shutil.which(player)
            if not binary:
                continue
            cmd = (
                [binary, str(path)]
                if player == "play-audio"
                else [binary, "play", str(path)]
            )
            try:
                subprocess.run(cmd, check=True)
                return
            except Exception as exc:  # pragma: no cover
                logger.warning("%s playback failed: %s", player, exc)
        raise AudioError(
            "No working audio output. Install PortAudio (pkg install portaudio) "
            "or termux-api (pkg install termux-api)."
        )


def _rms(frame: np.ndarray) -> float:
    """Root-mean-square amplitude of an int16 frame, as a float."""
    if frame.size == 0:
        return 0.0
    # Work in float64 to avoid int16 overflow when squaring.
    return float(np.sqrt(np.mean(np.square(frame.astype(np.float64)))))
