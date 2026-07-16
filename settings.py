"""Configuration for Cosanta.

This module contains *configuration only* — no business logic. Every value is
either a sensible default or read from the environment (optionally via a
``.env`` file). The single ``Settings`` dataclass is passed explicitly to the
components that need it, so there is no hidden global state.

Design notes
------------
* We use a frozen dataclass so configuration is immutable once loaded — a
  component can never accidentally mutate shared settings at runtime.
* Paths are resolved relative to this file, so Cosanta works regardless of the
  current working directory (important in Termux where you may launch it from
  anywhere).
* Secrets (API keys) are never hard-coded; they come from the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent
MODELS_DIR: Path = BASE_DIR / "models"
VOICES_DIR: Path = BASE_DIR / "voices"
LOGS_DIR: Path = BASE_DIR / "logs"


def _load_dotenv() -> None:
    """Best-effort load of a ``.env`` file next to this module.

    We avoid a hard dependency on ``python-dotenv``: if it is installed we use
    it, otherwise we fall back to a tiny hand-rolled parser. Either way a
    missing ``.env`` is not an error — real environment variables win.
    """
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    try:  # pragma: no cover - optional dependency
        from dotenv import load_dotenv

        load_dotenv(env_path)
        return
    except ImportError:
        pass

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        # Do not override variables already present in the real environment.
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of all Cosanta configuration."""

    # -- Credentials -------------------------------------------------------- #
    groq_api_key: str = ""

    # -- Wake word (OpenWakeWord) ------------------------------------------ #
    # ``wakeword_model`` is either a pretrained model name ("hey_jarvis",
    # "alexa", "hey_mycroft", "hey_rhasspy", ...) or a path to a custom
    # ``.onnx``/``.tflite`` model placed in ``models/``.
    wakeword_model: str = "hey_jarvis"
    # Detection fires when a model score meets or exceeds this threshold (0-1).
    # Raise it to reduce false triggers; lower it if the word is missed.
    wakeword_threshold: float = 0.5
    # "onnx" has broad aarch64/Termux support and shares ONNX Runtime with
    # Piper; "tflite" is the alternative if you prefer tflite_runtime.
    wakeword_inference_framework: str = "onnx"
    # Samples fed to the model per step. 1280 = 80 ms at 16 kHz, OpenWakeWord's
    # native hop size. This also sizes the shared microphone frame.
    wakeword_frame_length: int = 1280

    # -- Speech-to-text (whisper.cpp) -------------------------------------- #
    # whisper.cpp is built once from pkg/source (no ctranslate2, no Rust) and
    # invoked as a subprocess, so Python needs zero native STT dependencies.
    whisper_binary: str = "whisper-cli"   # falls back to "main" if not found
    whisper_model_path: str = str(MODELS_DIR / "ggml-base.en.bin")
    whisper_language: str = "en"
    whisper_threads: int = 4              # SD636 has 8 cores; 4 keeps it cool
    whisper_timeout: float = 120.0

    # -- LLM (Groq REST) ---------------------------------------------------- #
    # No official SDK (it needs pydantic-core / Rust). We speak the OpenAI-
    # compatible REST API directly with `requests`.
    llm_provider: str = "groq"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512
    llm_timeout: float = 30.0
    llm_max_retries: int = 2
    llm_stream: bool = False              # accumulate SSE deltas when True
    system_prompt: str = (
        "You are Cosanta, a concise and helpful voice assistant running on a "
        "phone. Keep answers short and natural because they will be spoken "
        "aloud. Avoid markdown, lists, code blocks, and emoji."
    )
    # How many (user, assistant) exchanges to keep in context. Older turns are
    # dropped to bound memory and token usage.
    history_max_turns: int = 8

    # -- Text-to-speech ----------------------------------------------------- #
    # "android" uses the phone's built-in TTS via termux-tts-speak (no models,
    # no native build). "piper" keeps the old Piper engine for swap-back.
    tts_backend: str = "android"
    # Android TTS (termux-tts-speak) options:
    android_tts_engine: str = ""          # "" = system default engine
    android_tts_language: str = ""        # "" = system default (e.g. "en-US")
    android_tts_pitch: float = 1.0
    android_tts_rate: float = 1.0
    android_tts_stream: str = "NOTIFICATION"  # Android audio stream
    # Piper (only used when tts_backend == "piper"):
    piper_voice_path: str = ""            # path to a Piper ``.onnx`` voice model
    piper_length_scale: float = 1.0
    piper_noise_scale: float = 0.667
    piper_noise_w: float = 0.8

    # -- Audio -------------------------------------------------------------- #
    sample_rate: int = 16000            # required by OpenWakeWord & Whisper
    channels: int = 1
    # Speech recording: stop after this much trailing silence, but never record
    # longer than ``record_max_seconds`` or shorter than ``record_min_seconds``.
    record_max_seconds: float = 15.0
    record_min_seconds: float = 0.6
    silence_threshold: float = 500.0    # RMS amplitude (int16) considered silence
    silence_duration: float = 1.0       # seconds of silence that ends recording
    input_device: int | None = None     # sounddevice input device index
    output_device: int | None = None    # sounddevice output device index

    # -- Logging ------------------------------------------------------------ #
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = str(LOGS_DIR / "cosanta.log")

    # -- Derived ------------------------------------------------------------ #
    _dirs: tuple[Path, ...] = field(
        default=(MODELS_DIR, VOICES_DIR, LOGS_DIR), repr=False
    )

    @classmethod
    def from_env(cls) -> "Settings":
        """Build a ``Settings`` instance from environment variables.

        Only a handful of the most commonly-overridden values are wired to the
        environment; the rest use the dataclass defaults above. Extend this as
        needed — it is deliberately explicit rather than magic.
        """
        _load_dotenv()

        def env(name: str, default: str) -> str:
            return os.getenv(name, default)

        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            wakeword_model=env("COSANTA_WAKEWORD_MODEL", cls.wakeword_model),
            wakeword_threshold=float(
                env("COSANTA_WAKEWORD_THRESHOLD", str(cls.wakeword_threshold))
            ),
            wakeword_inference_framework=env(
                "COSANTA_WAKEWORD_FRAMEWORK", cls.wakeword_inference_framework
            ),
            whisper_binary=env("COSANTA_WHISPER_BINARY", cls.whisper_binary),
            whisper_model_path=env("COSANTA_WHISPER_MODEL", cls.whisper_model_path),
            whisper_language=env("COSANTA_WHISPER_LANGUAGE", cls.whisper_language),
            whisper_threads=int(env("COSANTA_WHISPER_THREADS", str(cls.whisper_threads))),
            groq_base_url=env("COSANTA_GROQ_BASE_URL", cls.groq_base_url),
            groq_model=env("COSANTA_GROQ_MODEL", cls.groq_model),
            llm_stream=_get_bool("COSANTA_LLM_STREAM", cls.llm_stream),
            tts_backend=env("COSANTA_TTS_BACKEND", cls.tts_backend),
            android_tts_language=env("COSANTA_TTS_LANGUAGE", cls.android_tts_language),
            android_tts_engine=env("COSANTA_TTS_ENGINE", cls.android_tts_engine),
            piper_voice_path=env("COSANTA_PIPER_VOICE", ""),
            log_level=env("COSANTA_LOG_LEVEL", cls.log_level),
            log_to_file=_get_bool("COSANTA_LOG_TO_FILE", cls.log_to_file),
        )

    def ensure_dirs(self) -> None:
        """Create the models/voices/logs directories if they do not exist."""
        for path in self._dirs:
            path.mkdir(parents=True, exist_ok=True)
