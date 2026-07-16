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
    porcupine_access_key: str = ""

    # -- Wake word (Porcupine) --------------------------------------------- #
    # If ``wakeword_keyword_path`` points at a ``.ppn`` file it is used;
    # otherwise ``wakeword_builtin`` selects one of Porcupine's built-in words
    # (e.g. "porcupine", "jarvis", "computer").
    wakeword_keyword_path: str = ""
    wakeword_builtin: str = "porcupine"
    wakeword_sensitivity: float = 0.6
    wakeword_model_path: str = ""  # optional custom Porcupine model (.pv)

    # -- Speech-to-text (Faster Whisper) ----------------------------------- #
    whisper_model: str = "base.en"
    whisper_device: str = "cpu"          # "cpu" is the safe default on Termux
    whisper_compute_type: str = "int8"   # int8 keeps memory/CPU low on ARM
    whisper_language: str = "en"

    # -- LLM (Groq) --------------------------------------------------------- #
    llm_provider: str = "groq"
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512
    llm_timeout: float = 30.0
    llm_max_retries: int = 2
    system_prompt: str = (
        "You are Cosanta, a concise and helpful voice assistant running on a "
        "phone. Keep answers short and natural because they will be spoken "
        "aloud. Avoid markdown, lists, code blocks, and emoji."
    )
    # How many (user, assistant) exchanges to keep in context. Older turns are
    # dropped to bound memory and token usage.
    history_max_turns: int = 8

    # -- Text-to-speech (Piper) -------------------------------------------- #
    piper_voice_path: str = ""  # path to a Piper ``.onnx`` voice model
    piper_length_scale: float = 1.0
    piper_noise_scale: float = 0.667
    piper_noise_w: float = 0.8

    # -- Audio -------------------------------------------------------------- #
    sample_rate: int = 16000            # required by Porcupine & Whisper
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
            porcupine_access_key=os.getenv("PORCUPINE_ACCESS_KEY", ""),
            wakeword_keyword_path=env("COSANTA_WAKEWORD_PATH", ""),
            wakeword_builtin=env("COSANTA_WAKEWORD_BUILTIN", cls.wakeword_builtin),
            wakeword_sensitivity=float(
                env("COSANTA_WAKEWORD_SENSITIVITY", str(cls.wakeword_sensitivity))
            ),
            whisper_model=env("COSANTA_WHISPER_MODEL", cls.whisper_model),
            whisper_device=env("COSANTA_WHISPER_DEVICE", cls.whisper_device),
            whisper_compute_type=env(
                "COSANTA_WHISPER_COMPUTE_TYPE", cls.whisper_compute_type
            ),
            whisper_language=env("COSANTA_WHISPER_LANGUAGE", cls.whisper_language),
            groq_model=env("COSANTA_GROQ_MODEL", cls.groq_model),
            piper_voice_path=env("COSANTA_PIPER_VOICE", ""),
            log_level=env("COSANTA_LOG_LEVEL", cls.log_level),
            log_to_file=_get_bool("COSANTA_LOG_TO_FILE", cls.log_to_file),
        )

    def ensure_dirs(self) -> None:
        """Create the models/voices/logs directories if they do not exist."""
        for path in self._dirs:
            path.mkdir(parents=True, exist_ok=True)
