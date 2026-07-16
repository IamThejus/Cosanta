"""Shared exception hierarchy for Cosanta.

Keeping every custom exception in one place means the orchestration layer
(``conversation.py``) can catch a single base class (``CosantaError``) to decide
whether a failure is recoverable, while still allowing callers to catch a
specific subclass when they can do something smarter about it.
"""

from __future__ import annotations


class CosantaError(Exception):
    """Base class for every error raised inside Cosanta."""


class ConfigError(CosantaError):
    """Raised when configuration is missing or invalid (e.g. no API key)."""


class AudioError(CosantaError):
    """Raised for microphone capture or playback failures."""


class WakeWordError(CosantaError):
    """Raised when Porcupine cannot be initialised or fails at runtime."""


class TranscriptionError(CosantaError):
    """Raised when Faster Whisper fails to transcribe audio."""


class LLMError(CosantaError):
    """Raised when the language-model provider fails to return a response."""


class TTSError(CosantaError):
    """Raised when Piper fails to synthesise speech."""
