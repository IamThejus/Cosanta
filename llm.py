"""Language-model providers for Cosanta.

The pipeline talks to an abstract :class:`BaseLLM` interface, and this module
ships one concrete implementation, :class:`GroqLLM`. Adding a new backend later
(Ollama for local inference, OpenAI, Anthropic, ...) is just a new subclass —
``conversation.py`` never has to change. That is the whole point of keeping a
provider interface here.

The provider is stateless with respect to conversation history: callers pass the
full message list on every call. History management (trimming, persistence)
lives in ``conversation.py`` so that memory/history features can be added later
without touching provider code.
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Sequence

from errors import ConfigError, LLMError
from settings import Settings

logger = logging.getLogger("cosanta.llm")

# A chat message is a plain dict so it is trivially JSON-serialisable and works
# with every provider SDK unchanged.
Message = dict[str, str]


class BaseLLM(abc.ABC):
    """Abstract chat-completion provider."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @abc.abstractmethod
    def generate(self, messages: Sequence[Message]) -> str:
        """Return the assistant's reply for a full message list.

        ``messages`` follows the OpenAI/Groq convention: a list of
        ``{"role": "system"|"user"|"assistant", "content": str}`` dicts.
        Implementations should raise :class:`LLMError` on failure.
        """
        raise NotImplementedError


class GroqLLM(BaseLLM):
    """Groq chat-completions provider using the official ``groq`` SDK."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.groq_api_key:
            raise ConfigError(
                "GROQ_API_KEY is not set. Get a key at https://console.groq.com/ "
                "and add it to your .env."
            )
        try:
            from groq import Groq
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ConfigError(
                "The 'groq' package is not installed. Run: pip install groq"
            ) from exc

        self._client = Groq(
            api_key=settings.groq_api_key,
            timeout=settings.llm_timeout,
            max_retries=0,  # we implement our own retry/backoff below
        )

    def generate(self, messages: Sequence[Message]) -> str:
        s = self._settings
        last_exc: Exception | None = None

        # Retry transient failures (network blips, rate limits) with a small
        # linear backoff. Non-recoverable errors surface after the last attempt.
        for attempt in range(1, s.llm_max_retries + 2):
            try:
                logger.info("[Groq] Sending request (attempt %d)...", attempt)
                completion = self._client.chat.completions.create(
                    model=s.groq_model,
                    messages=list(messages),
                    temperature=s.llm_temperature,
                    max_tokens=s.llm_max_tokens,
                )
                content = (completion.choices[0].message.content or "").strip()
                logger.info("[Groq] Response received (%d chars)", len(content))
                return content
            except Exception as exc:  # SDK raises various APIError subclasses
                last_exc = exc
                logger.warning("[Groq] Request failed: %s", exc)
                if attempt <= s.llm_max_retries:
                    time.sleep(attempt)  # 1s, 2s, ...

        raise LLMError(f"Groq request failed after retries: {last_exc}") from last_exc


def build_llm(settings: Settings) -> BaseLLM:
    """Factory that returns the configured provider.

    Extend the mapping below as new providers are added.
    """
    providers = {
        "groq": GroqLLM,
    }
    provider_cls = providers.get(settings.llm_provider.lower())
    if provider_cls is None:
        raise ConfigError(
            f"Unknown llm_provider '{settings.llm_provider}'. "
            f"Available: {', '.join(sorted(providers))}."
        )
    return provider_cls(settings)
