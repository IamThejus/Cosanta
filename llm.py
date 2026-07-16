"""Language-model providers for Cosanta.

The pipeline talks to an abstract :class:`BaseLLM`; this module ships
:class:`GroqLLM`, a lightweight client that speaks Groq's OpenAI-compatible REST
API **directly over HTTPS with ``requests``**.

Why not the official ``groq`` SDK
---------------------------------
The SDK depends on pydantic v2, whose ``pydantic-core`` is a Rust extension
built with maturin. On Termux with Python 3.14 there is no matching wheel, so
pip tries to compile Rust and fails. ``requests`` (and its deps) are pure
Python, install cleanly, and the REST surface we need is tiny — so we implement
it ourselves. This keeps installation reliable, which is the project's top
priority.

Adding another backend later (Ollama, OpenAI, ...) is just a new subclass —
``conversation.py`` only ever sees :class:`BaseLLM`.
"""

from __future__ import annotations

import abc
import json
import logging
import time
from typing import Iterator, Sequence

import requests

from errors import ConfigError, LLMError
from settings import Settings

logger = logging.getLogger("cosanta.llm")

# A chat message is a plain dict so it is trivially JSON-serialisable.
Message = dict[str, str]

# HTTP statuses worth retrying: rate-limit + transient server/gateway errors.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class BaseLLM(abc.ABC):
    """Abstract chat-completion provider."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @abc.abstractmethod
    def generate(self, messages: Sequence[Message]) -> str:
        """Return the assistant's reply for a full message list.

        ``messages`` follows the OpenAI/Groq convention: a list of
        ``{"role": "system"|"user"|"assistant", "content": str}`` dicts.
        Implementations raise :class:`LLMError` on failure.
        """
        raise NotImplementedError


class GroqLLM(BaseLLM):
    """Groq chat-completions over REST using ``requests`` (no SDK)."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.groq_api_key:
            raise ConfigError(
                "GROQ_API_KEY is not set. Get a key at https://console.groq.com/ "
                "and add it to your .env."
            )
        self._url = settings.groq_base_url.rstrip("/") + "/chat/completions"
        # Reuse one TCP connection across turns (keep-alive) for lower latency.
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            }
        )

    # -- public API --------------------------------------------------------- #
    def generate(self, messages: Sequence[Message]) -> str:
        s = self._settings
        if s.llm_stream:
            return "".join(self.stream_generate(messages)).strip()

        payload = self._payload(messages, stream=False)
        resp = self._request_with_retry(payload, stream=False)
        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError) as exc:
            raise LLMError(f"Malformed Groq response: {exc}") from exc
        logger.info("[Groq] Response received (%d chars)", len(content))
        return content.strip()

    def stream_generate(self, messages: Sequence[Message]) -> Iterator[str]:
        """Yield reply text incrementally from the SSE stream.

        The pipeline currently buffers the whole reply before speaking, but this
        generator is the seam for future sentence-by-sentence TTS.
        """
        payload = self._payload(messages, stream=True)
        resp = self._request_with_retry(payload, stream=True)
        logger.info("[Groq] Streaming response...")
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content")
                except (ValueError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta
        except requests.RequestException as exc:
            raise LLMError(f"Groq stream interrupted: {exc}") from exc
        finally:
            resp.close()

    # -- internals ---------------------------------------------------------- #
    def _payload(self, messages: Sequence[Message], *, stream: bool) -> dict:
        s = self._settings
        return {
            "model": s.groq_model,
            "messages": list(messages),
            "temperature": s.llm_temperature,
            "max_tokens": s.llm_max_tokens,
            "stream": stream,
        }

    def _request_with_retry(self, payload: dict, *, stream: bool) -> requests.Response:
        """POST with linear backoff on transient network/HTTP failures."""
        s = self._settings
        last_error: str | None = None

        for attempt in range(1, s.llm_max_retries + 2):
            try:
                logger.info("[Groq] Sending request (attempt %d)...", attempt)
                resp = self._session.post(
                    self._url,
                    json=payload,
                    timeout=s.llm_timeout,
                    stream=stream,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = f"network error: {exc}"  # transient → retry
            else:
                if resp.status_code == 200:
                    return resp
                body = resp.text[:200]
                if resp.status_code in _RETRYABLE_STATUS:
                    last_error = f"HTTP {resp.status_code}: {body}"
                    resp.close()
                else:
                    # Auth/quota/bad-request errors will not fix themselves.
                    resp.close()
                    raise LLMError(f"Groq HTTP {resp.status_code}: {body}")

            logger.warning("[Groq] Request failed (%s)", last_error)
            if attempt <= s.llm_max_retries:
                time.sleep(attempt)  # 1s, 2s, ...

        raise LLMError(f"Groq request failed after retries: {last_error}")


def build_llm(settings: Settings) -> BaseLLM:
    """Factory returning the configured provider. Extend as backends are added."""
    providers = {"groq": GroqLLM}
    provider_cls = providers.get(settings.llm_provider.lower())
    if provider_cls is None:
        raise ConfigError(
            f"Unknown llm_provider '{settings.llm_provider}'. "
            f"Available: {', '.join(sorted(providers))}."
        )
    return provider_cls(settings)
