"""Conversation orchestration for Cosanta.

This is the only module that knows about *all* the others. It wires the pipeline
together and runs the main loop:

    wait for wake word -> record -> transcribe -> LLM -> speak -> repeat

Design choices:

* Components are injected via the constructor rather than created here, so this
  class is easy to test with fakes and the wiring lives in ``main.py``.
* Each turn is wrapped in its own try/except. A failure in one turn (network
  drop, a Whisper hiccup) is logged and the loop returns to wake-word
  listening instead of crashing — matching the "recover whenever possible"
  requirement.
* Conversation history lives here (not in the LLM provider), trimmed to the
  configured number of turns. This is the natural seam for a future persistent
  memory feature.
"""

from __future__ import annotations

import logging
import threading

from audio import AudioRecorder
from errors import CosantaError, LLMError, TranscriptionError, TTSError, WakeWordError
from llm import BaseLLM, Message
from settings import Settings
from speech_to_text import Transcriber
from tts import PiperTTS
from wakeword import WakeWordDetector

logger = logging.getLogger("cosanta.conversation")


class ConversationManager:
    """Orchestrates the full voice-assistant pipeline."""

    def __init__(
        self,
        settings: Settings,
        recorder: AudioRecorder,
        wakeword: WakeWordDetector,
        transcriber: Transcriber,
        llm: BaseLLM,
        tts: PiperTTS,
    ) -> None:
        self._settings = settings
        self._recorder = recorder
        self._wakeword = wakeword
        self._transcriber = transcriber
        self._llm = llm
        self._tts = tts

        self._stop = threading.Event()
        # Seed history with the system prompt; user/assistant turns append after.
        self._history: list[Message] = [
            {"role": "system", "content": settings.system_prompt}
        ]

    # -- shutdown ----------------------------------------------------------- #
    def stop(self) -> None:
        """Request a graceful shutdown (safe to call from a signal handler)."""
        self._stop.set()

    # -- history ------------------------------------------------------------ #
    def _append(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})
        self._trim_history()

    def _trim_history(self) -> None:
        """Keep the system prompt plus the last N (user, assistant) turns."""
        max_msgs = 1 + self._settings.history_max_turns * 2
        if len(self._history) > max_msgs:
            # Preserve index 0 (system prompt); drop the oldest turns.
            self._history = [self._history[0]] + self._history[-(max_msgs - 1):]

    # -- one turn ----------------------------------------------------------- #
    def _handle_turn(self) -> None:
        """Run a single record -> transcribe -> respond -> speak cycle."""
        logger.info("[Recorder] Recording...")
        audio = self._recorder.record_speech()

        try:
            text = self._transcriber.transcribe(audio)
        except TranscriptionError as exc:
            logger.error("Transcription error: %s", exc)
            return

        if not text:
            logger.info("Heard nothing intelligible; back to listening.")
            return

        self._append("user", text)

        try:
            reply = self._llm.generate(self._history)
        except LLMError as exc:
            logger.error("LLM error: %s", exc)
            self._safe_speak("Sorry, I could not reach my language model.")
            # Roll back the user turn so a failed exchange does not poison history.
            self._history.pop()
            return

        self._append("assistant", reply)
        logger.info("Cosanta: %s", reply)
        self._safe_speak(reply)

    def _safe_speak(self, text: str) -> None:
        """Speak text, but never let a TTS failure break the loop."""
        try:
            self._tts.speak(text)
        except TTSError as exc:
            logger.error("TTS error (continuing): %s", exc)

    # -- main loop ---------------------------------------------------------- #
    def run(self) -> None:
        """Open resources, run the loop, and always release them.

        The recorder owns the microphone for the whole session, so it is opened
        once here and shared between wake-word listening and speech capture.
        Porcupine's native handle is released on exit.
        """
        try:
            with self._recorder:
                self.run_forever()
        finally:
            self._wakeword.close()

    def run_forever(self) -> None:
        """Run the pipeline until :meth:`stop` is called or Ctrl+C is pressed."""
        logger.info("Cosanta is online. Say the wake word to begin.")
        while not self._stop.is_set():
            try:
                detected = self._wakeword.listen(self._recorder, self._stop)
                if not detected:
                    break  # stop_event was set during listening
                self._handle_turn()
            except WakeWordError as exc:
                # Wake-word engine failures are not per-turn recoverable.
                logger.critical("Wake-word engine failed: %s", exc)
                break
            except CosantaError as exc:
                logger.error("Recoverable error in turn: %s", exc)
                continue
            except Exception as exc:  # last-resort guard; keep the loop alive
                logger.exception("Unexpected error in turn: %s", exc)
                continue
        logger.info("Conversation loop stopped.")
