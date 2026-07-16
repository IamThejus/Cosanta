"""Cosanta entry point.

Responsibilities:
* configure logging,
* load settings,
* build and wire every component (failing fast with a clear message if a
  dependency is missing or misconfigured),
* run the conversation loop,
* shut down gracefully on Ctrl+C / SIGTERM.

Run it with::

    python main.py
"""

from __future__ import annotations

import logging
import signal
import sys

from audio import AudioPlayer, AudioRecorder
from conversation import ConversationManager
from errors import CosantaError
from llm import build_llm
from settings import Settings
from speech_to_text import Transcriber
from tts import PiperTTS
from wakeword import WakeWordDetector


def configure_logging(settings: Settings) -> None:
    """Set up structured console (and optional file) logging.

    The format mirrors the bracketed component tags used throughout the code
    (e.g. ``[WakeWord] Listening...``) so logs read like a live transcript of
    the pipeline.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if settings.log_to_file:
        settings.ensure_dirs()
        handlers.append(logging.FileHandler(settings.log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def build_manager(settings: Settings) -> ConversationManager:
    """Construct every component and wire them into a ConversationManager.

    Ordering matters: Porcupine dictates the audio frame length, so we create
    the detector first and size the recorder to match.
    """
    log = logging.getLogger("cosanta.main")

    log.info("Initialising wake-word engine...")
    wakeword = WakeWordDetector(settings)

    log.info("Initialising microphone...")
    recorder = AudioRecorder(settings, frame_length=wakeword.frame_length)

    log.info("Initialising speech-to-text...")
    transcriber = Transcriber(settings)

    log.info("Initialising LLM provider (%s)...", settings.llm_provider)
    llm = build_llm(settings)

    log.info("Initialising text-to-speech...")
    player = AudioPlayer(settings)
    tts = PiperTTS(settings, player)

    return ConversationManager(
        settings=settings,
        recorder=recorder,
        wakeword=wakeword,
        transcriber=transcriber,
        llm=llm,
        tts=tts,
    )


def main() -> int:
    settings = Settings.from_env()
    settings.ensure_dirs()
    configure_logging(settings)
    log = logging.getLogger("cosanta.main")

    try:
        manager = build_manager(settings)
    except CosantaError as exc:
        # Configuration / dependency problems: report clearly and exit non-zero.
        log.critical("Startup failed: %s", exc)
        return 1

    # Graceful shutdown: Ctrl+C (SIGINT) and SIGTERM both ask the loop to stop.
    def _handle_signal(signum, _frame):
        log.info("Received signal %s; shutting down...", signum)
        manager.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        manager.run()
    except KeyboardInterrupt:  # belt-and-braces if the signal handler misses it
        log.info("Interrupted; shutting down...")
    except CosantaError as exc:
        log.critical("Fatal error: %s", exc)
        return 1

    log.info("Cosanta stopped. Goodbye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
