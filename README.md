# Cosanta — a modular voice assistant

A modular, offline-first voice assistant that runs entirely inside **Termux on
Android** (aarch64, Python 3.11+). Everything is local except the LLM call to
Groq.

```
Microphone
  → Porcupine wake word      (wakeword.py)
  → Record speech            (audio.py)
  → Faster Whisper (STT)     (speech_to_text.py)
  → Groq LLM                 (llm.py)
  → Piper (TTS)              (tts.py)
  → Speaker                  (audio.py)
  → back to wake word        (conversation.py)
```

## Architecture

Each module has one job and holds no hidden global state — configuration is an
immutable `Settings` object passed in explicitly, and components are wired
together in `main.py`.

| File                 | Responsibility                                             |
|----------------------|------------------------------------------------------------|
| `settings.py`        | Configuration only (env-driven, immutable dataclass).      |
| `errors.py`          | Shared exception hierarchy (`CosantaError` and friends).      |
| `audio.py`           | Mic capture (single stream owner) + WAV I/O + playback.    |
| `wakeword.py`        | Porcupine wake-word detection.                             |
| `speech_to_text.py`  | Faster Whisper transcription.                              |
| `llm.py`             | `BaseLLM` provider interface + `GroqLLM`.                  |
| `tts.py`             | Piper synthesis + playback (via injected player).          |
| `conversation.py`    | Orchestrates the loop; owns history; per-turn recovery.    |
| `main.py`            | Logging, wiring, graceful shutdown.                        |

Two decisions worth calling out:

- **One microphone owner.** `AudioRecorder` opens a single PortAudio stream and
  both the wake-word loop and speech capture read frames from it. On Android,
  opening the mic from two places causes contention; sharing one stream avoids
  it. Porcupine dictates the frame size, so the recorder is sized to match.
- **Provider interface for the LLM.** `conversation.py` only knows `BaseLLM`.
  Adding Ollama/OpenAI later is a new subclass in `llm.py` plus one line in
  `build_llm()` — no orchestration changes.

## Setup (Termux)

```bash
# System packages
pkg update && pkg upgrade
pkg install python portaudio git

# Grant Termux mic access once (from termux-api)
pkg install termux-api        # optional playback fallback

# Python deps
pip install -r requirements.txt
```

> Some wheels (`ctranslate2` for faster-whisper, `onnxruntime` for Piper) can be
> slow or tricky to build on aarch64. Prefer small Whisper models
> (`tiny.en`/`base.en`) and `compute_type=int8`.

### Keys and models

1. `cp .env.example .env` and fill in `GROQ_API_KEY`
   (https://console.groq.com/) and `PORCUPINE_ACCESS_KEY`
   (https://console.picovoice.ai/).
2. Download a Piper voice into `voices/` from
   https://huggingface.co/rhasspy/piper-voices (grab both the `.onnx` and the
   `.onnx.json`) and set `COSANTA_PIPER_VOICE`.
3. (Optional) Train a custom `cosanta` wake word in the Picovoice console, drop the
   `.ppn` into `models/`, and set `COSANTA_WAKEWORD_PATH`.

## Run

```bash
python main.py
```

Say the wake word ("porcupine" by default), speak your request, and Cosanta
replies through the speaker. Press **Ctrl+C** to stop.

## Extending later

The seams are already in place for: a FastAPI/WebSocket server (wrap
`ConversationManager`), a local LLM (new `BaseLLM` subclass), persistent memory
(history already lives in `conversation.py`), and IoT/MQTT (subscribe to
transcription/response events). None of these require refactoring the core.
