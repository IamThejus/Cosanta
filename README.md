# Cosanta — a modular voice assistant

A modular, offline-first voice assistant that runs entirely inside **Termux on
Android** (aarch64, Python 3.11+). Everything is local except the LLM call to
Groq.

```
Microphone
  → OpenWakeWord             (wakeword.py)
  → Record speech            (audio.py)
  → whisper.cpp (STT)        (speech_to_text.py)
  → Groq LLM (REST)          (llm.py)
  → Android TTS              (tts.py)
  → Speaker                  (Android)
  → back to wake word        (conversation.py)
```

The stack is chosen for **reliable, near-compilation-free installation on
Termux**. See [DEPENDENCIES.md](DEPENDENCIES.md) for the full rationale and the
`pkg` vs `pip` split.

## Architecture

Each module has one job and holds no hidden global state — configuration is an
immutable `Settings` object passed in explicitly, and components are wired
together in `main.py`.

| File                 | Responsibility                                             |
|----------------------|------------------------------------------------------------|
| `settings.py`        | Configuration only (env-driven, immutable dataclass).      |
| `errors.py`          | Shared exception hierarchy (`CosantaError` and friends).      |
| `audio.py`           | Mic capture (single stream owner) + WAV I/O + playback.    |
| `wakeword.py`        | `WakeWordEngine` interface + `OpenWakeWordEngine`.         |
| `speech_to_text.py`  | whisper.cpp transcription (subprocess).                    |
| `llm.py`             | `BaseLLM` interface + `GroqLLM` (REST via `requests`).     |
| `tts.py`             | `TextToSpeech` interface + `AndroidTTS` / `PiperTTS`.      |
| `conversation.py`    | Orchestrates the loop; owns history; per-turn recovery.    |
| `main.py`            | Logging, wiring, graceful shutdown.                        |

Three decisions worth calling out:

- **One microphone owner.** `AudioRecorder` opens a single PortAudio stream and
  both the wake-word loop and speech capture read frames from it. On Android,
  opening the mic from two places causes contention; sharing one stream avoids
  it. The frame size (1280 samples = OpenWakeWord's 80 ms hop) sizes the stream.
- **Wake-word engine behind an interface.** `conversation.py` depends only on
  the abstract `WakeWordEngine` (`start` / `stop` / `wait_for_wake_word`), never
  on a concrete engine. `OpenWakeWordEngine` is the shipped implementation;
  swapping it out touches only `wakeword.py` and the wiring in `main.py`.
- **Provider interface for the LLM.** `conversation.py` only knows `BaseLLM`.
  Adding Ollama/OpenAI later is a new subclass in `llm.py` plus one line in
  `build_llm()` — no orchestration changes.

## Setup (Termux)

**One command** does the whole install — system packages, Python env, pip
packages, whisper.cpp + model, and verification of every stage:

```bash
cp .env.example .env      # then add your GROQ_API_KEY
./setup.sh
```

Prerequisites: a fresh **Termux** (from F-Droid) plus the **Termux:API app**
(also F-Droid). The only key you need is a free Groq key from
https://console.groq.com/. No wake-word account is required — OpenWakeWord is
open-source, and its `hey_jarvis` model downloads automatically on first run.

`setup.sh` follows the `pkg`-first / `pip`-for-pure-Python rule so it compiles
almost nothing. For the manual step-by-step and the reasoning behind each
dependency, see [DEPENDENCIES.md](DEPENDENCIES.md).

### Optional configuration

- **Wake word** — `COSANTA_WAKEWORD_MODEL`: a pretrained name (`hey_jarvis`,
  `alexa`, `hey_mycroft`, `hey_rhasspy`) or a path to your own `.onnx` model.
- **STT model** — the `base.en` GGML model is fetched by `setup.sh`; swap it via
  `COSANTA_WHISPER_MODEL` (e.g. `models/ggml-tiny.en.bin` for less RAM/latency).
- **TTS** — defaults to Android's built-in engine. Set `COSANTA_TTS_BACKEND=piper`
  (plus `COSANTA_PIPER_VOICE`) to use Piper instead.

## Run

```bash
source .venv/bin/activate
python main.py
```

Say the wake word ("hey jarvis" by default), speak your request, and Cosanta
answers through the phone's speaker. Press **Ctrl+C** to stop.

## Extending later

The seams are already in place for: a FastAPI/WebSocket server (wrap
`ConversationManager`), a local LLM (new `BaseLLM` subclass), persistent memory
(history already lives in `conversation.py`), and IoT/MQTT (subscribe to
transcription/response events). None of these require refactoring the core.
