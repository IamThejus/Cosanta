# Cosanta dependencies (Android / Termux, aarch64)

The guiding rule: **anything with native code is installed from `pkg`** (Termux
ships prebuilt aarch64 binaries), and **pip is used only for pure-Python
packages**. This keeps a fresh-Termux install essentially compilation-free,
which is the project's #1 priority.

`./setup.sh` performs everything below in order.

---

## Why the old desktop stack failed on Termux + Python 3.14

| Old dependency | Native build it triggered | Result on Termux |
|----------------|---------------------------|------------------|
| `groq` (SDK)   | pydantic-core (Rust, maturin) | ❌ no cp314 wheel → Rust build fails |
| `faster-whisper` | ctranslate2 (C++/CMake) + tokenizers (Rust) | ❌ hard build failures |
| `onnxruntime` (pip) | massive C++ project | ❌ no cp314 aarch64 wheel |
| `piper-tts`    | onnxruntime + espeak-ng phonemiser | ❌ fragile / fails |
| `numpy`/`scipy` (pip) | meson + ninja + BLAS/Fortran | ⚠️ slow, often fails |
| `sounddevice`  | CFFI | ⚠️ needs a compiler + PortAudio |

Every one of these is replaced or re-sourced below.

---

## `pkg install` — Termux packages (native, prebuilt)

Enable the Termux User Repository first: `pkg install tur-repo`.

| Package | Why it's needed | Why pkg (not pip) |
|---------|-----------------|-------------------|
| `python` | The interpreter. | — |
| `python-numpy` | Array math for audio frames / STT input. | Prebuilt; avoids a meson + BLAS build. |
| `python-scipy` | OpenWakeWord's signal-processing utilities import it. | Prebuilt; avoids a meson + Fortran build. |
| `python-scikit-learn` | OpenWakeWord's `__init__` eagerly imports `sklearn` (training code) even for inference-only use, so the package won't import without it. | Prebuilt; avoids a heavy Cython/OpenMP build. |
| `python-onnxruntime` | Inference backend that runs OpenWakeWord's ONNX models. | Prebuilt; the pip wheel doesn't exist for cp314/aarch64. On Android it prints a harmless "Unsupported platform" warning but runs fine on CPU. |
| `python-cffi` | `sounddevice` binds to PortAudio through CFFI. | Prebuilt; avoids compiling CFFI. |
| `portaudio` | Native audio library `sounddevice` drives for mic capture. | System library, only available via pkg. |
| `termux-api` | Provides `termux-tts-speak` (Android TTS) and `termux-microphone-record` (permission grant). | System bridge to Android; not a pip package. |
| `whisper.cpp` *(tur-repo; else built from source)* | Speech-to-text engine, run as a subprocess. | C/C++ program; building once beats compiling ctranslate2 for every install. |
| `git`, `cmake`, `clang`, `make` | Only used **if** whisper.cpp must be built from source. | Standard toolchain. |
| `curl` | Downloads the GGML whisper model. | — |

You also need the **Termux:API app** (from F-Droid) installed alongside the
`termux-api` package — the package is only the CLI half.

---

## `pip install` — pure-Python packages only

Installed into a venv created with `--system-site-packages` so the pkg-installed
native modules above remain visible.

| Package | Why it's needed | Why this choice |
|---------|-----------------|-----------------|
| `sounddevice` | Streams raw 16 kHz PCM frames from the mic — required by OpenWakeWord's continuous listening. | Pure Python; the only option that *streams* (`termux-microphone-record` only writes files and can't feed a wake-word loop). |
| `requests` | Talks to the Groq REST API and downloads models. | Pure Python. Directly replaces the `groq` SDK, which needs pydantic-core (Rust). |
| `tqdm` | Progress bars during OpenWakeWord's one-time model download. | Pure Python; an OpenWakeWord runtime import. |
| `python-dotenv` | Loads `.env` (a minimal fallback parser is built in, so this is optional). | Pure Python. |
| `openwakeword` | Wake-word detection. | Installed with **`--no-deps`** so pip cannot pull `tflite-runtime`/`scipy`/`onnxruntime` builds — its native needs are satisfied by the pkg packages. |

---

## Component decisions

- **Wake word — kept OpenWakeWord.** It runs fine on Termux once its native deps
  come from pkg and it's installed `--no-deps` on the ONNX backend. No change to
  the app's `WakeWordEngine` interface.
- **STT — Faster Whisper → whisper.cpp.** ctranslate2 + tokenizers can't build
  on Termux. whisper.cpp is a standalone binary with **no Python native
  dependency**; Cosanta shells out to it. Trade-off: the CLI reloads the model
  each turn (a second or two on an SD636). Reliability and maintainability
  outrank latency here; for lower latency run `whisper-server` and point
  `COSANTA_WHISPER_BINARY`/a server URL at it — the `Transcriber` API is
  unchanged either way.
- **TTS — Piper → Android TTS.** Piper's ONNX + espeak-ng chain is awkward on
  Termux. `termux-tts-speak` offloads synthesis to Android: no models, no native
  build. Piper is retained behind the same `TextToSpeech` interface
  (`COSANTA_TTS_BACKEND=piper`) for swap-back.
- **LLM — groq SDK → `requests`.** The SDK's pydantic-core (Rust) can't build;
  the REST surface we use is tiny, so `GroqLLM` calls it directly with retries,
  timeouts, and optional SSE streaming.

None of these changed a public API, so `conversation.py` and the overall
architecture are untouched.
