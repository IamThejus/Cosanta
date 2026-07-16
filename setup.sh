#!/data/data/com.termux/files/usr/bin/bash
#
# Cosanta setup for Android / Termux (aarch64).
#
# Strategy: everything with native code comes from `pkg` (prebuilt for aarch64);
# pip is used only for pure-Python packages. This keeps the install
# compilation-free and reliable on a fresh Termux.
#
# Steps: pkg packages -> verify -> Python env -> pip packages -> whisper.cpp +
# model -> mic permission -> verify OpenWakeWord / STT / Groq / TTS.
#
# Safe to re-run; it skips work that is already done.

set -u
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

PASS=0; FAIL=0
ok()   { echo "  [ OK ] $*"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
info() { echo; echo "==> $*"; }

# --------------------------------------------------------------------------- #
# 1. Termux system packages (native code, prebuilt — no compilation)
# --------------------------------------------------------------------------- #
info "1/9  Installing Termux packages"
pkg update -y
# tur-repo provides onnxruntime / scipy / whisper.cpp.
pkg install -y tur-repo
PKGS=(
  python            # interpreter
  python-numpy      # arrays (avoids meson/BLAS build)
  python-scipy      # OpenWakeWord signal utils (avoids meson/Fortran build)
  python-onnxruntime  # OpenWakeWord inference backend (avoids huge C++ build)
  python-cffi       # sounddevice needs CFFI (avoids compiling it)
  portaudio         # native audio lib for sounddevice mic capture
  termux-api        # termux-tts-speak (Android TTS) + microphone permission
  git cmake         # only needed if whisper.cpp must be built from source
  clang make        # toolchain fallback for the whisper.cpp build
  curl              # model downloads
)
pkg install -y "${PKGS[@]}"

# --------------------------------------------------------------------------- #
# 2. Verify system packages
# --------------------------------------------------------------------------- #
info "2/9  Verifying system packages"
python -c "import numpy"       && ok "python-numpy"       || bad "python-numpy"
python -c "import scipy"       && ok "python-scipy"       || bad "python-scipy"
python -c "import onnxruntime" && ok "python-onnxruntime" || bad "python-onnxruntime"
command -v termux-tts-speak >/dev/null && ok "termux-api" || bad "termux-api (install the Termux:API app too)"

# --------------------------------------------------------------------------- #
# 3. Python virtual environment (inherits pkg-installed native modules)
# --------------------------------------------------------------------------- #
info "3/9  Creating Python environment (.venv, --system-site-packages)"
if [ ! -d .venv ]; then
  # --system-site-packages so numpy/scipy/onnxruntime from pkg are visible.
  python -m venv --system-site-packages .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null 2>&1 && ok "venv ready" || bad "venv/pip"

# --------------------------------------------------------------------------- #
# 4. pip packages (pure Python only)
# --------------------------------------------------------------------------- #
info "4/9  Installing pip packages"
pip install -r requirements.txt && ok "requirements.txt" || bad "requirements.txt"
# OpenWakeWord installed WITHOUT deps so pip cannot pull tflite/scipy/onnx builds.
pip install --no-deps openwakeword && ok "openwakeword (--no-deps)" || bad "openwakeword"

# --------------------------------------------------------------------------- #
# 5. whisper.cpp binary + model
# --------------------------------------------------------------------------- #
info "5/9  Setting up whisper.cpp"
if command -v whisper-cli >/dev/null || command -v whisper >/dev/null; then
  ok "whisper.cpp binary present"
else
  echo "  whisper.cpp not found; trying to build from source..."
  if [ ! -d "$HOME/whisper.cpp" ]; then
    git clone --depth 1 https://github.com/ggerganov/whisper.cpp "$HOME/whisper.cpp"
  fi
  ( cd "$HOME/whisper.cpp" && cmake -B build && cmake --build build --config Release -j4 )
  BIN="$HOME/whisper.cpp/build/bin/whisper-cli"
  if [ -x "$BIN" ]; then
    ln -sf "$BIN" "$PREFIX/bin/whisper-cli"
    ok "whisper.cpp built"
  else
    bad "whisper.cpp build (set COSANTA_WHISPER_BINARY manually)"
  fi
fi

MODEL="models/ggml-base.en.bin"
if [ -f "$MODEL" ]; then
  ok "whisper model present ($MODEL)"
else
  echo "  Downloading $MODEL ..."
  curl -L -o "$MODEL" \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin \
    && ok "whisper model downloaded" || bad "whisper model download"
fi

# --------------------------------------------------------------------------- #
# 6. Microphone permission
# --------------------------------------------------------------------------- #
info "6/9  Verifying microphone permission"
if command -v termux-microphone-record >/dev/null; then
  if termux-microphone-record -d -l 1 >/dev/null 2>&1; then
    termux-microphone-record -q >/dev/null 2>&1 || true
    ok "microphone accessible"
  else
    bad "microphone (open Termux:API app and grant mic permission)"
  fi
else
  bad "termux-microphone-record missing (pkg install termux-api)"
fi

# --------------------------------------------------------------------------- #
# 7. Verify OpenWakeWord
# --------------------------------------------------------------------------- #
info "7/9  Verifying OpenWakeWord"
python - <<'PY' && ok "OpenWakeWord imports + downloads models" || bad "OpenWakeWord"
import openwakeword, openwakeword.utils
openwakeword.utils.download_models(model_names=["hey_jarvis"])
from openwakeword.model import Model
Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
PY

# --------------------------------------------------------------------------- #
# 8. Verify STT
# --------------------------------------------------------------------------- #
info "8/9  Verifying speech-to-text (whisper.cpp)"
python - <<'PY' && ok "whisper.cpp STT runs" || bad "whisper.cpp STT"
import numpy as np
from settings import Settings
from speech_to_text import Transcriber
# One second of silence is enough to prove the binary + model load and run.
Transcriber(Settings.from_env()).transcribe(np.zeros(16000, dtype=np.int16))
PY

# --------------------------------------------------------------------------- #
# 9. Verify Groq connection + TTS
# --------------------------------------------------------------------------- #
info "9/9  Verifying Groq + TTS"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
if [ -n "${GROQ_API_KEY:-}" ]; then
  python - <<'PY' && ok "Groq responded" || bad "Groq (check GROQ_API_KEY / network)"
from settings import Settings
from llm import build_llm
print("  Groq says:", build_llm(Settings.from_env()).generate(
    [{"role": "user", "content": "Reply with the single word: ok"}])[:40])
PY
else
  bad "GROQ_API_KEY not set (cp .env.example .env and add your key)"
fi

if command -v termux-tts-speak >/dev/null; then
  termux-tts-speak "Cosanta setup complete." && ok "Android TTS spoke" || bad "Android TTS"
else
  bad "termux-tts-speak missing"
fi

# --------------------------------------------------------------------------- #
echo
echo "=================================================="
echo "  Setup finished:  $PASS passed, $FAIL failed"
echo "  Run Cosanta with:   source .venv/bin/activate && python main.py"
echo "=================================================="
[ "$FAIL" -eq 0 ]
