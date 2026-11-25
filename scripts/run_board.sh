#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Cleanly stop previous EdgeShield instance using PID file if present
if [ -f /tmp/edgeshield.pid ]; then
  OLD_PID=$(cat /tmp/edgeshield.pid || true)
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
  fi
fi

# Pick Python interpreter (prefer explicit PYTHON_BIN, then board miniforge3, then PATH python3)
if [ -n "${PYTHON_BIN:-}" ]; then
  PY="$PYTHON_BIN"
elif [ -x /home/open/miniforge3/bin/python3 ]; then
  PY="/home/open/miniforge3/bin/python3"
else
  PY="$(command -v python3)"
fi

# Auto-detect Llama ExecuTorch artifacts if present; otherwise fall back cleanly to template advisory
MODEL_PATH="${EDGESHIELD_EXECUTOR_MODEL_PATH:-}"
if [ -z "$MODEL_PATH" ]; then
  if [ -e "$ROOT/models/llama.pte" ]; then
    MODEL_PATH="$ROOT/models/llama.pte"
  elif [ -e "$ROOT/models/llama_3_2_1b_xnnpack.pte" ]; then
    MODEL_PATH="$ROOT/models/llama_3_2_1b_xnnpack.pte"
  fi
fi

TOK_PATH="${EDGESHIELD_EXECUTOR_TOKENIZER_PATH:-}"
if [ -z "$TOK_PATH" ]; then
  if [ -e "$ROOT/models/tokenizer.model" ]; then
    TOK_PATH="$ROOT/models/tokenizer.model"
  elif [ -e "$ROOT/models/tokenizer.json" ]; then
    TOK_PATH="$ROOT/models/tokenizer.json"
  fi
fi

if [ -x "$ROOT/bin/executorch_runner" ] && [ -n "$MODEL_PATH" ] && [ -e "$MODEL_PATH" ] && [ -n "$TOK_PATH" ] && [ -e "$TOK_PATH" ]; then
  export EDGESHIELD_EXECUTOR="${EDGESHIELD_EXECUTOR:-command}"
  export EDGESHIELD_EXECUTOR_COMMAND="${EDGESHIELD_EXECUTOR_COMMAND:-$ROOT/bin/executorch_runner}"
  export EDGESHIELD_EXECUTOR_MODEL_PATH="$MODEL_PATH"
  export EDGESHIELD_EXECUTOR_TOKENIZER_PATH="$TOK_PATH"
  export EDGESHIELD_EXECUTOR_DAEMON="${EDGESHIELD_EXECUTOR_DAEMON:-true}"
else
  export EDGESHIELD_EXECUTOR="${EDGESHIELD_EXECUTOR:-template}"
  echo "INFO: Llama ExecuTorch model/tokenizer not found under $ROOT/models — starting with template advisory fallback."
fi

export EDGESHIELD_LIVE_MONITOR_ENABLED="${EDGESHIELD_LIVE_MONITOR_ENABLED:-true}"
export EDGESHIELD_LIVE_MONITOR_INTERFACE="${EDGESHIELD_LIVE_MONITOR_INTERFACE:-auto}"

HOST="${EDGESHIELD_HOST:-0.0.0.0}"
PORT="${EDGESHIELD_PORT:-8000}"

setsid "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT" > /tmp/edgeshield_board.log 2>&1 < /dev/null &
NEW_PID=$!
echo "$NEW_PID" > /tmp/edgeshield.pid
sleep 2
if kill -0 "$NEW_PID" 2>/dev/null; then
  echo "EDGESHIELD_STARTED pid=$NEW_PID url=http://$HOST:$PORT executor=$EDGESHIELD_EXECUTOR python=$PY"
else
  echo "EDGESHIELD_FAILED"
  tail -n 40 /tmp/edgeshield_board.log
  exit 1
fi
