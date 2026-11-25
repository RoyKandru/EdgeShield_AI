#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PID_FILE="/tmp/edgeshield.pid"
LOG_FILE="/tmp/edgeshield_board.log"

HOST="${EDGESHIELD_HOST:-0.0.0.0}"
PORT="${EDGESHIELD_PORT:-8000}"

# Pick main Python interpreter
if [ -n "${PYTHON_BIN:-}" ]; then
  PY="$PYTHON_BIN"
elif [ -x /home/open/miniforge3/bin/python3 ]; then
  PY="/home/open/miniforge3/bin/python3"
else
  PY="$(command -v python3)"
fi

# Auto-configure RKNN worker Python if not explicitly set
if [ -z "${EDGESHIELD_RKNN_PYTHON:-}" ] && [ -x /home/open/miniforge3/envs/rknn-python/bin/python ]; then
  export EDGESHIELD_RKNN_PYTHON="/home/open/miniforge3/envs/rknn-python/bin/python"
fi

configure_env() {
  local model_path="${EDGESHIELD_EXECUTOR_MODEL_PATH:-}"
  if [ -z "$model_path" ]; then
    if [ -e "$ROOT/models/llama.pte" ]; then
      model_path="$ROOT/models/llama.pte"
    elif [ -e "$ROOT/models/llama_3_2_1b_xnnpack.pte" ]; then
      model_path="$ROOT/models/llama_3_2_1b_xnnpack.pte"
    fi
  fi

  local tok_path="${EDGESHIELD_EXECUTOR_TOKENIZER_PATH:-}"
  if [ -z "$tok_path" ]; then
    if [ -e "$ROOT/models/tokenizer.model" ]; then
      tok_path="$ROOT/models/tokenizer.model"
    elif [ -e "$ROOT/models/tokenizer.json" ]; then
      tok_path="$ROOT/models/tokenizer.json"
    fi
  fi

  if [ -x "$ROOT/bin/executorch_runner" ] && [ -n "$model_path" ] && [ -e "$model_path" ] && [ -n "$tok_path" ] && [ -e "$tok_path" ]; then
    export EDGESHIELD_EXECUTOR="${EDGESHIELD_EXECUTOR:-command}"
    export EDGESHIELD_EXECUTOR_COMMAND="${EDGESHIELD_EXECUTOR_COMMAND:-$ROOT/bin/executorch_runner}"
    export EDGESHIELD_EXECUTOR_MODEL_PATH="$model_path"
    export EDGESHIELD_EXECUTOR_TOKENIZER_PATH="$tok_path"
    export EDGESHIELD_EXECUTOR_DAEMON="${EDGESHIELD_EXECUTOR_DAEMON:-true}"
  else
    export EDGESHIELD_EXECUTOR="${EDGESHIELD_EXECUTOR:-template}"
    echo "INFO: Llama ExecuTorch model/tokenizer not found under $ROOT/models — using template advisory fallback."
  fi

  export EDGESHIELD_LIVE_MONITOR_ENABLED="${EDGESHIELD_LIVE_MONITOR_ENABLED:-true}"
  export EDGESHIELD_LIVE_MONITOR_INTERFACE="${EDGESHIELD_LIVE_MONITOR_INTERFACE:-auto}"

  # Helpful hint if bin/live_sniffer lacks raw capture capability (e.g. after unzipping)
  local getcap_bin=""
  if command -v getcap >/dev/null 2>&1; then
    getcap_bin="$(command -v getcap)"
  elif [ -x /usr/sbin/getcap ]; then
    getcap_bin="/usr/sbin/getcap"
  fi
  if [ -x "$ROOT/bin/live_sniffer" ] && [ -n "$getcap_bin" ]; then
    if ! "$getcap_bin" "$ROOT/bin/live_sniffer" 2>/dev/null | grep -q "cap_net_raw"; then
      echo "WARN: bin/live_sniffer lacks cap_net_raw. For live NIC capture run:"
      echo "      sudo setcap cap_net_raw,cap_net_admin=eip $ROOT/bin/live_sniffer"
    fi
  fi
}

usage() {
  cat <<EOF
Usage: ./run.sh [command]

Commands:
  start      Start EdgeShield AI in background (default)
  fg         Start EdgeShield AI in foreground (Ctrl+C to stop)
  stop       Stop background EdgeShield AI
  restart    Restart EdgeShield AI
  status     Show running status, health, and 3-core NPU status
  logs       Follow $LOG_FILE

Environment overrides:
  EDGESHIELD_HOST=$HOST   EDGESHIELD_PORT=$PORT
  EDGESHIELD_RKNN_PYTHON=${EDGESHIELD_RKNN_PYTHON:-<auto>}
  EDGESHIELD_LIVE_MONITOR_INTERFACE=${EDGESHIELD_LIVE_MONITOR_INTERFACE:-auto}
  PYTHON_BIN=$PY
EOF
}

is_running() {
  [ -f "$PID_FILE" ] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

stop_service() {
  if is_running; then
    local pid
    pid="$(cat "$PID_FILE")"
    # Send SIGTERM to process group (setsid) so child NPU/ExecuTorch/sniffer workers exit cleanly
    kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.3
    done
    kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "EdgeShield AI stopped (pid=$pid)."
  else
    rm -f "$PID_FILE"
    echo "EdgeShield AI is not running."
  fi
}

show_status() {
  if is_running; then
    local pid
    pid="$(cat "$PID_FILE")"
    echo "EdgeShield AI RUNNING (pid=$pid) -> http://$HOST:$PORT"
    "$PY" - <<PY 2>/dev/null || true
import json, urllib.request
base = "http://127.0.0.1:$PORT"
for path in ("/api/health", "/api/hardware/rknn/status", "/api/network/monitor/status"):
    with urllib.request.urlopen(base + path, timeout=3) as r:
        d = json.loads(r.read().decode())
    if path == "/api/health":
        print("  Health  :", d)
    elif path == "/api/hardware/rknn/status":
        print("  NPU     :", d.get("cores"))
    else:
        print("  Monitor :", {"running": d.get("running"), "interface": d.get("interface")})
PY
  else
    echo "EdgeShield AI STOPPED"
    return 1
  fi
}

start_bg() {
  if is_running; then
    stop_service
  fi
  configure_env

  setsid "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT" > "$LOG_FILE" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PID_FILE"

  for _ in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: EdgeShield AI exited unexpectedly. Recent logs:" >&2
      tail -n 40 "$LOG_FILE" >&2 || true
      rm -f "$PID_FILE"
      exit 1
    fi
    if "$PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/api/health', timeout=1).read(1)" >/dev/null 2>&1; then
      local ip
      ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
      echo "EdgeShield AI STARTED"
      echo "  PID       : $pid"
      echo "  Dashboard : http://${ip:-127.0.0.1}:$PORT (listening on $HOST:$PORT)"
      echo "  Advisory  : $EDGESHIELD_EXECUTOR"
      echo "  Python    : $PY"
      echo "  Logs      : $LOG_FILE"
      return 0
    fi
    sleep 0.5
  done

  echo "EdgeShield AI started (pid=$pid) — still warming up. Check logs: $LOG_FILE"
}

start_fg() {
  if is_running; then
    stop_service
  fi
  configure_env
  echo "Starting EdgeShield AI in foreground on http://$HOST:$PORT (advisory=$EDGESHIELD_EXECUTOR)..."
  exec "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT"
}

CMD="${1:-start}"
case "$CMD" in
  start)
    start_bg
    ;;
  fg|foreground)
    start_fg
    ;;
  stop)
    stop_service
    ;;
  restart)
    stop_service
    start_bg
    ;;
  status)
    show_status
    ;;
  logs)
    touch "$LOG_FILE"
    exec tail -f "$LOG_FILE"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: $CMD" >&2
    usage >&2
    exit 1
    ;;
esac
