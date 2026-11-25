#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec "$ROOT/build/edgeshieldd" serve "${EDGESHIELD_PORT:-8080}" "${EDGESHIELD_BIND_IP:-127.0.0.1}"
