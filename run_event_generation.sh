#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi
CONFIG="${1:-configs/paper_10m/event_generation.json}"

exec "$PYTHON_BIN" run_stage.py \
  --name event_generation \
  --script scripts/EFT_event_generation.py \
  --config "$CONFIG"
