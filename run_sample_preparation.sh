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
PROCESS="${EFT_PROCESS:-WBF}"
CONFIG="${1:-configs/$PROCESS/sample_preparation.json}"

exec "$PYTHON_BIN" run_stage.py \
  --name sample_preparation \
  --script scripts/EFT_prepare_samples.py \
  --config "$CONFIG"
