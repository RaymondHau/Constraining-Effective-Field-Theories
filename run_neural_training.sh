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
CONFIG="${1:-configs/paper_10m/neural_training.json}"

exec "$PYTHON_BIN" run_stage.py \
  --name neural_training \
  --script scripts/EFT_train_estimators.py \
  --config "$CONFIG"
