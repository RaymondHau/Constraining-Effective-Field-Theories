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

exec "$PYTHON_BIN" scripts/compare_validation_morphing.py "$@"
