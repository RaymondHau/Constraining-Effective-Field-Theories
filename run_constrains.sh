#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ $# -gt 0 && "$1" != --* ]]; then
  exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/constrains.py" --config "$1" "${@:2}"
else
  exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/constrains.py" "$@"
fi
