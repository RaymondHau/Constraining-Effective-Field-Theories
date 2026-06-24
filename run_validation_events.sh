#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROCESS="${EFT_PROCESS:-WBF}"
CONFIG="${1:-configs/$PROCESS/validation_events_config.json}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/validation_events.py" --config "$CONFIG"
