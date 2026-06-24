#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"
PROCESS="${EFT_PROCESS:-WBF}"
CONFIG="${1:-configs/$PROCESS/event_generation.json}"
read -r -a EXTRA_SBATCH_ARGS <<< "${SBATCH_ARGS:-}"
if [[ -n "${ACCOUNT:-}" ]]; then
  EXTRA_SBATCH_ARGS+=("--account=$ACCOUNT")
fi

sbatch "${EXTRA_SBATCH_ARGS[@]}" "$SCRIPT_DIR/run_event_generation.sbatch" "$CONFIG"
