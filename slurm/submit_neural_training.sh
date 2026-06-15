#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-configs/paper_10m/neural_training.json}"
read -r -a EXTRA_SBATCH_ARGS <<< "${SBATCH_ARGS:-}"
if [[ -n "${ACCOUNT:-}" ]]; then
  EXTRA_SBATCH_ARGS+=("--account=$ACCOUNT")
fi

sbatch "${EXTRA_SBATCH_ARGS[@]}" "$SCRIPT_DIR/run_neural_training.sbatch" "$CONFIG"
