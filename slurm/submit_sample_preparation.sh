#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-configs/paper_10m/sample_preparation.json}"
read -r -a EXTRA_SBATCH_ARGS <<< "${SBATCH_ARGS:-}"
if [[ -n "${ACCOUNT:-}" ]]; then
  EXTRA_SBATCH_ARGS+=("--account=$ACCOUNT")
fi

sbatch "${EXTRA_SBATCH_ARGS[@]}" "$SCRIPT_DIR/run_sample_preparation.sbatch" "$CONFIG"
