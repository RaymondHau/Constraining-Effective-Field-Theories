#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"
PROCESS="${EFT_PROCESS:-WBF}"
VALIDATION_CONFIG="${1:-configs/$PROCESS/validation_events_config.json}"
SAMPLE_CONFIG="${2:-configs/$PROCESS/sample_preparation.json}"
read -r -a EXTRA_SBATCH_ARGS <<< "${SBATCH_ARGS:-}"
if [[ -n "${ACCOUNT:-}" ]]; then
  EXTRA_SBATCH_ARGS+=("--account=$ACCOUNT")
fi

sbatch "${EXTRA_SBATCH_ARGS[@]}" "$SCRIPT_DIR/run_validation_morphing_comparison.sbatch" \
  "$VALIDATION_CONFIG" "$SAMPLE_CONFIG"
