#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

PROCESS="${EFT_PROCESS:-WBF}"
CONFIG_DIR="${CONFIG_DIR:-configs/$PROCESS}"
EVENT_CONFIG="${EVENT_CONFIG:-$CONFIG_DIR/event_generation.json}"
SAMPLE_CONFIG="${SAMPLE_CONFIG:-$CONFIG_DIR/sample_preparation.json}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$CONFIG_DIR/neural_training.json}"
VALIDATION_CONFIG="${VALIDATION_CONFIG:-$CONFIG_DIR/validation_events_config.json}"
CONSTRAINS_CONFIG="${CONSTRAINS_CONFIG:-$CONFIG_DIR/constrains_config.json}"

read -r -a EXTRA_SBATCH_ARGS <<< "${SBATCH_ARGS:-}"
CPU_SBATCH_ARGS=("${EXTRA_SBATCH_ARGS[@]}")
GPU_SBATCH_ARGS=("${EXTRA_SBATCH_ARGS[@]}")
if [[ -n "${CPU_ACCOUNT:-}" ]]; then
  CPU_SBATCH_ARGS+=("--account=$CPU_ACCOUNT")
fi
if [[ -n "${GPU_ACCOUNT:-}" ]]; then
  GPU_SBATCH_ARGS+=("--account=$GPU_ACCOUNT")
fi

extract_job_id() {
  awk '{print $NF}'
}

echo "[submit] hybrid workflow:"
echo "[submit]   process: $PROCESS ($CONFIG_DIR)"
echo "[submit]   CPU partition: icelake for event generation, sample preparation, validation, constraints"
echo "[submit]   GPU partition: ampere for neural training"

event_job="$(sbatch --parsable "${CPU_SBATCH_ARGS[@]}" "$SCRIPT_DIR/run_event_generation.sbatch" "$EVENT_CONFIG" | extract_job_id)"
echo "[submit] event_generation  icelake  job: $event_job"

sample_job="$(sbatch --parsable --dependency="afterok:$event_job" "${CPU_SBATCH_ARGS[@]}" "$SCRIPT_DIR/run_sample_preparation.sbatch" "$SAMPLE_CONFIG" | extract_job_id)"
echo "[submit] sample_preparation icelake  job: $sample_job afterok:$event_job"

train_job="$(sbatch --parsable --dependency="afterok:$sample_job" "${GPU_SBATCH_ARGS[@]}" "$SCRIPT_DIR/run_neural_training.sbatch" "$TRAIN_CONFIG" | extract_job_id)"
echo "[submit] neural_training    ampere   job: $train_job afterok:$sample_job"

validation_job="$(sbatch --parsable --dependency="afterok:$train_job" "${CPU_SBATCH_ARGS[@]}" "$SCRIPT_DIR/run_validation_events.sbatch" "$VALIDATION_CONFIG" | extract_job_id)"
echo "[submit] validation_events  icelake  job: $validation_job afterok:$train_job"

constrains_job="$(sbatch --parsable --dependency="afterok:$validation_job" "${CPU_SBATCH_ARGS[@]}" "$SCRIPT_DIR/run_constrains.sbatch" "$CONSTRAINS_CONFIG" | extract_job_id)"
echo "[submit] constrains         icelake  job: $constrains_job afterok:$validation_job"
