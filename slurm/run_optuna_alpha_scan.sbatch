#!/usr/bin/env bash
#SBATCH --job-name=eft_optuna_alpha
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=ampere
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rs2398@cam.ac.uk
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  if [[ "${MODULE_PURGE:-1}" == "1" ]]; then
    module purge
  fi
  for module_name in ${MODULES:-python/3.11.0-icl}; do
    module load "$module_name"
  done
fi

VENV="${VENV:-$HOME/eft_venv}"
if [[ -d "$VENV" ]]; then
  source "$VENV/bin/activate"
elif [[ -n "${CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
else
  echo "Virtual environment not found at $VENV. Run: bash slurm/setup_environment.sh" >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

PROCESS="${EFT_PROCESS:-WBF}"
CONFIG="${1:-configs/$PROCESS/neural_training.json}"
TRIALS_PER_METHOD="${2:-20}"
SCAN_EPOCHS="${3:-100}"
OUTPUT_ROOT="${OPTUNA_OUTPUT_DIR:-table_outputs/optuna_alpha_scan}"

METHODS=(RASCAL CASCAL ALICES)
METHOD="${METHODS[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}]}"
METHOD_LOWER="${METHOD,,}"

echo "[optuna] array task ${SLURM_ARRAY_TASK_ID}: scanning ${METHOD} on ${CUDA_VISIBLE_DEVICES:-assigned GPU}"

exec python -u scripts/optuna_alpha_scan.py \
  --config "$CONFIG" \
  --methods "$METHOD" \
  --output-dir "$OUTPUT_ROOT/$METHOD_LOWER" \
  --trials-per-method "$TRIALS_PER_METHOD" \
  --epochs "$SCAN_EPOCHS"
