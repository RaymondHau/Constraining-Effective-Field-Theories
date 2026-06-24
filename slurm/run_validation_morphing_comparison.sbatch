#!/usr/bin/env bash
#SBATCH --job-name=eft_validate_morphing
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=icelake
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rs2398@cam.ac.uk
#SBATCH --account=mphil-dis-sl2-cpu

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
export EFT_LOG_DIR="${EFT_LOG_DIR:-$HOME/eft_slurm_logs/${SLURM_JOB_NAME:-eft}/${SLURM_JOB_ID:-manual}}"
mkdir -p "$EFT_LOG_DIR"

PROCESS="${EFT_PROCESS:-WBF}"
VALIDATION_CONFIG="${1:-configs/$PROCESS/validation_events_config.json}"
SAMPLE_CONFIG="${2:-configs/$PROCESS/sample_preparation.json}"
exec bash ./run_validation_morphing_comparison.sh \
  --validation-config "$VALIDATION_CONFIG" \
  --sample-config "$SAMPLE_CONFIG"
