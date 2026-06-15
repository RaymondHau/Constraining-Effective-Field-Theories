#!/usr/bin/env bash
#SBATCH --job-name=eft_neural_training
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=ampere
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rs2398@cam.ac.uk
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --gres=gpu:1
## For CPU-only training, use --partition=icelake, remove --gres, and consider --cpus-per-task=8.

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

if [[ -n "${VENV:-}" ]]; then
  source "$VENV/bin/activate"
elif [[ -n "${CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export EFT_LOG_DIR="${EFT_LOG_DIR:-$HOME/eft_slurm_logs/${SLURM_JOB_NAME:-eft}/${SLURM_JOB_ID:-manual}}"
mkdir -p "$EFT_LOG_DIR"

CONFIG="${1:-configs/paper_10m/neural_training.json}"
exec bash ./run_neural_training.sh "$CONFIG"
