#!/bin/bash
#SBATCH --job-name=prithvi_sample_inference
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/prithvi_sample_inference_%j.out
#SBATCH --error=logs/prithvi_sample_inference_%j.err

set -euo pipefail

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -d "${REPO_ROOT}/src" ]; then
  echo "REPO_ROOT=${REPO_ROOT} doesn't look like the repo (no src/ dir). Run sbatch from the repo root." >&2
  exit 1
fi
export SCRATCH_ROOT="${SCRATCH_ROOT:-${REPO_ROOT}}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/xdg_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH_ROOT}/mpl_cache}"
mkdir -p logs "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

source "${CONDA_SH:-/scratch/morrill/users/hmp278/miniforge3/etc/profile.d/conda.sh}"
PRITHVI_ENV="$(cat "${SCRATCH_ROOT}/prithvi_env_path.txt")"
conda activate "${PRITHVI_ENV}"
echo "Activated env: ${PRITHVI_ENV}"

# Runs the OFFICIAL Merra2Dataset/PrithviWxC/preproc pipeline (same code the
# notebook uses), but points at the sample MERRA-2/climatology data already
# downloaded in full by 01_setup_prithvi.sbatch, instead of re-running the
# notebook's own hf_hub_download cells -- those point at
# ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M, whose merra-2/climatology dirs
# are empty upstream (confirmed 2026-09-30). See
# src/prithvi/sample_validation.py's docstring for the full story.
WEIGHTS_DIR="${WEIGHTS_DIR:-${SCRATCH_ROOT}/weights/prithvi.wxc.2300m.v1}"

python "${REPO_ROOT}/src/prithvi/sample_validation.py" \
  --weights-dir "${WEIGHTS_DIR}"

echo "Job finished: $(date)"
