#!/bin/bash
#SBATCH --job-name=extract_prithvi_embeddings
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/extract_embeddings_%j.out
#SBATCH --error=logs/extract_embeddings_%j.err

set -euo pipefail
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -d "${REPO_ROOT}/src" ]; then
  echo "REPO_ROOT=${REPO_ROOT} doesn't look like the repo (no src/ dir). Run sbatch from the repo root." >&2
  exit 1
fi
export SCRATCH_ROOT="${SCRATCH_ROOT:-${REPO_ROOT}}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_cache}"
# Several libraries (matplotlib, misc XDG-aware tools) default to $HOME
# for cache dirs regardless of HF_HOME/PIP_CACHE_DIR -- redirect those too
# (a full $HOME caused a hard failure here before; see
# NEW_PROJECT_MORRILL_CONTEXT.md).
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/xdg_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH_ROOT}/mpl_cache}"
mkdir -p logs "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

source "${CONDA_SH:-/scratch/morrill/users/hmp278/miniforge3/etc/profile.d/conda.sh}"
conda activate "$(cat "${SCRATCH_ROOT}/prithvi_env_path.txt")"

WEIGHTS_DIR="${WEIGHTS_DIR:-${SCRATCH_ROOT}/weights/prithvi.wxc.2300m.v1}"
MERRA2_SURFACE_DIR="${MERRA2_SURFACE_DIR:-${SCRATCH_ROOT}/data/merra2_2024_daily}"
MERRA2_VERTICAL_DIR="${MERRA2_VERTICAL_DIR:-${SCRATCH_ROOT}/data/merra2_2024_daily}"
CLIMATOLOGY_DIR="${CLIMATOLOGY_DIR:-${SCRATCH_ROOT}/data/climatology_regional}"
OUT_PARQUET="${OUT_PARQUET:-${SCRATCH_ROOT}/data/compact/prithvi_embeddings_2024.parquet}"
START="${START:-2024-01-01}"
END="${END:-2024-12-31T23:59:59}"
BATCH_SIZE="${BATCH_SIZE:-4}"

python "${REPO_ROOT}/src/prithvi/extract_embeddings.py" \
  --weights-dir "${WEIGHTS_DIR}" \
  --merra2-surface-dir "${MERRA2_SURFACE_DIR}" \
  --merra2-vertical-dir "${MERRA2_VERTICAL_DIR}" \
  --climatology-surface-dir "${CLIMATOLOGY_DIR}" \
  --climatology-vertical-dir "${CLIMATOLOGY_DIR}" \
  --start "${START}" --end "${END}" \
  --batch-size "${BATCH_SIZE}" \
  --out-parquet "${OUT_PARQUET}"

echo "Job finished: $(date)"
