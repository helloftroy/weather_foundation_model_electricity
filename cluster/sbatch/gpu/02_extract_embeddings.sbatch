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

# Show Python output in the log as it happens instead of in delayed chunks.
export PYTHONUNBUFFERED=1

# Year to process. Override at submit time: sbatch --export=ALL,YEAR=2023 ...
YEAR="${YEAR:-2024}"
echo "Year: ${YEAR}"
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
# GRID=global (default) runs on the full MERRA-2 grid, which is the only
# setup in which the model performs properly (see sample_validation.py).
# GRID=regional is the old 30-60N, 90-50W crop, kept for comparison.
GRID="${GRID:-global}"
if [ "${GRID}" = "global" ]; then
  DAILY_DIR="${SCRATCH_ROOT}/data/merra2_global_${YEAR}_daily"
  # 2020 ships ready-made with the model weights; use it unless we built our own.
  if [ ! -d "${DAILY_DIR}" ] && [ "${YEAR}" = "2020" ]; then DAILY_DIR="${WEIGHTS_DIR}/merra-2"; fi
  DEFAULT_CLIMATOLOGY="${WEIGHTS_DIR}/climatology"
else
  DAILY_DIR="${SCRATCH_ROOT}/data/merra2_${YEAR}_daily"
  DEFAULT_CLIMATOLOGY="${SCRATCH_ROOT}/data/climatology_regional"
fi
echo "Grid: ${GRID}; daily MERRA-2 files: ${DAILY_DIR}"
MERRA2_SURFACE_DIR="${MERRA2_SURFACE_DIR:-${DAILY_DIR}}"
MERRA2_VERTICAL_DIR="${MERRA2_VERTICAL_DIR:-${DAILY_DIR}}"
CLIMATOLOGY_DIR="${CLIMATOLOGY_DIR:-${DEFAULT_CLIMATOLOGY}}"
# Extra encoder taps, each another ~0.85 GB per year: "embed" and/or block
# indices that are multiples of 4, e.g. LAYERS="embed 0 12". Empty = none.
LAYERS="${LAYERS:-}"
# Hours ahead the model is asked to describe (0 = the embedding's own time).
# Each sample is issued at its timestamp (the latest input time) and
# describes timestamp + LEAD_TIME. A non-zero lead gets its own file names,
# so it never overwrites the lead-0 outputs.
LEAD_TIME="${LEAD_TIME:-0}"
SUFFIX=""
if [ "${LEAD_TIME}" != "0" ]; then SUFFIX="_lead${LEAD_TIME}"; fi
# REGION=ercot: Texas instead of New England -- token box over ERCOT, ERCOT pooled extents, a wide box around
# Texas, and output names prithvi_*_tx_* so nothing of the New England run is overwritten.
REGION="${REGION:-ne}"
if [ "${REGION}" = "ercot" ]; then
  TAG="tx"; TOKEN_BOX="${TOKEN_BOX:-26 37 -106.25 -93.75}"; POOL_SET="ercot"; WIDE_BOX="${WIDE_BOX:-24 40 -110 -88}"
else
  TAG="ne"; TOKEN_BOX="${TOKEN_BOX:-}"; POOL_SET="ne"
fi
PARQUET_TAG=""
if [ "${REGION}" = "ercot" ]; then PARQUET_TAG="_tx"; fi
OUT_PARQUET="${OUT_PARQUET:-${SCRATCH_ROOT}/data/compact/prithvi_embeddings_${YEAR}${SUFFIX}${PARQUET_TAG}.parquet}"
# Un-pooled patch tokens over New England (about 0.85 GB per year), plus the
# model's normalised inputs (~0.2 GB) and decoder output for the same box.
OUT_TOKENS="${OUT_TOKENS:-${SCRATCH_ROOT}/data/compact/prithvi_tokens_${TAG}_${YEAR}${SUFFIX}.npy}"
START="${START:-${YEAR}-01-01}"
END="${END:-${YEAR}-12-31T23:59:59}"
BATCH_SIZE="${BATCH_SIZE:-2}"
# Optional larger box for regional weather features, "LAT_MIN LAT_MAX LON_MIN LON_MAX",
# e.g. WIDE_BOX="32 50 -90 -64" (Northeast, mid-Atlantic, Ohio valley). Empty = off.
WIDE_BOX="${WIDE_BOX:-}"
# POOLED=1: also save pooled tokens / inputs / decoder output per extent (small; see --pooled).
# WIDE_PATCH_TOKENS=1: also save every patch token over the wide box (~5-6 GB per year).
POOLED="${POOLED:-}"
WIDE_PATCH_TOKENS="${WIDE_PATCH_TOKENS:-}"

python "${REPO_ROOT}/src/prithvi/extract_embeddings.py" \
  --weights-dir "${WEIGHTS_DIR}" \
  --merra2-surface-dir "${MERRA2_SURFACE_DIR}" \
  --merra2-vertical-dir "${MERRA2_VERTICAL_DIR}" \
  --climatology-surface-dir "${CLIMATOLOGY_DIR}" \
  --climatology-vertical-dir "${CLIMATOLOGY_DIR}" \
  --start "${START}" --end "${END}" \
  --batch-size "${BATCH_SIZE}" \
  --out-parquet "${OUT_PARQUET}" \
  --out-tokens "${OUT_TOKENS}" \
  --lead-time "${LEAD_TIME}" \
  --layers ${LAYERS} \
  ${WIDE_BOX:+--wide-box ${WIDE_BOX}} \
  ${POOLED:+--pooled} \
  --pool-set "${POOL_SET}" \
  ${TOKEN_BOX:+--token-box ${TOKEN_BOX}} \
  ${WIDE_PATCH_TOKENS:+--save-wide-patch-tokens}

echo "Job finished: $(date)"
