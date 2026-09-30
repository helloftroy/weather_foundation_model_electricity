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

export SCRATCH_ROOT="${SCRATCH_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_cache}"
mkdir -p logs

source "${CONDA_SH:-/scratch/morrill/users/hmp278/miniforge3/etc/profile.d/conda.sh}"
PRITHVI_ENV="$(cat "${SCRATCH_ROOT}/prithvi_env_path.txt")"
conda activate "${PRITHVI_ENV}"
echo "Activated env: ${PRITHVI_ENV}"

PRITHVI_REPO_DIR="${PRITHVI_REPO_DIR:-${SCRATCH_ROOT}/Prithvi-WxC}"
cd "${PRITHVI_REPO_DIR}"

# Run the repository's own basic-inference notebook as the official small/
# sample validation test (downloads its own sample data + weights on first
# run if not already cached under HF_HOME). This is the "official
# basic-inference and validation workflow" from Phase 1 step 2 -- do not
# replace with custom inference code at this stage.
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 \
  --output "prithvi_sample_inference_executed_${SLURM_JOB_ID}.ipynb" \
  examples/PrithviWxC_inference.ipynb

echo "Executed notebook saved in ${PRITHVI_REPO_DIR}/examples/"
echo "Job finished: $(date)"
