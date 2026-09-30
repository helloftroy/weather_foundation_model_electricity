#!/bin/bash
# Decide whether to reuse the existing faire-agent conda env or create a
# dedicated env in scratch for Prithvi-WxC, then install Prithvi-WxC into
# whichever is chosen.
#
# Run on the service node (needs internet for pip installs).
set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/morrill/users/hmp278/weather_foundation_model_electricity}"
PRITHVI_REPO_DIR="${PRITHVI_REPO_DIR:-${SCRATCH_ROOT}/Prithvi-WxC}"

# Conda isn't on PATH on the login/compute nodes by default -- source it the
# same way FAIRe_Ocean_Agent/cluster/env_activate.sh does.
CONDA_SH="${CONDA_SH:-/scratch/morrill/users/hmp278/miniforge3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"

# Mirrors the existing FAIRe_Ocean_Agent conda layout: shared conda_envs/ dir
# in scratch, one prefix-based env per project, shared pip/pkg caches.
CONDA_ENVS_ROOT="${CONDA_ENVS_ROOT:-/scratch/morrill/users/hmp278/conda_envs}"
FAIRE_ENV_PREFIX="${FAIRE_ENV_PREFIX:-${CONDA_ENVS_ROOT}/faire-agent}"
PRITHVI_ENV_PREFIX="${PRITHVI_ENV_PREFIX:-${CONDA_ENVS_ROOT}/prithvi-wxc}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CONDA_ENVS_ROOT}/pip_cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${CONDA_ENVS_ROOT}/conda_pkgs}"

echo "=== Existing conda envs ==="
conda env list

echo ""
echo "=== Checking ${FAIRE_ENV_PREFIX} for a compatible Python/torch ==="
FAIRE_OK=0
if [ -d "${FAIRE_ENV_PREFIX}" ] && conda activate "${FAIRE_ENV_PREFIX}" 2>/dev/null; then
  PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "none")"
  TORCH_VER="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "none")"
  echo "faire-agent python=${PY_VER} torch=${TORCH_VER}"
  # Prithvi-WxC needs python>=3.10 and a torch build with CUDA support
  # (cuda.is_available() itself will be False on a login/service node with no
  # GPU -- that's expected and not a disqualifier; check the build tag instead).
  if python -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null && [ "${TORCH_VER}" != "none" ]; then
    FAIRE_OK=1
  fi
  conda deactivate
fi

if [ "${FAIRE_OK}" = "1" ] && [ "${FORCE_NEW_ENV:-0}" != "1" ]; then
  echo ""
  echo "Reusing faire-agent: installing Prithvi-WxC into it."
  TARGET_ENV="${FAIRE_ENV_PREFIX}"
else
  echo ""
  echo "Creating dedicated env at ${PRITHVI_ENV_PREFIX}"
  mkdir -p "${CONDA_ENVS_ROOT}"
  conda create -y -p "${PRITHVI_ENV_PREFIX}" python=3.11
  TARGET_ENV="${PRITHVI_ENV_PREFIX}"
fi

conda activate "${TARGET_ENV}"
actual_prefix="$(python -c 'import sys; print(sys.prefix)')"
if [ "${actual_prefix}" != "${TARGET_ENV}" ]; then
  echo "conda activate '${TARGET_ENV}' did not take effect (got ${actual_prefix})." >&2
  echo "Check for a shell-profile-activated env overriding it, as noted in" >&2
  echo "FAIRe_Ocean_Agent/cluster/env_activate.sh." >&2
  exit 2
fi
echo "Active env: ${actual_prefix}"

mkdir -p "${SCRATCH_ROOT}"
if [ ! -d "${PRITHVI_REPO_DIR}" ]; then
  git clone https://github.com/NASA-IMPACT/Prithvi-WxC "${PRITHVI_REPO_DIR}"
else
  echo "Repo already present at ${PRITHVI_REPO_DIR}; pulling latest develop."
  git -C "${PRITHVI_REPO_DIR}" pull
fi

pip install -e "${PRITHVI_REPO_DIR}[examples]"

echo "${TARGET_ENV}" > "${SCRATCH_ROOT}/prithvi_env_path.txt"
echo ""
echo "Done. TARGET_ENV=${TARGET_ENV}"
echo "Recorded at ${SCRATCH_ROOT}/prithvi_env_path.txt -- later sbatch scripts read this to activate the same env."
