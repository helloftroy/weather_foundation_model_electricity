#!/bin/bash
# Decide whether to reuse the existing FAIRe conda env or create a dedicated
# env in scratch for Prithvi-WxC, then install Prithvi-WxC into whichever is chosen.
#
# Run on the service node (needs internet for pip installs).
set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/morrill/users/hmp278/weather_electricity_foundation}"
ENV_ROOT="${SCRATCH_ROOT}/envs"
PRITHVI_ENV_NAME="${PRITHVI_ENV_NAME:-prithvi_wxc}"
FAIRE_ENV_NAME="${FAIRE_ENV_NAME:-FAIRe}"
PRITHVI_REPO_DIR="${PRITHVI_REPO_DIR:-${SCRATCH_ROOT}/Prithvi-WxC}"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== Existing conda envs ==="
conda env list

echo ""
echo "=== Checking ${FAIRE_ENV_NAME} for a compatible Python/torch ==="
FAIRE_OK=0
if conda activate "${FAIRE_ENV_NAME}" 2>/dev/null; then
  PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "none")"
  TORCH_VER="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "none")"
  echo "FAIRe python=${PY_VER} torch=${TORCH_VER}"
  # Prithvi-WxC needs python>=3.10 and a reasonably recent torch with CUDA support.
  if python -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null && [ "${TORCH_VER}" != "none" ]; then
    FAIRE_OK=1
  fi
  conda deactivate
fi

if [ "${FAIRE_OK}" = "1" ] && [ "${FORCE_NEW_ENV:-0}" != "1" ]; then
  echo ""
  echo "Reusing ${FAIRE_ENV_NAME}: installing Prithvi-WxC into it."
  TARGET_ENV="${FAIRE_ENV_NAME}"
else
  echo ""
  echo "Creating dedicated env ${PRITHVI_ENV_NAME} in scratch: ${ENV_ROOT}/${PRITHVI_ENV_NAME}"
  mkdir -p "${ENV_ROOT}"
  conda create -y -p "${ENV_ROOT}/${PRITHVI_ENV_NAME}" python=3.11
  TARGET_ENV="${ENV_ROOT}/${PRITHVI_ENV_NAME}"
fi

conda activate "${TARGET_ENV}"
echo "Active env: $(python -c 'import sys; print(sys.prefix)')"

mkdir -p "${SCRATCH_ROOT}"
if [ ! -d "${PRITHVI_REPO_DIR}" ]; then
  git clone https://github.com/NASA-IMPACT/Prithvi-WxC "${PRITHVI_REPO_DIR}"
else
  echo "Repo already present at ${PRITHVI_REPO_DIR}; pulling latest develop."
  git -C "${PRITHVI_REPO_DIR}" pull
fi

pip install --cache-dir "${SCRATCH_ROOT}/pip_cache" -e "${PRITHVI_REPO_DIR}[examples]"

echo "${TARGET_ENV}" > "${SCRATCH_ROOT}/prithvi_env_path.txt"
echo ""
echo "Done. TARGET_ENV=${TARGET_ENV}"
echo "Recorded at ${SCRATCH_ROOT}/prithvi_env_path.txt -- later sbatch scripts read this to activate the same env."
