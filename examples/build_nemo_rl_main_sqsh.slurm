#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH --nodes=1
#SBATCH --exclusive
# Account comes from SBATCH_ACCOUNT in the submit environment.
#SBATCH --partition=cpu
#SBATCH --time=04:00:00

set -euo pipefail

# Rebase the public NeMo-RL release image onto the dependencies locked by the
# mounted source checkout. This is a CPU-only, registry-free substitute for the
# Docker build described in the Lightning guide. Architecture follows the build host.
: "${BASE_SQSH:?BASE_SQSH is required}"
: "${OUTPUT_SQSH:?OUTPUT_SQSH is required}"
: "${CODE_DIR:?CODE_DIR is required}"
: "${UV_CACHE_DIR_OVERRIDE:?UV_CACHE_DIR_OVERRIDE is required}"

if [[ -e "${OUTPUT_SQSH}" ]]; then
  echo "Refusing to overwrite existing image: ${OUTPUT_SQSH}" >&2
  exit 1
fi

CONTAINER_NAME="sciprobe-nemo-rl-main-${SLURM_JOB_ID}"
OUTPUT_TMP="${OUTPUT_SQSH}.tmp-${SLURM_JOB_ID}"
ENROOT_ROOT="/tmp/sciprobe-enroot-${SLURM_JOB_ID}"

export ENROOT_CACHE_PATH="${ENROOT_ROOT}/cache"
export ENROOT_DATA_PATH="${ENROOT_ROOT}/data"
export ENROOT_RUNTIME_PATH="${ENROOT_ROOT}/runtime"
export ENROOT_TEMP_PATH="${ENROOT_ROOT}/tmp"
mkdir -p \
  "$(dirname "${OUTPUT_SQSH}")" \
  "${UV_CACHE_DIR_OVERRIDE}" \
  "${ENROOT_CACHE_PATH}" \
  "${ENROOT_DATA_PATH}" \
  "${ENROOT_RUNTIME_PATH}" \
  "${ENROOT_TEMP_PATH}"

cleanup() {
  enroot remove -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "base_sqsh=${BASE_SQSH}"
echo "output_sqsh=${OUTPUT_SQSH}"
echo "code_dir=${CODE_DIR}"
uname -m
enroot version
df -h /tmp

enroot create --name "${CONTAINER_NAME}" "${BASE_SQSH}"

enroot start \
  --root \
  --rw \
  --mount "${CODE_DIR}:/workspace/RL" \
  --mount "${UV_CACHE_DIR_OVERRIDE}:/root/.cache/uv" \
  "${CONTAINER_NAME}" \
  bash -lc '
    set -euo pipefail
    export PATH=/opt/uv/bin:/opt/nemo_rl_venv/bin:/root/.local/bin:${PATH}
    export UV_INSTALL_DIR=/opt/uv/bin
    export UV_PYTHON_INSTALL_DIR=/opt/uv-python
    export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
    export UV_CACHE_DIR=/root/.cache/uv
    export NEMO_RL_VENV_DIR=/opt/ray_venvs
    export NRL_FORCE_REBUILD_VENVS=true

    mkdir -p "${UV_INSTALL_DIR}" "${UV_PYTHON_INSTALL_DIR}"
    curl -LsSf https://astral.sh/uv/0.11.28/install.sh | \
      env UV_INSTALL_DIR="${UV_INSTALL_DIR}" UV_NO_MODIFY_PATH=1 sh
    hash -r
    uv --version
    uv python install 3.13.14
    uv venv --clear --seed --python 3.13.14 /opt/nemo_rl_venv
    ln -sfn /opt/uv/bin/uv /usr/local/bin/uv

    cd /workspace/RL
    uv lock --check

    # Match the upstream Dockerfile: warm the two backends used by this
    # canary, then restore the driver environment.
    UV_LINK_MODE=copy uv sync --frozen --no-install-project
    UV_LINK_MODE=copy uv sync --frozen --extra vllm --no-install-project
    UV_LINK_MODE=copy uv sync --frozen --extra automodel --no-install-project
    UV_LINK_MODE=copy uv sync --frozen --no-install-project

    # Freeze only the worker environments used by colocated AutoModel + vLLM.
    UV_LINK_MODE=symlink uv run --locked python \
      nemo_rl/utils/prefetch_venvs.py \
      vllm dtensor_policy_worker_v2

    uv run --locked python tools/generate_fingerprint.py \
      > /opt/nemo_rl_container_fingerprint

    python --version
    python -c "import ray; print(f\"ray={ray.__version__}\")"
    test -x /usr/local/bin/python-VllmGenerationWorker
    test -x /usr/local/bin/python-DTensorPolicyWorkerV2
  '

enroot export --output "${OUTPUT_TMP}" "${CONTAINER_NAME}"
mv "${OUTPUT_TMP}" "${OUTPUT_SQSH}"
chmod a+r "${OUTPUT_SQSH}"
ls -lh "${OUTPUT_SQSH}"
