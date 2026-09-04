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
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
# Account comes from SBATCH_ACCOUNT in the submit environment.
#SBATCH --partition=cpu
#SBATCH --time=02:00:00

set -euo pipefail

# Rebuild the Ray actor environments into an immutable cache under /opt. The
# source image links those environments to /root/.cache/uv, which is external
# to the image and can be purged independently.
: "${BASE_SQSH:?BASE_SQSH is required}"
: "${OUTPUT_SQSH:?OUTPUT_SQSH is required}"
: "${CODE_DIR:?CODE_DIR is required}"

if [[ -e "${OUTPUT_SQSH}" ]]; then
  echo "Refusing to overwrite existing image: ${OUTPUT_SQSH}" >&2
  exit 1
fi

CONTAINER_NAME="sciprobe-nemo-rl-actor-venvs-${SLURM_JOB_ID}"
OUTPUT_TMP="${OUTPUT_SQSH}.tmp-${SLURM_JOB_ID}"
ENROOT_ROOT="/tmp/sciprobe-enroot-actor-venvs-${SLURM_JOB_ID}"

export ENROOT_CACHE_PATH="${ENROOT_ROOT}/cache"
export ENROOT_DATA_PATH="${ENROOT_ROOT}/data"
export ENROOT_RUNTIME_PATH="${ENROOT_ROOT}/runtime"
export ENROOT_TEMP_PATH="${ENROOT_ROOT}/tmp"
mkdir -p \
  "$(dirname "${OUTPUT_SQSH}")" \
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
  "${CONTAINER_NAME}" \
  bash -lc '
    set -euo pipefail
    export PATH=/opt/uv/bin:/opt/nemo_rl_venv/bin:/root/.local/bin:${PATH}
    export UV_PYTHON_INSTALL_DIR=/opt/uv-python
    export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
    export UV_CACHE_DIR=/opt/uv-cache
    export UV_TORCH_BACKEND=cu130
    export UV_LINK_MODE=symlink
    export NEMO_RL_VENV_DIR=/opt/ray_venvs
    export NRL_FORCE_REBUILD_VENVS=true

    mkdir -p "${UV_CACHE_DIR}"
    cd /workspace/RL
    uv lock --check
    uv run --locked --no-sync python -m nemo_rl.utils.prefetch_venvs \
      vllm \
      AsyncTrajectoryCollector \
      ReplayBuffer \
      SyncRolloutActor \
      dtensor_policy_worker_v2

    python_paths=(
      /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python
      /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
      /opt/ray_venvs/nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector/bin/python
      /opt/ray_venvs/nemo_rl.algorithms.async_utils.ReplayBuffer/bin/python
      /opt/ray_venvs/nemo_rl.experience.sync_rollout_actor.SyncRolloutActor/bin/python
      /opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python
    )
    for python_path in "${python_paths[@]}"; do
      "${python_path}" -c "import ray, requests, urllib3; from urllib3.exceptions import HTTPError; print(ray.__version__, urllib3.__version__)"
    done

    test -n "$(find /opt/uv-cache/archive-v0 -mindepth 1 -maxdepth 1 -type d -print -quit)"
    du -sh /opt/uv-cache /opt/ray_venvs
  '

enroot export --output "${OUTPUT_TMP}" "${CONTAINER_NAME}"
mv "${OUTPUT_TMP}" "${OUTPUT_SQSH}"
chmod a+r "${OUTPUT_SQSH}"
ls -lh "${OUTPUT_SQSH}"
