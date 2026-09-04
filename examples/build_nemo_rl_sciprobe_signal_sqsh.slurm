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

# Add the custom SciProbe reward-server venv to the proven Lightning/ns_tools
# image. GPU actor environments and the existing Gym venvs remain unchanged.
: "${BASE_SQSH:?BASE_SQSH is required}"
: "${OUTPUT_SQSH:?OUTPUT_SQSH is required}"
: "${CODE_DIR:?CODE_DIR is required}"

if [[ -e "${OUTPUT_SQSH}" ]]; then
  echo "Refusing to overwrite existing image: ${OUTPUT_SQSH}" >&2
  exit 1
fi

CONTAINER_NAME="sciprobe-nemo-rl-signal-${SLURM_JOB_ID}"
OUTPUT_TMP="${OUTPUT_SQSH}.tmp-${SLURM_JOB_ID}"
ENROOT_ROOT="/tmp/sciprobe-enroot-signal-${SLURM_JOB_ID}"

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
    export PATH=/opt/uv/bin:/opt/nemo_rl_venv/bin:${PATH}
    export PYTHONPATH=/workspace/RL:${PYTHONPATH:-}
    export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
    export UV_PYTHON_INSTALL_DIR=/opt/uv-python
    export UV_CACHE_DIR=/opt/uv-cache
    export UV_LINK_MODE=hardlink
    export NEMO_GYM_VENV_DIR=/opt/gym_venvs
    export NEMO_GYM_EXTRA_ROOTS=/workspace/RL/examples/nemo_gym_extensions
    export NRL_FORCE_REBUILD_VENVS=false

    cd /workspace/RL
    uv lock --check
    gym_cli=/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/gym
    test -x "${gym_cli}"
    "${gym_cli}" env prefetch \
      --search-dir "${NEMO_GYM_EXTRA_ROOTS}" \
      --config responses_api_models/vllm_model/configs/vllm_model_for_training.yaml \
      --config responses_api_agents/sciprobe_simple_agent/configs/sciprobe_simple_agent.yaml \
      --config resources_servers/sciprobe_checks/configs/sciprobe_checks.yaml \
      --config resources_servers/sciprobe_ns_tools/configs/sciprobe_ns_tools.yaml \
      +policy_model_name=dummy-model \
      +policy_base_url=http://127.0.0.1:8000/v1 \
      +policy_api_key=dummy \
      +uv_cache_dir="${UV_CACHE_DIR}" \
      +uv_venv_dir="${NEMO_GYM_VENV_DIR}" \
      +skip_venv_if_present=true

    test -d /opt/gym_venvs
    test -n "$(find /opt/gym_venvs -mindepth 1 -maxdepth 1 -type d -print -quit)"
    for server_venv in \
      responses_api_models/vllm_model \
      responses_api_agents/sciprobe_simple_agent \
      resources_servers/sciprobe_checks \
      resources_servers/sciprobe_ns_tools; do
      test -x "${NEMO_GYM_VENV_DIR}/${server_venv}/.venv/bin/python"
    done
    uv run --locked --no-sync python tools/generate_fingerprint.py \
      > /opt/nemo_rl_container_fingerprint
    du -sh /opt/gym_venvs /opt/ray_venvs /opt/uv-cache
  '

enroot export --output "${OUTPUT_TMP}" "${CONTAINER_NAME}"
mv "${OUTPUT_TMP}" "${OUTPUT_SQSH}"
chmod a+r "${OUTPUT_SQSH}"
ls -lh "${OUTPUT_SQSH}"
