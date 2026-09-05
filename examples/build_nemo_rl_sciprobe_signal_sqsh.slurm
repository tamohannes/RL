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
: "${UV_CACHE_DIR_OVERRIDE:?UV_CACHE_DIR_OVERRIDE is required}"

if [[ -e "${OUTPUT_SQSH}" ]]; then
  echo "Refusing to overwrite existing image: ${OUTPUT_SQSH}" >&2
  exit 1
fi

CONTAINER_NAME="sciprobe-nemo-rl-signal-${SLURM_JOB_ID}"
OUTPUT_TMP="${OUTPUT_SQSH}.tmp-${SLURM_JOB_ID}"
# The node root disk is about 98GB, and an unpacked image plus the venvs built on
# top of it does not fit there. Sites commonly provide a much larger node-local
# scratch, so pick the roomiest writable one instead of assuming /tmp, and refuse
# up front rather than failing forty minutes in with ENOSPC.
ENROOT_SCRATCH_MIN_GB="${ENROOT_SCRATCH_MIN_GB:-130}"
scratch_base=""
scratch_avail=0
for candidate in "${ENROOT_SCRATCH_DIR:-}" /raid/scratch /scratch /tmp; do
  [[ -n "${candidate}" && -d "${candidate}" && -w "${candidate}" ]] || continue
  candidate_avail="$(df -PBG "${candidate}" 2>/dev/null | awk 'NR==2 {gsub(/G/, "", $4); print $4}')"
  [[ -n "${candidate_avail}" ]] || continue
  if (( candidate_avail > scratch_avail )); then
    scratch_base="${candidate}"
    scratch_avail="${candidate_avail}"
  fi
done
if [[ -z "${scratch_base}" ]]; then
  echo "no writable scratch directory found" >&2
  exit 1
fi
if (( scratch_avail < ENROOT_SCRATCH_MIN_GB )); then
  echo "scratch ${scratch_base} has ${scratch_avail}GB free, need ${ENROOT_SCRATCH_MIN_GB}GB" >&2
  exit 1
fi
echo "scratch_base=${scratch_base} (${scratch_avail}GB free)"

ENROOT_ROOT="${scratch_base}/sciprobe-enroot-signal-${SLURM_JOB_ID}"

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
enroot version
df -h "${scratch_base}"

enroot create --name "${CONTAINER_NAME}" "${BASE_SQSH}"

enroot start \
  --root \
  --rw \
  --mount "${CODE_DIR}:/workspace/RL" \
  --mount "${UV_CACHE_DIR_OVERRIDE}:/opt/uv-cache" \
  "${CONTAINER_NAME}" \
  bash -lc '
    set -euo pipefail
    export PATH=/opt/uv/bin:/opt/nemo_rl_venv/bin:/root/.local/bin:${PATH}
    export PYTHONPATH=/workspace/RL:${PYTHONPATH:-}
    export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
    export UV_PYTHON_INSTALL_DIR=/opt/uv-python
    export UV_CACHE_DIR=/opt/uv-cache
    # The cache is bind-mounted from a different filesystem, so hardlinking
    # into /opt/gym_venvs cannot work and uv would fall back with a warning.
    export UV_LINK_MODE=copy
    export NEMO_GYM_VENV_DIR=/opt/gym_venvs
    export NEMO_GYM_EXTRA_ROOTS=/workspace/RL/examples/nemo_gym_extensions
    export NRL_FORCE_REBUILD_VENVS=false

    # Enroot does not preserve the managed Python the release image installed
    # under /root, so any stage that starts from an image which has not already
    # been repaired has to reinstate uv and relink every frozen venv interpreter
    # before the project can be invoked. This is idempotent: when an earlier
    # stage already did it, the probe below finds a working uv and skips.
    #
    # Being idempotent is what lets this stage build directly on the base image.
    # The intermediate ns-tools stage bakes servers this run does not use, since
    # the config_paths of the recipe name only vllm_model, sciprobe_simple_agent,
    # sciprobe_checks and sciprobe_ns_tools, and the chain deletes ns_tools
    # outright. Skipping it saves an image and a build.
    export UV_INSTALL_DIR=/opt/uv/bin
    if ! command -v uv >/dev/null 2>&1 || ! /opt/nemo_rl_venv/bin/python --version >/dev/null 2>&1; then
      echo "repairing uv and frozen venv interpreters"
      mkdir -p "${UV_INSTALL_DIR}" "${UV_PYTHON_INSTALL_DIR}"
      curl -LsSf https://astral.sh/uv/0.11.28/install.sh | \
        env UV_INSTALL_DIR="${UV_INSTALL_DIR}" UV_NO_MODIFY_PATH=1 sh
      hash -r
      uv python install 3.13.14
      portable_python="$(uv python find --managed-python --no-project --resolve-links 3.13.14)"
      test -x "${portable_python}"
      while IFS= read -r -d "" python_link; do
        ln -sfn "${portable_python}" "${python_link}"
      done < <(find /opt/nemo_rl_venv /opt/ray_venvs -type l -name python -print0)
      ln -sfn /opt/uv/bin/uv /usr/local/bin/uv
    else
      echo "uv already present, skipping repair"
    fi
    /opt/nemo_rl_venv/bin/python --version

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
