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
# Take the whole node rather than naming a memory figure. The cpu nodes here
# carry 176GB, so the previous --mem=192G was unsatisfiable and Slurm rejected
# the job outright with "Requested node configuration is not available".
#SBATCH --exclusive
# Account comes from SBATCH_ACCOUNT in the submit environment.
#SBATCH --partition=cpu
# Matches the ns-tools stage. Unpacking, repairing uv, resolving and copying
# several hundred packages, then re-exporting does not reliably fit in two hours.
#SBATCH --time=04:00:00

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

    # Enroot does not preserve the managed Python the release image installs
    # under /root, so the frozen venv interpreters arrive here dangling and every
    # console script built from them fails with exit 127.
    #
    # Repair what is broken and leave what works. An earlier attempt gated the
    # whole repair on /opt/nemo_rl_venv having a usable python, which it did, so
    # the repair was skipped while the interpreters that actually matter, the ray
    # venvs, stayed dangling. The gym CLI then died on a missing bin/python3.
    #
    # Idempotent so this runs safely whatever the base image has already had
    # done to it. It does not make the stage self-sufficient: the ns-tools stage
    # is still required first. Its gym server venvs are indeed unused here, but
    # it also rebuilds the NemoGym actor venv, whose package symlinks in the
    # release image point into a uv cache archive that was never shipped.
    # Building straight off the base image gets past this repair and then dies
    # on ModuleNotFoundError: No module named nemo_gym.
    export UV_INSTALL_DIR=/opt/uv/bin
    mkdir -p "${UV_INSTALL_DIR}" "${UV_PYTHON_INSTALL_DIR}"
    if ! command -v uv >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/0.11.28/install.sh | \
        env UV_INSTALL_DIR="${UV_INSTALL_DIR}" UV_NO_MODIFY_PATH=1 sh
      hash -r
    fi
    ln -sfn /opt/uv/bin/uv /usr/local/bin/uv
    uv python install 3.13.14
    portable_python="$(uv python find --managed-python --no-project --resolve-links 3.13.14)"
    test -x "${portable_python}"

    # A venv refers to its interpreter as python, python3 and a versioned name.
    # Relink every one that does not currently resolve, and create it when the
    # release image never shipped it. The -x test is false for a dangling
    # symlink and true for a working one, so a healthy interpreter is untouched.
    for venv_bin in /opt/nemo_rl_venv/bin /opt/ray_venvs/*/bin; do
      [ -d "${venv_bin}" ] || continue
      for interpreter_name in python python3 python3.13; do
        if [ ! -x "${venv_bin}/${interpreter_name}" ]; then
          ln -sfn "${portable_python}" "${venv_bin}/${interpreter_name}"
        fi
      done
    done
    /opt/nemo_rl_venv/bin/python --version
    /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python --version

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
