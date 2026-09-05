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

# Extend the source-matched Lightning image with only the NeMo Gym server
# environments needed for ns_tools + deterministic math verification.
: "${BASE_SQSH:?BASE_SQSH is required}"
: "${OUTPUT_SQSH:?OUTPUT_SQSH is required}"
: "${CODE_DIR:?CODE_DIR is required}"
: "${UV_CACHE_DIR_OVERRIDE:?UV_CACHE_DIR_OVERRIDE is required}"

if [[ -e "${OUTPUT_SQSH}" ]]; then
  echo "Refusing to overwrite existing image: ${OUTPUT_SQSH}" >&2
  exit 1
fi

CONTAINER_NAME="sciprobe-nemo-rl-ns-tools-${SLURM_JOB_ID}"
OUTPUT_TMP="${OUTPUT_SQSH}.tmp-${SLURM_JOB_ID}"
# The node root disk is about 98GB, and an unpacked image plus the venvs built on
# top of it does not fit there. Sites commonly provide a much larger node-local
# scratch, so pick the roomiest writable one instead of assuming /tmp, and refuse
# up front rather than failing forty minutes in with ENOSPC.
ENROOT_SCRATCH_MIN_GB="${ENROOT_SCRATCH_MIN_GB:-110}"
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

ENROOT_ROOT="${scratch_base}/sciprobe-enroot-ns-tools-${SLURM_JOB_ID}"

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
df -h "${scratch_base}"

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
    export NEMO_GYM_VENV_DIR=/opt/gym_venvs
    # The release image contains an older NemoGym actor venv whose package
    # symlinks target a cache archive that is not part of the image. Rebuild
    # that actor venv from this checkout during the dry-run prefetch.
    export NRL_FORCE_REBUILD_VENVS=true
    export UV_TORCH_BACKEND=cu130
    export UV_LINK_MODE=hardlink
    mkdir -p "${NEMO_GYM_VENV_DIR}"

    # The base image was built with a managed Python under /root, which Enroot
    # did not preserve. Install uv + Python under /opt and repair every frozen
    # venv interpreter link before invoking the project.
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
    /opt/nemo_rl_venv/bin/python --version

    cd /workspace/RL
    uv lock --check

    # The vLLM release wheel is per-architecture. Derive it from the build host so the
    # same script produces an aarch64 image on a Grace cluster and an x86_64 one on an
    # Intel cluster, instead of silently building the wrong wheel into the image.
    wheel_arch="$(uname -m)"
    case "${wheel_arch}" in
      aarch64|x86_64) ;;
      *) echo "unsupported build architecture: ${wheel_arch}" >&2; exit 1 ;;
    esac
    echo "wheel_arch=${wheel_arch}"
    gym_override=/tmp/sciprobe-gym-vllm-override.txt
    printf "%s\n" \
      "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0-cp38-abi3-manylinux_2_28_${wheel_arch}.whl" \
      > "${gym_override}"
    export UV_OVERRIDE="${gym_override}"

    uv run --locked python examples/nemo_gym/prefetch_venvs.py \
      examples/nemo_gym/prefetch_sciprobe_ns_tools.yaml

    rm -f "${gym_override}"
    test -d /opt/gym_venvs
    test -n "$(find /opt/gym_venvs -mindepth 1 -maxdepth 1 -type d -print -quit)"
    du -sh /opt/gym_venvs /opt/ray_venvs

    uv run --locked python tools/generate_fingerprint.py \
      > /opt/nemo_rl_container_fingerprint
  '

enroot export --output "${OUTPUT_TMP}" "${CONTAINER_NAME}"
mv "${OUTPUT_TMP}" "${OUTPUT_SQSH}"
chmod a+r "${OUTPUT_SQSH}"
ls -lh "${OUTPUT_SQSH}"
