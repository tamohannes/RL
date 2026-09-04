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
#SBATCH --time=02:00:00

set -euo pipefail

: "${CONTAINER_SAVE:?CONTAINER_SAVE is required}"
: "${SCRIPT_DIR:?SCRIPT_DIR is required}"

TARGET_SQSH="${CONTAINER_SAVE}"

run_preflight() {
  env \
    -u CODE_DIR \
    -u REQUIRED_ARCH \
    -u REQUIRED_PYTHON \
    -u REQUIRED_RAY \
    -u REQUIRE_WORKER_WRAPPERS \
    bash "${SCRIPT_DIR}/examples/preflight_nemo_rl_container.slurm"
}

if [[ -s "${TARGET_SQSH}" ]]; then
  export CONTAINER_IMAGE="${TARGET_SQSH}"
  unset CONTAINER_SAVE
  if run_preflight; then
    exit 0
  fi
  mv "${TARGET_SQSH}" "${TARGET_SQSH}.invalid-${SLURM_JOB_ID}"
fi

export CONTAINER_IMAGE="nvcr.io#nvidia/nemo-rl:v0.7.0"
export CONTAINER_SAVE="${TARGET_SQSH}"
run_preflight
