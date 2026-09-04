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
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail

: "${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"

SAVE_ARGS=()
if [[ -n "${CONTAINER_SAVE:-}" ]]; then
  mkdir -p "$(dirname "${CONTAINER_SAVE}")"
  SAVE_ARGS+=(--container-save="${CONTAINER_SAVE}")
fi

MOUNT_ARGS=()
if [[ -n "${CODE_DIR:-}" ]]; then
  MOUNT_ARGS+=(--container-mounts="${CODE_DIR}:/workspace/RL")
fi

srun \
  --no-container-mount-home \
  --container-image="${CONTAINER_IMAGE}" \
  "${SAVE_ARGS[@]}" \
  "${MOUNT_ARGS[@]}" \
  bash -c '
    set -euo pipefail
    export PATH=/opt/uv/bin:/opt/nemo_rl_venv/bin:/root/.local/bin:${PATH}
    export PYTHONPATH=/workspace/RL:${PYTHONPATH:-}
    export UV_PYTHON_INSTALL_DIR=/opt/uv-python
    export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
    if [[ -x /opt/nemo_rl_venv/bin/python ]]; then
      PYTHON=/opt/nemo_rl_venv/bin/python
    else
      PYTHON=python
    fi
    uname -m
    "${PYTHON}" --version
    "${PYTHON}" -c "import ray, sys; print(sys.executable); print(f\"ray={ray.__version__}\")"
    uv --version

    if [[ -n "${REQUIRED_ARCH:-}" ]]; then
      [[ "$(uname -m)" == "${REQUIRED_ARCH}" ]]
    fi
    "${PYTHON}" -c "
import os
import ray
import sys

required_python = os.environ.get(\"REQUIRED_PYTHON\")
if required_python:
    required = tuple(int(part) for part in required_python.split(\".\"))
    assert sys.version_info[:3] >= required, (sys.version_info[:3], required)

required_ray = os.environ.get(\"REQUIRED_RAY\")
if required_ray:
    assert ray.__version__ == required_ray, (ray.__version__, required_ray)
"

    if [[ -n "${CODE_DIR:-}" ]]; then
      cd /workspace/RL
      "${PYTHON}" tools/generate_fingerprint.py > /tmp/runtime_fingerprint.json
      cmp /opt/nemo_rl_container_fingerprint /tmp/runtime_fingerprint.json
      uv run --locked --no-sync python -c "import nemo_rl, ray; print(nemo_rl.__file__); print(f\"source_ray={ray.__version__}\")"
      uv run --locked --no-sync python examples/validate_sciprobe_canary_configs.py
    fi

    if [[ "${REQUIRE_WORKER_WRAPPERS:-false}" == "true" ]]; then
      test -x /usr/local/bin/python-VllmGenerationWorker
      test -x /usr/local/bin/python-DTensorPolicyWorkerV2
    fi
    if [[ -d /opt/nemo-rl/.git ]]; then
      git -C /opt/nemo-rl rev-parse HEAD
    fi
    if [[ -f /opt/nemo-rl/examples/configs/recipes/llm/dapo-nanov3.5-30BA3B-4n8g-automodel.yaml ]]; then
      echo lightning_recipe=yes
    else
      echo lightning_recipe=no
    fi
  '
