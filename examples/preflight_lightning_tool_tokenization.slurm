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
# Account comes from SBATCH_ACCOUNT in the submit environment.
#SBATCH --partition=cpu
#SBATCH --time=00:20:00

set -euo pipefail

: "${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"
: "${CODE_DIR:?CODE_DIR is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"

srun \
  --no-container-mount-home \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CODE_DIR}:/workspace/RL,${MODEL_PATH}:${MODEL_PATH}:ro" \
  bash -lc '
    set -euo pipefail
    export PATH=/opt/uv/bin:/opt/nemo_rl_venv/bin:/root/.local/bin:${PATH}
    export PYTHONPATH=/workspace/RL:${PYTHONPATH:-}
    export UV_PYTHON_INSTALL_DIR=/opt/uv-python
    export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
    export NRL_FORCE_REBUILD_VENVS=false
    cd /workspace/RL
    uv run --locked --no-sync python examples/validate_sciprobe_canary_configs.py
    uv run --locked --no-sync python examples/validate_lightning_tool_tokenization.py \
      --model "${MODEL_PATH}"
  '
