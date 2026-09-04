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
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# The 91 GB sqsh can transiently exceed 120 GiB RSS while Pyxis prepares it.
#SBATCH --mem=200G
#SBATCH --time=00:30:00

set -euo pipefail

: "${CODE_DIR:?CODE_DIR is required}"
: "${CONTAINER:?CONTAINER is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${PERSISTENT_ROOT:?PERSISTENT_ROOT is required}"

export MODEL_PATH RUN_ROOT

srun \
  --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts="${CODE_DIR}:/workspace/RL,${MODEL_PATH}:${MODEL_PATH}:ro,${PERSISTENT_ROOT}:${PERSISTENT_ROOT}" \
  bash -lc '
    set -euo pipefail
    export PATH=/opt/uv/bin:/opt/nemo_rl_venv/bin:${PATH}
    export PYTHONPATH=/workspace/RL:/workspace/RL/3rdparty/Gym-workspace/Gym:${PYTHONPATH:-}
    export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
    export UV_PYTHON_INSTALL_DIR=/opt/uv-python
    cd /workspace/RL
    python -m py_compile examples/validate_sciprobe_overfit_canary_outputs.py
    uv run --locked --no-sync ruff check examples/validate_sciprobe_overfit_canary_outputs.py
    uv run --locked --no-sync python examples/validate_sciprobe_overfit_canary_outputs.py \
      --run-root "${RUN_ROOT}" \
      --model "${MODEL_PATH}" \
      --config examples/configs/recipes/llm/grpo-nemotron3.5-lightning-30ba3b-1n4g-automodel-sciprobe-overfit-canary.yaml \
      --expected-max-steps 6 \
      --expected-train-rollouts 16 \
      --expected-validation-rollouts 32
  '
