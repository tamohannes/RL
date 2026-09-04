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
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
# Account comes from SBATCH_ACCOUNT in the submit environment.
#SBATCH --partition=cpu
#SBATCH --time=00:20:00

set -euo pipefail

: "${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"
: "${CODE_DIR:?CODE_DIR is required}"

srun \
  --ntasks=1 \
  --cpus-per-task=2 \
  --exact \
  --kill-on-bad-exit=1 \
  --no-container-mount-home \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CODE_DIR}:/workspace/RL" \
  --container-workdir=/workspace/RL \
  bash -lc '
    set -euo pipefail
    export PATH=/opt/uv/bin:/opt/nemo_rl_venv/bin:/root/.local/bin:${PATH}
    gym_source=/workspace/RL/3rdparty/Gym-workspace/Gym
    test -f "${gym_source}/nemo_gym/__init__.py"
    export PYTHONPATH="${gym_source}:/workspace/RL:${PYTHONPATH:-}"
    export UV_PYTHON_INSTALL_DIR=/opt/uv-python
    export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
    export NEMO_RL_VENV_DIR=/opt/ray_venvs
    cd /workspace/RL
    gym_actor_python="${NEMO_RL_VENV_DIR}/nemo_rl.environments.nemo_gym.NemoGym/bin/python"
    test -x "${gym_actor_python}"
    "${gym_actor_python}" -c "import nemo_gym, openai"
    "${gym_actor_python}" examples/validate_sciprobe_canary_configs.py
  '
