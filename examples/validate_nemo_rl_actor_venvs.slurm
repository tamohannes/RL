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
#SBATCH --time=00:40:00

set -euo pipefail

: "${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"
: "${CODE_DIR:?CODE_DIR is required}"

srun \
  --no-container-mount-home \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CODE_DIR}:/workspace/RL:ro" \
  --container-workdir=/workspace/RL \
  bash -lc '
    set -euo pipefail

    vllm_python_paths=(
      /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python
      /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
      /opt/ray_venvs/nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector/bin/python
      /opt/ray_venvs/nemo_rl.algorithms.async_utils.ReplayBuffer/bin/python
      /opt/ray_venvs/nemo_rl.experience.sync_rollout_actor.SyncRolloutActor/bin/python
    )
    for python_path in "${vllm_python_paths[@]}"; do
      "${python_path}" -c "import pathlib, ray, requests, urllib3, vllm; from urllib3.exceptions import HTTPError; target=pathlib.Path(urllib3.__file__).resolve(); assert str(target).startswith(\"/opt/uv-cache/\"), target; print(target, ray.__version__, urllib3.__version__, vllm.__version__)"
    done

    automodel_python=/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python
    "${automodel_python}" -c "import nemo_automodel, pathlib, ray, requests, urllib3; from urllib3.exceptions import HTTPError; target=pathlib.Path(urllib3.__file__).resolve(); assert str(target).startswith(\"/opt/uv-cache/\"), target; print(target, ray.__version__, urllib3.__version__)"

    echo "actor_venv_validation=ok"
  '
