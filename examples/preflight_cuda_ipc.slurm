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

#SBATCH --job-name=sciprobe_rl_cuda-ipc-preflight-r1
# Account comes from SBATCH_ACCOUNT in the submit environment.
#SBATCH --partition=batch
#SBATCH --nodes=1
# the account QOS requires the full four-GPU node even for a tiny smoke.
#SBATCH --gres=gpu:4
# The 84 GB squashfs can take several minutes to mount on a cold node.
#SBATCH --time=00:15:00
#SBATCH --output=cuda-ipc-r1-%j.log

set -euo pipefail

CODE_DIR="${CODE_DIR:?set CODE_DIR to the staged NeMo RL checkout}"
CONTAINER="${CONTAINER:?set CONTAINER to the repaired NeMo RL image}"
CONTAINER_CODE_DIR="${CONTAINER_CODE_DIR:-/workspace/RL}"

srun \
  --container-image="${CONTAINER}" \
  --container-mounts="${CODE_DIR}:${CONTAINER_CODE_DIR}" \
  --container-workdir="${CONTAINER_CODE_DIR}" \
  --no-container-mount-home \
  bash -lc '
    set -euo pipefail
    export PYTHONPATH=/workspace/RL
    PYTHON=/opt/nemo_rl_venv/bin/python

    echo "case=policy-expandable-generation-default expected=pidfd-ebadf"
    env -u PYTORCH_CUDA_ALLOC_CONF "$PYTHON" examples/preflight_cuda_ipc.py \
      --producer-alloc-conf expandable_segments:True \
      --consumer-alloc-conf unset \
      --expect pidfd-ebadf

    echo "case=expandable-no-warmup expected=pidfd-ebadf"
    env -u PYTORCH_CUDA_ALLOC_CONF "$PYTHON" examples/preflight_cuda_ipc.py \
      --producer-alloc-conf expandable_segments:True \
      --consumer-alloc-conf expandable_segments:True \
      --expect pidfd-ebadf

    echo "case=expandable-with-warmup expected=pass"
    env -u PYTORCH_CUDA_ALLOC_CONF "$PYTHON" examples/preflight_cuda_ipc.py \
      --producer-alloc-conf expandable_segments:True \
      --consumer-alloc-conf expandable_segments:True \
      --warmup-consumer \
      --expect pass

    echo "case=legacy-no-warmup expected=pass"
    env -u PYTORCH_CUDA_ALLOC_CONF "$PYTHON" examples/preflight_cuda_ipc.py \
      --producer-alloc-conf expandable_segments:False \
      --consumer-alloc-conf unset \
      --expect pass

    echo "cuda_ipc_preflight=ok"
  '
