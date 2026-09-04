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

set -euo pipefail

# One-step, one-node NeMo-RL wiring canary for a prompt-only SciProbe task.
# Re-running with the same RUN_ID resumes the same output directory. Set a new
# RUN_ID only for a fresh experiment.

RUN_ID="${RUN_ID:-r1}"
EXP_NAME="sciprobe_rl_lightning-probe-canary-${RUN_ID}"

CODE_DIR="$(realpath "${CODE_DIR:-$PWD}")"
CONTAINER_CODE_DIR="${CONTAINER_CODE_DIR:-/workspace/RL}"
CONFIG_PATH="${CONFIG_PATH:-examples/configs/recipes/llm/grpo-nemotron3.5-lightning-30ba3b-1n4g-automodel-sciprobe-canary.yaml}"
TRAIN_PATH="${TRAIN_PATH:-examples/data/sciprobe/total-low-vaf-control-parents-canary.jsonl}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the Lightning checkpoint directory}"
CONTAINER="${CONTAINER:-nvcr.io#nvidia/nemo-rl:v0.7.0}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:?set PERSISTENT_ROOT}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:?set SLURM_ACCOUNT}"
SLURM_PARTITION="${SLURM_PARTITION:-batch}"
SLURM_TIME_LIMIT="${SLURM_TIME_LIMIT:-00:30:00}"
SLURM_DEPENDENCY="${SLURM_DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-false}"

RUN_ROOT="${PERSISTENT_ROOT}/runs/lightning-probe-canary-${RUN_ID}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
LOGGER_DIR="${RUN_ROOT}/logs"
RAY_LOG_DIR="${RUN_ROOT}/ray"
UV_CACHE_DIR_OVERRIDE="${PERSISTENT_ROOT}/cache/uv"
HF_HOME="${PERSISTENT_ROOT}/cache/huggingface"

export CONTAINER
export GPUS_PER_NODE=4
export NRL_FORCE_REBUILD_VENVS=false
export UV_CACHE_DIR_OVERRIDE
export HF_HOME
export BASE_LOG_DIR="${RAY_LOG_DIR}"
export CONTAINER_WORKDIR="${CONTAINER_CODE_DIR}"
export MOUNTS="${CODE_DIR}:${CONTAINER_CODE_DIR},${MODEL_PATH}:${MODEL_PATH}:ro,${PERSISTENT_ROOT}:${PERSISTENT_ROOT}"

export COMMAND="export PATH=/opt/uv/bin:/opt/nemo_rl_venv/bin:\${PATH} && \
  export PYTHONPATH=${CONTAINER_CODE_DIR}:\${PYTHONPATH:-} && \
  cd ${CONTAINER_CODE_DIR} && \
  NRL_FORCE_REBUILD_VENVS=false \
  UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv \
  UV_PYTHON_INSTALL_DIR=/opt/uv-python \
  NRL_VLLM_USE_V1=1 \
  NRL_WG_USE_RAY_REF=1 \
  UV_HTTP_TIMEOUT=300 \
  uv run --locked --no-sync python ./examples/run_grpo.py \
    --config ${CONFIG_PATH} \
    policy.model_name=${MODEL_PATH} \
    policy.tokenizer.name=${MODEL_PATH} \
    data.train.data_path=${TRAIN_PATH} \
    checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
    logger.log_dir=${LOGGER_DIR}"

SBATCH_CMD=(
  sbatch
  --nodes=1
  --account="${SLURM_ACCOUNT}"
  --job-name="${EXP_NAME}"
  --partition="${SLURM_PARTITION}"
  --time="${SLURM_TIME_LIMIT}"
  --gres=gpu:4
  --exclusive
)
if [[ -n "${SLURM_DEPENDENCY}" ]]; then
  SBATCH_CMD+=(--dependency="${SLURM_DEPENDENCY}")
fi
SBATCH_CMD+=(ray.sub)

echo "expname=${EXP_NAME}"
echo "output_dir=${CHECKPOINT_DIR}"
echo "container=${CONTAINER}"

if [[ "${DRY_RUN}" == "true" ]]; then
  printf 'command=%s\n' "${COMMAND}"
  printf 'sbatch='
  printf ' %q' "${SBATCH_CMD[@]}"
  printf '\n'
else
  mkdir -p \
    "${CHECKPOINT_DIR}" \
    "${LOGGER_DIR}" \
    "${RAY_LOG_DIR}" \
    "${UV_CACHE_DIR_OVERRIDE}" \
    "${HF_HOME}"
  cd "${CODE_DIR}"
  "${SBATCH_CMD[@]}"
fi
