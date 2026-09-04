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

# One-step Lightning GRPO canary through the native NeMo Gym + ns_tools path.
# The prompt requests two Python calls, and the second reads state created by
# the first. Reuse RUN_ID only to resume; choose a new one for a fresh run.

RUN_ID="${RUN_ID:-r1}"
EXP_NAME="sciprobe_rl_lightning-tool-canary-${RUN_ID}"

CODE_DIR="$(realpath "${CODE_DIR:-$PWD}")"
CONTAINER_CODE_DIR="${CONTAINER_CODE_DIR:-/workspace/RL}"
CONFIG_PATH="${CONFIG_PATH:-examples/configs/recipes/llm/grpo-nemotron3.5-lightning-30ba3b-1n4g-automodel-sciprobe-tool-canary.yaml}"
TRAIN_PATH="${TRAIN_PATH:-examples/data/sciprobe/stateful-python-session-canary.jsonl}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the Lightning checkpoint directory}"
CONTAINER="${CONTAINER:?set CONTAINER}"
SANDBOX_CONTAINER="${SANDBOX_CONTAINER:?set SANDBOX_CONTAINER to a pinned sandbox image}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:?set PERSISTENT_ROOT}"
PLAIN_CANARY_RUN_ROOT="${PLAIN_CANARY_RUN_ROOT:-${PERSISTENT_ROOT}/runs/lightning-probe-canary-r5}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:?set SLURM_ACCOUNT}"
SLURM_PARTITION="${SLURM_PARTITION:-batch}"
SLURM_TIME_LIMIT="${SLURM_TIME_LIMIT:-00:30:00}"
SLURM_DEPENDENCY="${SLURM_DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-false}"

RUN_ROOT="${PERSISTENT_ROOT}/runs/lightning-tool-canary-${RUN_ID}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
LOGGER_DIR="${RUN_ROOT}/logs"
NEMO_GYM_LOG_DIR="${RUN_ROOT}/nemo-gym"
RAY_LOG_DIR="${RUN_ROOT}/ray"
UV_CACHE_DIR_OVERRIDE="${PERSISTENT_ROOT}/cache/uv"
HF_HOME="${PERSISTENT_ROOT}/cache/huggingface"

export CONTAINER
export SANDBOX_CONTAINER
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-/start-with-nginx.sh}"
export NEMO_SKILLS_SANDBOX_HOST=127.0.0.1
export NEMO_SKILLS_SANDBOX_PORT=6000
export NEMO_GYM_VENV_DIR=/opt/gym_venvs
export GPUS_PER_NODE=4
export NRL_FORCE_REBUILD_VENVS=false
export UV_CACHE_DIR_OVERRIDE
export HF_HOME
export WANDB_MODE=offline
export BASE_LOG_DIR="${RAY_LOG_DIR}"
export CONTAINER_WORKDIR="${CONTAINER_CODE_DIR}"
export MOUNTS="${CODE_DIR}:${CONTAINER_CODE_DIR},${MODEL_PATH}:${MODEL_PATH}:ro,${PERSISTENT_ROOT}:${PERSISTENT_ROOT}"

export COMMAND="export PATH=/opt/uv/bin:/opt/nemo_rl_venv/bin:\${PATH} && \
  export PYTHONPATH=${CONTAINER_CODE_DIR}:\${PYTHONPATH:-} && \
  export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv && \
  export UV_PYTHON_INSTALL_DIR=/opt/uv-python && \
  export NRL_FORCE_REBUILD_VENVS=false && \
  cd ${CONTAINER_CODE_DIR} && \
  uv run --locked --no-sync python examples/validate_sciprobe_canary_configs.py && \
  uv run --locked --no-sync python examples/validate_lightning_tool_tokenization.py \
    --model ${MODEL_PATH} && \
  uv run --locked --no-sync python examples/validate_sciprobe_plain_canary_outputs.py \
    --run-root ${PLAIN_CANARY_RUN_ROOT} \
    --checkpoint-only && \
  NRL_VLLM_USE_V1=1 \
  NRL_WG_USE_RAY_REF=1 \
  UV_HTTP_TIMEOUT=300 \
  uv run --locked --no-sync python ./examples/nemo_gym/run_grpo_nemo_gym.py \
    --config ${CONFIG_PATH} \
    policy.model_name=${MODEL_PATH} \
    policy.tokenizer.name=${MODEL_PATH} \
    data.train.data_path=${TRAIN_PATH} \
    checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
    logger.log_dir=${LOGGER_DIR} \
    env.nemo_gym.nemo_gym_log_dir=${NEMO_GYM_LOG_DIR} && \
  uv run --locked --no-sync python examples/validate_sciprobe_tool_canary_outputs.py \
    --run-root ${RUN_ROOT} \
    --expected-rollouts 4"

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
echo "sandbox=${SANDBOX_CONTAINER}"

if [[ "${DRY_RUN}" == "true" ]]; then
  printf 'command=%s\n' "${COMMAND}"
  printf 'sbatch='
  printf ' %q' "${SBATCH_CMD[@]}"
  printf '\n'
else
  if [[ ! -e "${CONTAINER}" ]]; then
    if [[ -z "${SLURM_DEPENDENCY}" ]]; then
      echo "Missing required container: ${CONTAINER}" >&2
      exit 1
    fi
    echo "container_pending_dependency=${CONTAINER}"
  fi
  for required_path in \
    "${SANDBOX_CONTAINER}" \
    "${MODEL_PATH}" \
    "${PLAIN_CANARY_RUN_ROOT}/checkpoints/latest_checkpoint_status.json" \
    "${CODE_DIR}/${CONFIG_PATH}" \
    "${CODE_DIR}/${TRAIN_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
      echo "Missing required path: ${required_path}" >&2
      exit 1
    fi
  done
  mkdir -p \
    "${CHECKPOINT_DIR}" \
    "${LOGGER_DIR}" \
    "${NEMO_GYM_LOG_DIR}" \
    "${RAY_LOG_DIR}" \
    "${UV_CACHE_DIR_OVERRIDE}" \
    "${HF_HOME}"
  cd "${CODE_DIR}"
  "${SBATCH_CMD[@]}"
fi
