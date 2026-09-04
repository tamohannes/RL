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

# One SciProbe Harbor task, eight on-policy generations, one GRPO update.
# This launcher is fresh-run only: choose a new RUN_ID for every invocation.
# The container's prebuilt Harbor-agent service environment must include the
# landed SciProbe adapter and typed-failure contract.
# This repository pins the required Gym revision. Rebuild CONTAINER from this
# checkout so the Harbor-agent service environment contains that revision;
# the import check below rejects a stale image before training starts.

RUN_ID="${RUN_ID:?set RUN_ID to a fresh suffix, for example r1}"
BANK_DIR="${BANK_DIR:?set BANK_DIR to a validated SciProbe Harbor bank}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the Lightning checkpoint directory}"
CONTAINER="${CONTAINER:?set CONTAINER to the NeMo-RL image or .sqsh}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:?set PERSISTENT_ROOT to a shared writable directory}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(realpath "${CODE_DIR:-${SCRIPT_DIR}/..}")"
BANK_DIR="$(realpath "${BANK_DIR}")"
MODEL_PATH="$(realpath "${MODEL_PATH}")"
PERSISTENT_ROOT="$(realpath -m "${PERSISTENT_ROOT}")"

CONTAINER_CODE_DIR="${CONTAINER_CODE_DIR:-/workspace/RL}"
CONTAINER_BANK_DIR="/workspace/sciprobe-bank"
CONTAINER_HARBOR_PYTHON="/opt/gym_venvs/responses_api_agents/harbor_agent/.venv/bin/python"
CONFIG_PATH="examples/configs/recipes/llm/grpo-nemotron3.5-lightning-30ba3b-1n4g-automodel-sciprobe-harbor-canary.yaml"
EXP_NAME="sciprobe_rl_lightning-harbor-canary-${RUN_ID}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_reason_math}"
SLURM_PARTITION="${SLURM_PARTITION:-batch}"
SLURM_TIME_LIMIT="${SLURM_TIME_LIMIT:-01:00:00}"
SLURM_DEPENDENCY="${SLURM_DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-false}"

RUN_ROOT="${PERSISTENT_ROOT}/runs/lightning-harbor-canary-${RUN_ID}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
LOGGER_DIR="${RUN_ROOT}/logs"
NEMO_GYM_LOG_DIR="${RUN_ROOT}/nemo-gym"
RAY_LOG_DIR="${RUN_ROOT}/ray"
HARBOR_JOBS_DIR="${RUN_ROOT}/harbor-jobs"
SINGULARITY_CACHE_DIR="${PERSISTENT_ROOT}/cache/harbor-singularity"
UV_CACHE_DIR_OVERRIDE="${PERSISTENT_ROOT}/cache/uv"
HF_HOME="${PERSISTENT_ROOT}/cache/huggingface"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse fresh run root: ${RUN_ROOT}" >&2
  exit 1
fi
for required_path in \
  "${BANK_DIR}/train.jsonl" \
  "${BANK_DIR}/tasks" \
  "${MODEL_PATH}" \
  "${CODE_DIR}/${CONFIG_PATH}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

# Validate the bank on the host before reserving four GPUs.
sciprobe probes validate-harbor --bank "${BANK_DIR}"

export CONTAINER
export GPUS_PER_NODE=4
export NRL_FORCE_REBUILD_VENVS=false
export UV_CACHE_DIR_OVERRIDE
export HF_HOME
export WANDB_MODE="${WANDB_MODE:-offline}"
export BASE_LOG_DIR="${RAY_LOG_DIR}"
export CONTAINER_WORKDIR="${CONTAINER_CODE_DIR}"
export MOUNTS="${CODE_DIR}:${CONTAINER_CODE_DIR},${MODEL_PATH}:${MODEL_PATH}:ro,${PERSISTENT_ROOT}:${PERSISTENT_ROOT},${BANK_DIR}:${CONTAINER_BANK_DIR}:ro"

export COMMAND="export PATH=/opt/uv/bin:/opt/nemo_rl_venv/bin:\${PATH} && \
  export PYTHONPATH=${CONTAINER_CODE_DIR}:\${PYTHONPATH:-} && \
  export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv && \
  export UV_PYTHON_INSTALL_DIR=/opt/uv-python && \
  export NEMO_GYM_VENV_DIR=/opt/gym_venvs && \
  export SCIPROBE_HARBOR_BANK_DIR=${CONTAINER_BANK_DIR} && \
  export SCIPROBE_HARBOR_TASKS_DIR=${CONTAINER_BANK_DIR}/tasks && \
  export SCIPROBE_SINGULARITY_CACHE_DIR=${SINGULARITY_CACHE_DIR} && \
  export SCIPROBE_HARBOR_JOBS_DIR=${HARBOR_JOBS_DIR} && \
  ${CONTAINER_HARBOR_PYTHON} -c 'from responses_api_agents.harbor_agent.app import HARBOR_AGENT_FAILURE_CLASS; from responses_api_agents.harbor_agent.custom_envs.singularity.singularity import SCIPROBE_VERIFIER_ISOLATION_PROFILE; assert HARBOR_AGENT_FAILURE_CLASS == "harbor_agent_error"; assert SCIPROBE_VERIFIER_ISOLATION_PROFILE == "sciprobe-answer-json-v1"' && \
  cd ${CONTAINER_CODE_DIR} && \
  NRL_FORCE_REBUILD_VENVS=false \
  NRL_VLLM_USE_V1=1 \
  NRL_WG_USE_RAY_REF=1 \
  UV_HTTP_TIMEOUT=300 \
  uv run --locked --no-sync python ./examples/run_grpo_single_controller.py \
    --config ${CONFIG_PATH} \
    policy.model_name=${MODEL_PATH} \
    policy.tokenizer.name=${MODEL_PATH} \
    checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
    logger.log_dir=${LOGGER_DIR} \
    logger.wandb.name=lightning-harbor-canary-${RUN_ID} \
    env.nemo_gym.nemo_gym_log_dir=${NEMO_GYM_LOG_DIR}"

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
echo "bank=${BANK_DIR}"
echo "bank_mount=${BANK_DIR}:${CONTAINER_BANK_DIR}:ro"
echo "container=${CONTAINER}"

if [[ "${DRY_RUN}" == "true" ]]; then
  printf 'command=%s\n' "${COMMAND}"
  printf 'sbatch='
  printf ' %q' "${SBATCH_CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p \
  "${CHECKPOINT_DIR}" \
  "${LOGGER_DIR}" \
  "${NEMO_GYM_LOG_DIR}" \
  "${RAY_LOG_DIR}" \
  "${HARBOR_JOBS_DIR}" \
  "${SINGULARITY_CACHE_DIR}" \
  "${UV_CACHE_DIR_OVERRIDE}" \
  "${HF_HOME}"
cd "${CODE_DIR}"
"${SBATCH_CMD[@]}"
