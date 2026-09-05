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
umask 077

# GRPO over a SciProbe bank: many probes, eight on-policy generations each.
#
# Differs from the single-probe signal canary in three ways that matter.
#
# The bank keeps the policy and the grader apart. checks.py recomputes the answer
# from data/, so the sandbox is given only <bank>/policy and each row chdirs into
# its own probe directory. The canary mounted one probe's data at a fixed path,
# which leaks every probe to every rollout as soon as there is more than one.
#
# Grading runs in a real scientific environment. Most probes recompute their gold
# with numpy, pandas, scipy or R, none of which the Gym venv carries, so
# SCIPROBE_CHECKER_PYTHON points the checker subprocess at a prepared prefix while
# all hashing and provenance stay in the trusted venv.
#
# The verifier's probe map comes from the bank, not the repository. A run splices
# its own probes into a copy of the extensions tree so the committed config stays a
# single readable example.
#
# Automodel is on PYTHONPATH for the same reason Gym is. Both ship as editable
# installs in the image, and the editable finder does not resolve at runtime:
# the worker venv has its site-packages on sys.path and the source tree is
# present, yet importing the package still fails. Gym was already worked around
# this way, which is why nemo_gym imported and nemo_automodel did not. The
# submodule is pinned at the commit the venv was built from, so this points at
# the same source the editable install intended.
#
# Reuse RUN_ID only with RESUME=true; use a new suffix for a fresh experiment.

RUN_ID="${RUN_ID:-r1}"
EXP_NAME="sciprobe_rl_lightning-bank-${RUN_ID}"

CODE_DIR="$(realpath "${CODE_DIR:-$PWD}")"
CONTAINER_CODE_DIR="${CONTAINER_CODE_DIR:-/workspace/RL}"
CONFIG_PATH="${CONFIG_PATH:-examples/configs/recipes/llm/grpo-nemotron3.5-lightning-30ba3b-1n8g-automodel-sciprobe-bank-a100.yaml}"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the Lightning checkpoint directory}"
CONTAINER="${CONTAINER:?set CONTAINER to the training image}"
SANDBOX_CONTAINER="${SANDBOX_CONTAINER:?set SANDBOX_CONTAINER to a pinned sandbox image}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:?set PERSISTENT_ROOT}"
BANK_DIR="$(realpath "${BANK_DIR:?set BANK_DIR to a collected SciProbe bank}")"
# Interpreter that runs checks.py. Must hold the probes' scientific dependencies.
GRADING_PYTHON="$(realpath "${GRADING_PYTHON:?set GRADING_PYTHON to the grading environment interpreter}")"
# Defaults to the whole bank; point at a subset to train on part of it.
TRAIN_PATH="$(realpath "${TRAIN_PATH:-${BANK_DIR}/train.jsonl}")"

SLURM_ACCOUNT="${SLURM_ACCOUNT:?set SLURM_ACCOUNT}"
SLURM_PARTITION="${SLURM_PARTITION:-batch}"
SLURM_TIME_LIMIT="${SLURM_TIME_LIMIT:-04:00:00}"
SLURM_NODES="${SLURM_NODES:-1}"
SLURM_GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-8}"
SLURM_DEPENDENCY="${SLURM_DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-false}"
RESUME="${RESUME:-false}"

# Fresh credentials per submission. The model sandbox receives a minimal env and
# none of these.
SCIPROBE_VERIFIER_TOKEN="$(openssl rand -hex 32)"
SCIPROBE_CAPABILITY_SIGNING_KEY="$(openssl rand -hex 32)"
SCIPROBE_TRUSTED_INGRESS_TOKEN="$(openssl rand -hex 32)"
SCIPROBE_POLICY_GENERATION_TOKEN="$(openssl rand -hex 32)"
export SCIPROBE_VERIFIER_TOKEN SCIPROBE_CAPABILITY_SIGNING_KEY
export SCIPROBE_TRUSTED_INGRESS_TOKEN SCIPROBE_POLICY_GENERATION_TOKEN
unset SCIPROBE_VERIFIER_CAPABILITY

RUN_ROOT="${PERSISTENT_ROOT}/runs/lightning-bank-${RUN_ID}"
RUNTIME_EXTENSIONS="${RUN_ROOT}/extensions"
SCIPROBE_CAPABILITY_STORE_PATH="${RUN_ROOT}/runtime/capability-results.sqlite3"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
LOGGER_DIR="${RUN_ROOT}/logs"
NEMO_GYM_LOG_DIR="${RUN_ROOT}/nemo-gym"
RAY_LOG_DIR="${RUN_ROOT}/ray"
UV_CACHE_DIR_OVERRIDE="${PERSISTENT_ROOT}/cache/uv"
HF_HOME="${PERSISTENT_ROOT}/cache/huggingface"
export SCIPROBE_CAPABILITY_STORE_PATH

if [[ "${RESUME}" != "true" && -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse ${RUN_ROOT}; pick a new RUN_ID or set RESUME=true" >&2
  exit 1
fi

# The grader tree and the grading interpreter are read through the identity mount
# of PERSISTENT_ROOT, so both must live under it. A conda prefix bakes its own
# absolute path into shebangs and R configuration, so it cannot be remapped.
for required in "${BANK_DIR}" "${GRADING_PYTHON}" "${TRAIN_PATH}"; do
  case "${required}" in
    "${PERSISTENT_ROOT}"/*) ;;
    *) echo "Must live under PERSISTENT_ROOT (${PERSISTENT_ROOT}): ${required}" >&2; exit 1 ;;
  esac
done

export CONTAINER SANDBOX_CONTAINER
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-unshare --pid --fork --mount-proc --kill-child /workspace/start-sciprobe-loopback-sandbox.sh}"
# The sandbox sees the policy tree and nothing else of the bank: no checks.py, no
# gold, no other probe's grader copy.
export SANDBOX_EXTRA_MOUNTS="${BANK_DIR}/policy:/workspace/sciprobe-probe:ro,${CODE_DIR}/examples/sandbox_seccomp_hook:/workspace/sciprobe-seccomp-hook:ro,${CODE_DIR}/examples/validate_sciprobe_sandbox_seccomp.py:/workspace/validate-sciprobe-sandbox-seccomp.py:ro,${CODE_DIR}/examples/start_sciprobe_loopback_sandbox.sh:/workspace/start-sciprobe-loopback-sandbox.sh:ro"
# Lowest Landlock ABI the sandbox will accept. Left unset the hook holds its
# strict floor of 3 and refuses to start on an older kernel. Set it only for a
# cluster whose kernel cannot reach that floor, and only after checking what the
# kernel still enforces: below 3 the sandbox gives up rename and truncate
# confinement, which govern modifying files it can already open, while the
# network block and the read rules are ABI 1 and remain in force.
SCIPROBE_LANDLOCK_MIN_ABI="${SCIPROBE_LANDLOCK_MIN_ABI:-}"

export SANDBOX_ENV_VARS="NEMO_SKILLS_SANDBOX_BLOCK_NETWORK=1,SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK=1,PYTHONPATH=/workspace/sciprobe-seccomp-hook,NUM_WORKERS=1,SANDBOX_FORCE_SINGLE_NODE=1"
if [[ -n "${SCIPROBE_LANDLOCK_MIN_ABI}" ]]; then
  SANDBOX_ENV_VARS="${SANDBOX_ENV_VARS},SCIPROBE_LANDLOCK_MIN_ABI=${SCIPROBE_LANDLOCK_MIN_ABI}"
  export SANDBOX_ENV_VARS
fi
export SCIPROBE_REQUIRE_SANDBOX_SECCOMP_PREFLIGHT=1
export NEMO_SKILLS_SANDBOX_HOST=127.0.0.1
export NEMO_SKILLS_SANDBOX_PORT=6000

export NEMO_GYM_VENV_DIR=/opt/gym_venvs
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NEMO_GYM_EXTRA_ROOTS="${RUNTIME_EXTENSIONS}"
export GPUS_PER_NODE="${SLURM_GPUS_PER_NODE}"
export RAY_LOOPBACK_ONLY=1
export NRL_FORCE_REBUILD_VENVS=false
export UV_CACHE_DIR_OVERRIDE HF_HOME
export WANDB_MODE=offline
export BASE_LOG_DIR="${RAY_LOG_DIR}"
export CONTAINER_WORKDIR="${CONTAINER_CODE_DIR}"
export MOUNTS="${CODE_DIR}:${CONTAINER_CODE_DIR},${MODEL_PATH}:${MODEL_PATH}:ro,${PERSISTENT_ROOT}:${PERSISTENT_ROOT}"

export SCIPROBE_PROBE_BANK_ROOT="${BANK_DIR}/grader"
export SCIPROBE_CHECKER_PYTHON="${GRADING_PYTHON}"

export COMMAND="export PATH=/opt/uv/bin:/opt/nemo_rl_venv/bin:\${PATH} && \
  export PYTHONPATH=${CONTAINER_CODE_DIR}:${CONTAINER_CODE_DIR}/3rdparty/Gym-workspace/Gym:${CONTAINER_CODE_DIR}/3rdparty/Automodel-workspace/Automodel:\${PYTHONPATH:-} && \
  export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv && \
  export UV_PYTHON_INSTALL_DIR=/opt/uv-python && \
  export NEMO_RL_VENV_DIR=/opt/ray_venvs && \
  export NEMO_GYM_VENV_DIR=/opt/gym_venvs && \
  export NEMO_GYM_EXTRA_ROOTS=${RUNTIME_EXTENSIONS} && \
  export SCIPROBE_CAPABILITY_STORE_PATH=${SCIPROBE_CAPABILITY_STORE_PATH} && \
  export SCIPROBE_PROBE_BANK_ROOT=${SCIPROBE_PROBE_BANK_ROOT} && \
  export SCIPROBE_CHECKER_PYTHON=${SCIPROBE_CHECKER_PYTHON} && \
  export NRL_FORCE_REBUILD_VENVS=false && \
  cd ${CONTAINER_CODE_DIR} && \
  ray_session_log_dir=\$(readlink -f /tmp/ray/session_latest/logs) && \
  test -d \"\${ray_session_log_dir}\" && \
  uv run --locked --no-sync python examples/validate_sciprobe_ray_loopback.py \
    --require-worker --require-token-auth --ray-log-dir \"\${ray_session_log_dir}\" && \
  test -x /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python && \
  /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python -c 'import nemo_gym, openai' && \
  /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python \
    examples/validate_sciprobe_bank.py \
      --bank ${BANK_DIR} \
      --train-path ${TRAIN_PATH} \
      --checker-python ${SCIPROBE_CHECKER_PYTHON} \
      --execute-sample 2 && \
  /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python \
    examples/validate_sciprobe_no_replay.py && \
  test -x /opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python && \
  /opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python \
    examples/validate_lightning_mtp_disabled.py --model ${MODEL_PATH} && \
  uv run --locked --no-sync python examples/validate_lightning_tool_tokenization.py \
    --model ${MODEL_PATH} --config ${CONFIG_PATH} && \
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
    logger.wandb.name=lightning-bank-${RUN_ID} \
    env.nemo_gym.nemo_gym_log_dir=${NEMO_GYM_LOG_DIR}"

SBATCH_CMD=(
  sbatch
  --nodes="${SLURM_NODES}"
  --job-name="${EXP_NAME}"
  --account="${SLURM_ACCOUNT}"
  --partition="${SLURM_PARTITION}"
  --time="${SLURM_TIME_LIMIT}"
  --gres="gpu:${SLURM_GPUS_PER_NODE}"
  --exclusive
  --output="${RAY_LOG_DIR}/slurm-%j.out"
  --error="${RAY_LOG_DIR}/slurm-%j.err"
)
if [[ -n "${SLURM_DEPENDENCY}" ]]; then
  SBATCH_CMD+=(--dependency="${SLURM_DEPENDENCY}")
fi
SBATCH_CMD+=(ray.sub)

echo "exp_name=${EXP_NAME}"
echo "run_root=${RUN_ROOT}"
echo "config=${CONFIG_PATH}"
echo "bank=${BANK_DIR}"
echo "train_path=${TRAIN_PATH}  rows=$(grep -c . "${TRAIN_PATH}")"
echo "probe_bank_root=${SCIPROBE_PROBE_BANK_ROOT}"
echo "checker_python=${SCIPROBE_CHECKER_PYTHON}"
echo "sandbox_mount=${BANK_DIR}/policy:/workspace/sciprobe-probe:ro"
echo "nodes=${SLURM_NODES} gpus_per_node=${SLURM_GPUS_PER_NODE}"
echo "landlock_min_abi=${SCIPROBE_LANDLOCK_MIN_ABI:-3 (strict default)}"

if [[ "${DRY_RUN}" == "true" ]]; then
  printf 'sbatch_cmd='
  printf ' %q' "${SBATCH_CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${RUN_ROOT}/runtime" "${CHECKPOINT_DIR}" "${LOGGER_DIR}" \
         "${NEMO_GYM_LOG_DIR}" "${RAY_LOG_DIR}" "${UV_CACHE_DIR_OVERRIDE}" "${HF_HOME}"

for required in \
  "${MODEL_PATH}" \
  "${CONTAINER}" \
  "${SANDBOX_CONTAINER}" \
  "${BANK_DIR}/train.jsonl" \
  "${BANK_DIR}/sciprobe_checks.probes.yaml" \
  "${BANK_DIR}/policy" \
  "${BANK_DIR}/grader" \
  "${GRADING_PYTHON}" \
  "${TRAIN_PATH}" \
  "${CODE_DIR}/${CONFIG_PATH}"; do
  test -e "${required}"
done

# This run's own extensions tree, carrying only the probes it trains on.
"${GRADING_PYTHON}" "${CODE_DIR}/examples/splice_sciprobe_bank_probes.py" \
  --extensions "${CODE_DIR}/examples/nemo_gym_extensions" \
  --bank-probes "${BANK_DIR}/sciprobe_checks.probes.yaml" \
  --train-path "${TRAIN_PATH}" \
  --output "${RUNTIME_EXTENSIONS}"

"${SBATCH_CMD[@]}"
