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

# CPU-only rehearsal of the exact overfit-canary command prefix. It starts the
# production Ray and sandbox sidecars and runs every gate before model loading.

RUN_ID="${RUN_ID:-r4}"
EXP_NAME="sciprobe_rl_overfit-prefix-preflight-${RUN_ID}"

CODE_DIR="$(realpath "${CODE_DIR:-$PWD}")"
CONTAINER_CODE_DIR="${CONTAINER_CODE_DIR:-/workspace/RL}"
CONFIG_PATH="${CONFIG_PATH:-examples/configs/recipes/llm/grpo-nemotron3.5-lightning-30ba3b-1n4g-automodel-sciprobe-overfit-canary.yaml}"
TRAIN_PATH="${TRAIN_PATH:-examples/data/sciprobe/stateful-choice-overfit-canary.jsonl}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the Lightning checkpoint directory}"
CONTAINER="${CONTAINER:?set CONTAINER}"
SANDBOX_CONTAINER="${SANDBOX_CONTAINER:?set SANDBOX_CONTAINER to a pinned sandbox image}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:?set PERSISTENT_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PERSISTENT_ROOT}/preflights/overfit-command-prefix-${RUN_ID}}"
RUNTIME_TRAIN_PATH="${OUTPUT_ROOT}/runtime/train.jsonl"
SCIPROBE_CAPABILITY_STORE_PATH="${OUTPUT_ROOT}/runtime/capability-results.sqlite3"
RAY_LOG_DIR="${OUTPUT_ROOT}/ray"

SLURM_ACCOUNT="${SLURM_ACCOUNT:?set SLURM_ACCOUNT}"
SLURM_PARTITION="${SLURM_PARTITION:-cpu}"
SLURM_TIME_LIMIT="${SLURM_TIME_LIMIT:-00:30:00}"
RESUME="${RESUME:-false}"

SCIPROBE_VERIFIER_TOKEN="$(openssl rand -hex 32)"
SCIPROBE_CAPABILITY_SIGNING_KEY="$(openssl rand -hex 32)"
SCIPROBE_TRUSTED_INGRESS_TOKEN="$(openssl rand -hex 32)"
SCIPROBE_POLICY_GENERATION_TOKEN="$(openssl rand -hex 32)"
export SCIPROBE_VERIFIER_TOKEN
export SCIPROBE_CAPABILITY_SIGNING_KEY
export SCIPROBE_TRUSTED_INGRESS_TOKEN
export SCIPROBE_POLICY_GENERATION_TOKEN
unset SCIPROBE_VERIFIER_CAPABILITY

export SCIPROBE_RUNTIME_DATASET_PATH="${RUNTIME_TRAIN_PATH}"
export SCIPROBE_CAPABILITY_STORE_PATH
export CONTAINER
export SANDBOX_CONTAINER
export SANDBOX_COMMAND="unshare --pid --fork --mount-proc --kill-child /workspace/start-sciprobe-loopback-sandbox.sh"
export SANDBOX_EXTRA_MOUNTS="${CODE_DIR}/examples/sandbox_seccomp_hook:/workspace/sciprobe-seccomp-hook:ro,${CODE_DIR}/examples/validate_sciprobe_sandbox_seccomp.py:/workspace/validate-sciprobe-sandbox-seccomp.py:ro,${CODE_DIR}/examples/start_sciprobe_loopback_sandbox.sh:/workspace/start-sciprobe-loopback-sandbox.sh:ro"
export SANDBOX_ENV_VARS="NEMO_SKILLS_SANDBOX_BLOCK_NETWORK=1,SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK=1,PYTHONPATH=/workspace/sciprobe-seccomp-hook,NUM_WORKERS=1,SANDBOX_FORCE_SINGLE_NODE=1"
export SCIPROBE_REQUIRE_SANDBOX_SECCOMP_PREFLIGHT=1
export NEMO_SKILLS_SANDBOX_HOST=127.0.0.1
export NEMO_SKILLS_SANDBOX_PORT=6000
export NEMO_GYM_VENV_DIR=/opt/gym_venvs
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NEMO_GYM_EXTRA_ROOTS="${CONTAINER_CODE_DIR}/examples/nemo_gym_extensions"
export GPUS_PER_NODE=1
export CPUS_PER_WORKER=16
export RAY_LOOPBACK_ONLY=1
export NRL_FORCE_REBUILD_VENVS=false
export BASE_LOG_DIR="${RAY_LOG_DIR}"
export CONTAINER_WORKDIR="${CONTAINER_CODE_DIR}"
export MOUNTS="${CODE_DIR}:${CONTAINER_CODE_DIR},${MODEL_PATH}:${MODEL_PATH}:ro,${PERSISTENT_ROOT}:${PERSISTENT_ROOT}"

export COMMAND="export PATH=/opt/uv/bin:/opt/nemo_rl_venv/bin:\${PATH} && \
  export PYTHONPATH=${CONTAINER_CODE_DIR}:${CONTAINER_CODE_DIR}/3rdparty/Gym-workspace/Gym:\${PYTHONPATH:-} && \
  export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv && \
  export UV_PYTHON_INSTALL_DIR=/opt/uv-python && \
  export NEMO_RL_VENV_DIR=/opt/ray_venvs && \
  export NEMO_GYM_VENV_DIR=/opt/gym_venvs && \
  export NEMO_GYM_EXTRA_ROOTS=${CONTAINER_CODE_DIR}/examples/nemo_gym_extensions && \
  export SCIPROBE_CAPABILITY_STORE_PATH=${SCIPROBE_CAPABILITY_STORE_PATH} && \
  export NRL_FORCE_REBUILD_VENVS=false && \
  cd ${CONTAINER_CODE_DIR} && \
  ray_session_log_dir=\$(readlink -f /tmp/ray/session_latest/logs) && \
  test -d \"\${ray_session_log_dir}\" && \
  uv run --locked --no-sync python examples/validate_sciprobe_ray_loopback.py \
    --require-worker --require-token-auth --ray-log-dir \"\${ray_session_log_dir}\" && \
  test -x /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python && \
  /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python -c 'import nemo_gym, openai' && \
  /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python \
    examples/validate_sciprobe_canary_configs.py && \
  /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python \
    examples/validate_sciprobe_overfit_canary_config.py --config ${CONFIG_PATH} && \
  /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python \
    examples/validate_sciprobe_no_replay.py && \
  test -x /opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python && \
  /opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python \
    examples/validate_lightning_mtp_disabled.py --model ${MODEL_PATH} && \
  uv run --locked --no-sync python examples/validate_lightning_tool_tokenization.py \
    --model ${MODEL_PATH} --config ${CONFIG_PATH} && \
  uv run --locked --no-sync python examples/validate_sciprobe_overfit_reward.py && \
  test -x /opt/gym_venvs/resources_servers/sciprobe_ns_tools/.venv/bin/python && \
  /opt/gym_venvs/resources_servers/sciprobe_ns_tools/.venv/bin/python \
    examples/validate_sciprobe_signal_canary_auth.py && \
  /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python \
    examples/validate_nemo_gym_identity_token_audit.py && \
  uv run --locked --no-sync ruff check \
    nemo_rl/environments/nemo_gym.py \
    tests/unit/environments/test_nemo_gym.py \
    examples/validate_nemo_gym_identity_token_audit.py \
    examples/validate_lightning_tool_tokenization.py \
    examples/materialize_sciprobe_overfit_runtime_dataset.py \
    examples/validate_sciprobe_overfit_canary_config.py \
    examples/validate_sciprobe_overfit_canary_outputs.py \
    examples/validate_sciprobe_overfit_reward.py \
    examples/validate_sciprobe_overfit_reward_profile.py \
    examples/nemo_gym_extensions/resources_servers/sciprobe_overfit_checks && \
  echo '[SCIPROBE_OVERFIT_PREFIX_PREFLIGHT_OK]'"

if [[ "${RESUME}" != "true" && -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to reuse fresh preflight root: ${OUTPUT_ROOT}" >&2
  exit 1
fi
for required_path in \
  "${CONTAINER}" \
  "${SANDBOX_CONTAINER}" \
  "${MODEL_PATH}" \
  "${CODE_DIR}/examples/sandbox_seccomp_hook/sitecustomize.py" \
  "${CODE_DIR}/examples/validate_sciprobe_sandbox_seccomp.py" \
  "${CODE_DIR}/examples/start_sciprobe_loopback_sandbox.sh" \
  "${CODE_DIR}/${CONFIG_PATH}" \
  "${CODE_DIR}/${TRAIN_PATH}"; do
  [[ -e "${required_path}" ]] || {
    echo "Missing required path: ${required_path}" >&2
    exit 1
  }
done

mkdir -p "${OUTPUT_ROOT}/runtime" "${RAY_LOG_DIR}"
chmod 700 "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/runtime"
python3 "${CODE_DIR}/examples/materialize_sciprobe_overfit_runtime_dataset.py" \
  --source "${CODE_DIR}/${TRAIN_PATH}" \
  --output "${RUNTIME_TRAIN_PATH}" \
  --capability-store "${SCIPROBE_CAPABILITY_STORE_PATH}"
test "$(stat -c '%a' "${RUNTIME_TRAIN_PATH}")" = "600"
test "$(stat -c '%a' "${SCIPROBE_CAPABILITY_STORE_PATH}")" = "600"

echo "expname=${EXP_NAME}"
echo "output_dir=${OUTPUT_ROOT}"
cd "${CODE_DIR}"
sbatch \
  --nodes=1 \
  --cpus-per-task=16 \
  --mem=64G \
  --account="${SLURM_ACCOUNT}" \
  --job-name="${EXP_NAME}" \
  --partition="${SLURM_PARTITION}" \
  --time="${SLURM_TIME_LIMIT}" \
  --exclusive \
  --output="${OUTPUT_ROOT}/slurm-%j.out" \
  --error="${OUTPUT_ROOT}/slurm-%j.err" \
  ray.sub
