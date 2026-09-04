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

# Fresh-process proof that step-6 weights reload and retain the learned reward
# shift. This job performs one 32-rollout validation and no optimizer step.

RUN_ID="${RUN_ID:-r1}"
[[ "${RUN_ID}" =~ ^r[1-9][0-9]*$ ]] || {
  echo "RUN_ID must match r<positive-integer>; received ${RUN_ID}" >&2
  exit 1
}
SOURCE_RUN_ID="${SOURCE_RUN_ID:-${RUN_ID}}"
[[ "${SOURCE_RUN_ID}" =~ ^r[1-9][0-9]*$ ]] || {
  echo "SOURCE_RUN_ID must match r<positive-integer>; received ${SOURCE_RUN_ID}" >&2
  exit 1
}
EXP_NAME="sciprobe_rl_lightning-overfit-checkpoint-reload-${RUN_ID}"

CODE_DIR="$(realpath "${CODE_DIR:-$PWD}")"
CONTAINER_CODE_DIR="${CONTAINER_CODE_DIR:-/workspace/RL}"
CONFIG_PATH="${CONFIG_PATH:-examples/configs/recipes/llm/grpo-nemotron3.5-lightning-30ba3b-1n4g-automodel-sciprobe-overfit-canary.yaml}"
TRAIN_PATH="${TRAIN_PATH:-examples/data/sciprobe/stateful-choice-overfit-canary.jsonl}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the Lightning checkpoint directory}"
CONTAINER="${CONTAINER:?set CONTAINER}"
SANDBOX_CONTAINER="${SANDBOX_CONTAINER:?set SANDBOX_CONTAINER to a pinned sandbox image}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:?set PERSISTENT_ROOT}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:?set SLURM_ACCOUNT}"
SLURM_PARTITION="${SLURM_PARTITION:-batch}"
SLURM_TIME_LIMIT="02:00:00"
DRY_RUN="${DRY_RUN:-false}"

SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${PERSISTENT_ROOT}/runs/lightning-overfit-canary-${SOURCE_RUN_ID}}"
SOURCE_CHECKPOINT_DIR="${SOURCE_RUN_ROOT}/checkpoints"
SOURCE_STEP_DIR="${SOURCE_CHECKPOINT_DIR}/step_6"
RUN_ROOT="${PERSISTENT_ROOT}/runs/lightning-overfit-checkpoint-reload-${RUN_ID}"
SOURCE_CHECKPOINT_MANIFEST="${RUN_ROOT}/source-checkpoint-manifest.json"
RUNTIME_TRAIN_PATH="${RUN_ROOT}/runtime/train.jsonl"
SCIPROBE_CAPABILITY_STORE_PATH="${RUN_ROOT}/runtime/capability-results.sqlite3"
LOGGER_DIR="${RUN_ROOT}/logs"
NEMO_GYM_LOG_DIR="${RUN_ROOT}/nemo-gym"
RAY_LOG_DIR="${RUN_ROOT}/ray"
UV_CACHE_DIR_OVERRIDE="${PERSISTENT_ROOT}/cache/uv"
HF_HOME="${PERSISTENT_ROOT}/cache/huggingface"

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
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-unshare --pid --fork --mount-proc --kill-child /workspace/start-sciprobe-loopback-sandbox.sh}"
export SANDBOX_EXTRA_MOUNTS="${CODE_DIR}/examples/sandbox_seccomp_hook:/workspace/sciprobe-seccomp-hook:ro,${CODE_DIR}/examples/validate_sciprobe_sandbox_seccomp.py:/workspace/validate-sciprobe-sandbox-seccomp.py:ro,${CODE_DIR}/examples/start_sciprobe_loopback_sandbox.sh:/workspace/start-sciprobe-loopback-sandbox.sh:ro"
export SANDBOX_ENV_VARS="NEMO_SKILLS_SANDBOX_BLOCK_NETWORK=1,SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK=1,PYTHONPATH=/workspace/sciprobe-seccomp-hook,NUM_WORKERS=1,SANDBOX_FORCE_SINGLE_NODE=1"
export SCIPROBE_REQUIRE_SANDBOX_SECCOMP_PREFLIGHT=1
export NEMO_SKILLS_SANDBOX_HOST=127.0.0.1
export NEMO_SKILLS_SANDBOX_PORT=6000
export NEMO_GYM_VENV_DIR=/opt/gym_venvs
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NEMO_GYM_EXTRA_ROOTS="${CONTAINER_CODE_DIR}/examples/nemo_gym_extensions"
export GPUS_PER_NODE=4
export RAY_LOOPBACK_ONLY=1
export NRL_FORCE_REBUILD_VENVS=false
export UV_CACHE_DIR_OVERRIDE
export HF_HOME
export WANDB_MODE=offline
export BASE_LOG_DIR="${RAY_LOG_DIR}"
export CONTAINER_WORKDIR="${CONTAINER_CODE_DIR}"
export MOUNTS="${CODE_DIR}:${CONTAINER_CODE_DIR},${MODEL_PATH}:${MODEL_PATH}:ro,${PERSISTENT_ROOT}:${PERSISTENT_ROOT},${SOURCE_CHECKPOINT_DIR}:${SOURCE_CHECKPOINT_DIR}:ro"

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
  source_checkpoint_mount_target=\$(findmnt --mountpoint ${SOURCE_CHECKPOINT_DIR} --noheadings --output TARGET) && \
  source_checkpoint_vfs_options=\$(findmnt --mountpoint ${SOURCE_CHECKPOINT_DIR} --noheadings --output VFS-OPTIONS) && \
  test \"\${source_checkpoint_mount_target}\" = ${SOURCE_CHECKPOINT_DIR} && \
  case \",\${source_checkpoint_vfs_options// /},\" in *,ro,*) ;; *) echo 'Source checkpoint mount is not read-only' >&2; exit 1 ;; esac && \
  NRL_VLLM_USE_V1=1 NRL_WG_USE_RAY_REF=1 UV_HTTP_TIMEOUT=300 \
  uv run --locked --no-sync python ./examples/nemo_gym/run_grpo_nemo_gym.py \
    --config ${CONFIG_PATH} \
    policy.model_name=${MODEL_PATH} \
    policy.tokenizer.name=${MODEL_PATH} \
    data.train.data_path=${RUNTIME_TRAIN_PATH} \
    data.validation.data_path=${RUNTIME_TRAIN_PATH} \
    checkpointing.checkpoint_dir=${SOURCE_CHECKPOINT_DIR} \
    logger.log_dir=${LOGGER_DIR} \
    logger.wandb.name=lightning-overfit-checkpoint-reload-${RUN_ID} \
    env.nemo_gym.nemo_gym_log_dir=${NEMO_GYM_LOG_DIR} \
    +env.nemo_gym.is_validation_only=true && \
  uv run --locked --no-sync python \
    examples/validate_sciprobe_overfit_checkpoint_reload_outputs.py \
    --source-run-root ${SOURCE_RUN_ROOT} \
    --reload-run-root ${RUN_ROOT} \
    --source-checkpoint-manifest ${SOURCE_CHECKPOINT_MANIFEST} \
    --expected-source-run-id ${SOURCE_RUN_ID} \
    --expected-step 6 \
    --expected-validation-rollouts 32"

SBATCH_CMD=(
  sbatch
  --nodes=1
  --account="${SLURM_ACCOUNT}"
  --job-name="${EXP_NAME}"
  --partition="${SLURM_PARTITION}"
  --time="${SLURM_TIME_LIMIT}"
  --gres=gpu:4
  --exclusive
  --output="${RUN_ROOT}/slurm-%j.out"
  --error="${RUN_ROOT}/slurm-%j.err"
  ray.sub
)

echo "expname=${EXP_NAME}"
echo "source_run_id=${SOURCE_RUN_ID}"
echo "source_checkpoint_dir=${SOURCE_CHECKPOINT_DIR}"
echo "source_checkpoint_mount=read-only"
echo "source_checkpoint_manifest=${SOURCE_CHECKPOINT_MANIFEST}"
echo "output_dir=${RUN_ROOT}"
echo "logger_dir=${LOGGER_DIR}"
echo "ray_log_dir=${RAY_LOG_DIR}"
echo "nemo_gym_log_dir=${NEMO_GYM_LOG_DIR}"
echo "runtime_train_path=${RUNTIME_TRAIN_PATH}"
echo "capability_store_path=${SCIPROBE_CAPABILITY_STORE_PATH}"
echo "validation_rollouts=32"
echo "optimizer_steps=0"

if [[ "${DRY_RUN}" == "true" ]]; then
  printf 'command=%s\n' "${COMMAND}"
  printf 'sbatch='
  printf ' %q' "${SBATCH_CMD[@]}"
  printf '\n'
  exit 0
fi

if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse fresh reload run root: ${RUN_ROOT}" >&2
  exit 1
fi
for required_path in \
  "${CONTAINER}" \
  "${SANDBOX_CONTAINER}" \
  "${MODEL_PATH}" \
  "${SOURCE_CHECKPOINT_DIR}/latest_checkpoint_status.json" \
  "${SOURCE_STEP_DIR}/training_info.json" \
  "${CODE_DIR}/examples/sandbox_seccomp_hook/sitecustomize.py" \
  "${CODE_DIR}/examples/validate_sciprobe_sandbox_seccomp.py" \
  "${CODE_DIR}/examples/start_sciprobe_loopback_sandbox.sh" \
  "${CODE_DIR}/${CONFIG_PATH}" \
  "${CODE_DIR}/${TRAIN_PATH}" \
  "${CODE_DIR}/examples/validate_sciprobe_overfit_checkpoint_reload_outputs.py"; do
  [[ -e "${required_path}" ]] || {
    echo "Missing required path: ${required_path}" >&2
    exit 1
  }
done

python3 - "${SOURCE_CHECKPOINT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

checkpoint_root = Path(sys.argv[1])
steps = {}
for path in checkpoint_root.iterdir():
    suffix = path.name.removeprefix("step_")
    if path.is_dir() and path.name.startswith("step_") and suffix.isdigit():
        steps[int(suffix)] = path
if not steps or max(steps) != 6:
    raise RuntimeError(f"latest source checkpoint must be step 6; found {sorted(steps)}")
training_info = json.loads((steps[6] / "training_info.json").read_text())
if training_info.get("total_steps") != 6:
    raise RuntimeError("source step-6 training_info.total_steps must equal 6")
status = json.loads((checkpoint_root / "latest_checkpoint_status.json").read_text())
if status.get("last_checkpoint_step") != 6:
    raise RuntimeError("source latest_checkpoint_status must name step 6")
PY

mkdir -p \
  "${RUN_ROOT}/runtime" \
  "${LOGGER_DIR}" \
  "${NEMO_GYM_LOG_DIR}" \
  "${RAY_LOG_DIR}" \
  "${UV_CACHE_DIR_OVERRIDE}" \
  "${HF_HOME}"
chmod 700 "${RUN_ROOT}/runtime"
python3 - "${SOURCE_CHECKPOINT_DIR}" "${SOURCE_CHECKPOINT_MANIFEST}" <<'PY'
import hashlib
import json
import stat
import sys
from pathlib import Path

checkpoint_root = Path(sys.argv[1]).resolve(strict=True)
manifest_path = Path(sys.argv[2])
directories = []
files = []
for path in sorted(
    checkpoint_root.rglob("*"),
    key=lambda candidate: candidate.relative_to(checkpoint_root).as_posix(),
):
    path_stat = path.lstat()
    relative_path = path.relative_to(checkpoint_root).as_posix()
    if stat.S_ISLNK(path_stat.st_mode):
        raise RuntimeError(f"source checkpoint contains a symlink: {relative_path}")
    if stat.S_ISDIR(path_stat.st_mode):
        directories.append(relative_path)
        continue
    if not stat.S_ISREG(path_stat.st_mode):
        raise RuntimeError(
            f"source checkpoint contains a non-regular entry: {relative_path}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    files.append(
        {
            "path": relative_path,
            "bytes": path_stat.st_size,
            "sha256": digest.hexdigest(),
        }
    )

step_inventory = sorted(
    int(path.name.removeprefix("step_"))
    for path in checkpoint_root.iterdir()
    if path.is_dir()
    and path.name.startswith("step_")
    and path.name.removeprefix("step_").isdigit()
)
manifest = {
    "version": 1,
    "checkpoint_root": str(checkpoint_root),
    "step_inventory": step_inventory,
    "directories": directories,
    "files": files,
}
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
manifest_path.chmod(0o600)
print(
    "source_checkpoint_manifest="
    f"steps={step_inventory} files={len(files)} "
    f"bytes={sum(item['bytes'] for item in files)}"
)
PY
test "$(stat -c '%a' "${SOURCE_CHECKPOINT_MANIFEST}")" = "600"
python3 "${CODE_DIR}/examples/materialize_sciprobe_overfit_runtime_dataset.py" \
  --source "${CODE_DIR}/${TRAIN_PATH}" \
  --output "${RUNTIME_TRAIN_PATH}" \
  --capability-store "${SCIPROBE_CAPABILITY_STORE_PATH}"
test "$(stat -c '%a' "${RUNTIME_TRAIN_PATH}")" = "600"
test "$(stat -c '%a' "${SCIPROBE_CAPABILITY_STORE_PATH}")" = "600"
cd "${CODE_DIR}"
"${SBATCH_CMD[@]}"
