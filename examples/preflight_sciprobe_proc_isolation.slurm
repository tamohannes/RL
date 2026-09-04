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
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
# Account comes from SBATCH_ACCOUNT in the submit environment.
#SBATCH --partition=cpu
#SBATCH --time=00:30:00

set -euo pipefail
umask 077
export PATH="/cm/local/apps/slurm/current/bin:/cm/local/apps/python3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

: "${CODE_DIR:?CODE_DIR is required}"
: "${DRIVER_CONTAINER:?DRIVER_CONTAINER is required}"
: "${SANDBOX_CONTAINER:?SANDBOX_CONTAINER is required}"
: "${PROBE_ROOT:?PROBE_ROOT is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

mkdir -p "${OUTPUT_DIR}"
chmod 700 "${OUTPUT_DIR}"
control_dir="${OUTPUT_DIR}/control-${SLURM_JOB_ID}"
private_dir="${OUTPUT_DIR}/private-${SLURM_JOB_ID}"
mkdir -p "${control_dir}" "${private_dir}"
chmod 700 "${control_dir}" "${private_dir}"

private_filename="private-$(openssl rand -hex 12).bin"
openssl rand 32 > "${private_dir}/${private_filename}"
chmod 600 "${private_dir}/${private_filename}"
env_sentinel="$(openssl rand -hex 32)"
primary_pid_file="${control_dir}/primary.pid"

SCIPROBE_POSITIVE_CONTROL_SENTINEL="${env_sentinel}" \
SCIPROBE_POSITIVE_CONTROL_PRIVATE_FILE="${private_dir}/${private_filename}" \
python3 "${CODE_DIR}/examples/validate_sciprobe_proc_positive_control.py" \
  "${control_dir}/bare-host-positive-control.json"

srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --container-image="${DRIVER_CONTAINER}" \
  --container-mounts="${CODE_DIR}:/workspace/RL:ro,${control_dir}:/workspace/control,${private_dir}:/workspace/private-sentinel:ro" \
  --no-container-mount-home \
  --export="SCIPROBE_PROC_ENV_SENTINEL=${env_sentinel}" \
  /bin/bash -lc '
    set -euo pipefail
    candidate=/opt/gym_venvs/resources_servers/ns_tools/.venv/bin/python
    "$candidate" -c "from nemo_skills.mcp.servers.python_tool import DirectPythonTool"
    exec "$candidate" /workspace/RL/examples/validate_sciprobe_proc_isolation.py \
      --serve \
      --variants baseline inner-unshare-pid \
      --control-dir /workspace/control \
      --primary-pid-file /workspace/control/primary.pid \
      --private-filename "$1"
  ' bash "${private_filename}" \
  >"${control_dir}/primary.log" 2>&1 &
primary_srun_pid=$!

cleanup() {
  kill "${primary_srun_pid}" >/dev/null 2>&1 || true
  wait "${primary_srun_pid}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 1200); do
  [[ -s "${primary_pid_file}" ]] && break
  kill -0 "${primary_srun_pid}" 2>/dev/null || {
    echo "primary sentinel process exited" >&2
    exit 1
  }
  sleep 1
done
[[ -s "${primary_pid_file}" ]]

run_variant() {
  local variant="$1"
  local sidecar_mode="$2"
  local variant_dir="${control_dir}/sandbox-${variant}"
  mkdir -p "${variant_dir}"
  local ready_file="${variant_dir}/ready"
  local stop_file="${variant_dir}/stop"
  local sandbox_exports="SANDBOX_PORTS_DIR=${variant_dir},SANDBOX_WORKER_BASE_PORT=6001,NGINX_PORT=6000,NUM_WORKERS=1,SANDBOX_FORCE_SINGLE_NODE=1,SCIPROBE_SIDECAR_READY_FILE=${ready_file},SCIPROBE_SIDECAR_STOP_FILE=${stop_file},SCIPROBE_SIDECAR_MODE=${sidecar_mode}"

  local -a srun_args=(
    srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=4
    --container-image="${SANDBOX_CONTAINER}"
    --container-mounts="${PROBE_ROOT}/data:/workspace/sciprobe-probe/data:ro,${variant_dir}:${variant_dir},${CODE_DIR}/examples/start_sciprobe_proc_preflight_sidecar.sh:/workspace/start-sidecar.sh:ro"
    --no-container-mount-home
    --export="${sandbox_exports}"
  )
  "${srun_args[@]}" /bin/bash /workspace/start-sidecar.sh \
    >"${variant_dir}/sidecar.log" 2>&1 &
  local sidecar_srun_pid=$!

  for _ in $(seq 1 300); do
    [[ -e "${ready_file}" ]] && break
    kill -0 "${sidecar_srun_pid}" 2>/dev/null || {
      echo "${variant} sidecar exited" >&2
      wait "${sidecar_srun_pid}" || true
      return 1
    }
    sleep 1
  done
  [[ -e "${ready_file}" ]]

  touch "${control_dir}/${variant}.request"
  for _ in $(seq 1 120); do
    [[ -s "${control_dir}/${variant}.json" ]] && break
    kill -0 "${primary_srun_pid}" 2>/dev/null || {
      echo "primary driver exited during ${variant}" >&2
      return 1
    }
    kill -0 "${sidecar_srun_pid}" 2>/dev/null || {
      echo "${variant} sidecar exited during proof" >&2
      wait "${sidecar_srun_pid}" || true
      return 1
    }
    sleep 1
  done
  [[ -s "${control_dir}/${variant}.json" ]]

  touch "${stop_file}"
  wait "${sidecar_srun_pid}"
}

run_variant baseline direct
run_variant inner-unshare-pid unshare-pid
wait "${primary_srun_pid}"

export SCIPROBE_LOG_CHECK_ENV_SENTINEL="${env_sentinel}"
export SCIPROBE_LOG_CHECK_PRIVATE_FILE="${private_dir}/${private_filename}"
python3 - "${control_dir}" "${OUTPUT_DIR}" <<'PY'
import json
import os
import sys
from pathlib import Path
root = Path(sys.argv[1])
output_root = Path(sys.argv[2])
env_sentinel = os.environ["SCIPROBE_LOG_CHECK_ENV_SENTINEL"].encode()
private_sentinel = Path(os.environ["SCIPROBE_LOG_CHECK_PRIVATE_FILE"]).read_bytes()
results = {}
for path in sorted(root.glob("*.json")):
    name = path.stem
    results[name] = json.loads((root / f"{name}.json").read_text())

positive = results["bare-host-positive-control"]["proof"]
assert positive == {
    "target_proc_visible": True,
    "target_environ_readable": True,
    "target_env_sentinel_found": True,
    "target_private_file_readable": True,
}

for variant in ("baseline", "inner-unshare-pid"):
    proof = results[variant]["proof"]
    assert proof["data_mount_valid"] is True
    assert proof["data_mount_write_blocked"] is True
    assert proof["target_env_sentinel_found"] is False
    assert proof["target_private_file_readable"] is False
    assert proof["any_sibling_env_sentinel_found"] is False
    assert proof["any_sibling_private_file_readable"] is False
    assert proof["private_file_readable_via_pid1_root"] is False
log_paths = [
    path
    for path in output_root.rglob("*")
    if path.is_file()
    and (path.suffix == ".log" or (path.name.startswith("slurm-") and path.suffix == ".out"))
]
log_bytes = b"".join(path.read_bytes() for path in log_paths)
sentinels_absent = env_sentinel not in log_bytes and private_sentinel not in log_bytes
assert sentinels_absent
print(json.dumps({"status": "ok", "sentinel_values_absent_from_logs": True, "results": results}, sort_keys=True))
PY
