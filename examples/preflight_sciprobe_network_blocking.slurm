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
#SBATCH --exclusive

set -euo pipefail
umask 077
export PATH="/cm/local/apps/slurm/current/bin:/cm/local/apps/python3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

: "${CODE_DIR:?CODE_DIR is required}"
: "${DRIVER_CONTAINER:?DRIVER_CONTAINER is required}"
: "${SANDBOX_CONTAINER:?SANDBOX_CONTAINER is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

mkdir -p "${OUTPUT_DIR}"
chmod 700 "${OUTPUT_DIR}"
control_dir="${OUTPUT_DIR}/control-${SLURM_JOB_ID}"
mkdir -p "${control_dir}"
chmod 700 "${control_dir}"
driver_ready="${control_dir}/driver.ready"
sandbox_ready="${control_dir}/sandbox.ready"
result_file="${control_dir}/network-proof.json"

srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --container-image="${DRIVER_CONTAINER}" \
  --container-mounts="${CODE_DIR}:/workspace/RL:ro,${control_dir}:/workspace/control" \
  --no-container-mount-home \
  /bin/bash -lc '
    set -euo pipefail
    candidate=/opt/gym_venvs/resources_servers/ns_tools/.venv/bin/python
    "$candidate" -c "from nemo_skills.mcp.servers.python_tool import DirectPythonTool"
    PYTHONPATH=/workspace/RL/3rdparty/Gym-workspace/Gym \
      "$candidate" /workspace/RL/examples/validate_sciprobe_no_replay.py
    touch /workspace/control/driver.ready
    for _ in $(seq 1 600); do
      [[ -e /workspace/control/sandbox.ready ]] && break
      sleep 1
    done
    [[ -e /workspace/control/sandbox.ready ]]
    remote_host="$(hostname -I)"
    remote_host="${remote_host%% *}"
    exec "$candidate" /workspace/RL/examples/validate_sciprobe_network_blocking.py \
      --remote-sandbox-host "$remote_host" \
      --output /workspace/control/network-proof.json
  ' >"${control_dir}/driver.log" 2>&1 &
driver_srun_pid=$!

sidecar_srun_pid=""
cleanup() {
  if [[ -n "${sidecar_srun_pid}" ]]; then
    kill -KILL "${sidecar_srun_pid}" >/dev/null 2>&1 || true
    wait "${sidecar_srun_pid}" >/dev/null 2>&1 || true
  fi
  kill "${driver_srun_pid}" >/dev/null 2>&1 || true
  wait "${driver_srun_pid}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 1200); do
  [[ -e "${driver_ready}" ]] && break
  kill -0 "${driver_srun_pid}" 2>/dev/null || {
    echo "driver container exited before readiness" >&2
    exit 1
  }
  sleep 1
done
[[ -e "${driver_ready}" ]]

sandbox_exports="SANDBOX_PORTS_DIR=/tmp/sciprobe-preflight-${SLURM_JOB_ID},SANDBOX_WORKER_BASE_PORT=6001,NGINX_PORT=6000,NUM_WORKERS=1,SANDBOX_FORCE_SINGLE_NODE=1,NEMO_SKILLS_SANDBOX_BLOCK_NETWORK=1,SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK=1,PYTHONPATH=/workspace/sciprobe-seccomp-hook,SCIPROBE_SIDECAR_MODE=unshare-pid,SCIPROBE_SANDBOX_START_COMMAND=/workspace/start-loopback-sandbox.sh"
sandbox_env_names="SANDBOX_PORTS_DIR,SANDBOX_WORKER_BASE_PORT,NGINX_PORT,NUM_WORKERS,SANDBOX_FORCE_SINGLE_NODE,NEMO_SKILLS_SANDBOX_BLOCK_NETWORK,SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK,PYTHONPATH,SCIPROBE_SIDECAR_MODE,SCIPROBE_SANDBOX_START_COMMAND"
srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --container-image="${SANDBOX_CONTAINER}" \
  --container-mounts="${CODE_DIR}/examples/start_sciprobe_proc_preflight_sidecar.sh:/workspace/start-sidecar.sh:ro,${CODE_DIR}/examples/start_sciprobe_loopback_sandbox.sh:/workspace/start-loopback-sandbox.sh:ro,${CODE_DIR}/examples/validate_sciprobe_sandbox_seccomp.py:/workspace/validate-sciprobe-sandbox-seccomp.py:ro,${CODE_DIR}/examples/sandbox_seccomp_hook:/workspace/sciprobe-seccomp-hook:ro" \
  --no-container-mount-home \
  --export="${sandbox_exports}" \
  --container-env="${sandbox_env_names}" \
  /bin/bash /workspace/start-sidecar.sh \
  >"${control_dir}/sidecar.log" 2>&1 &
sidecar_srun_pid=$!

for _ in $(seq 1 300); do
  if grep -Fqx "[SCIPROBE_SANDBOX_SECCOMP_PREFLIGHT_OK]" "${control_dir}/sidecar.log" 2>/dev/null \
    && grep -Fqx "[SCIPROBE_SANDBOX_READY]" "${control_dir}/sidecar.log" 2>/dev/null; then
    break
  fi
  kill -0 "${sidecar_srun_pid}" 2>/dev/null || {
    echo "network-blocked sidecar exited before readiness" >&2
    wait "${sidecar_srun_pid}" || true
    exit 1
  }
  sleep 1
done
grep -Fqx "[SCIPROBE_SANDBOX_SECCOMP_PREFLIGHT_OK]" "${control_dir}/sidecar.log"
grep -Fqx "[SCIPROBE_SANDBOX_READY]" "${control_dir}/sidecar.log"

touch "${sandbox_ready}"
for _ in $(seq 1 120); do
  [[ -s "${result_file}" ]] && break
  kill -0 "${driver_srun_pid}" 2>/dev/null || {
    echo "driver exited during network proof" >&2
    wait "${driver_srun_pid}" || true
    exit 1
  }
  kill -0 "${sidecar_srun_pid}" 2>/dev/null || {
    echo "sidecar exited during network proof" >&2
    wait "${sidecar_srun_pid}" || true
    exit 1
  }
  sleep 1
done
[[ -s "${result_file}" ]]

kill -KILL "${sidecar_srun_pid}" >/dev/null 2>&1 || true
wait "${sidecar_srun_pid}" >/dev/null 2>&1 || true
sidecar_srun_pid=""
wait "${driver_srun_pid}"

if (echo > /dev/tcp/127.0.0.1/6000) 2>/dev/null; then
  echo "sandbox port 6000 remains open after teardown" >&2
  exit 1
fi

python3 - "${result_file}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
print(json.dumps(result, sort_keys=True))
assert result["first"]["state_value"] == 40
assert result["first"]["landlock_active"] is True
assert result["second"]["state_value"] == 42
assert result["second"]["landlock_active"] is True
assert all(
    result["cross_session"][name] is True
    for name in (
        "landlock_active",
        "session_directories_distinct",
        "own_state_write_allowed",
        "sibling_read_denied",
        "sibling_write_denied",
        "sibling_list_denied",
        "sibling_delete_denied",
        "sibling_signal_zero_denied",
        "all_processes_signal_zero_denied",
        "sibling_tkill_denied",
        "sibling_tgkill_denied",
        "sibling_rt_sigqueueinfo_denied",
        "sibling_rt_tgsigqueueinfo_denied",
        "sibling_prlimit_denied",
        "sibling_sched_getaffinity_denied",
        "sibling_sched_setaffinity_denied",
        "sibling_sched_setparam_denied",
        "sibling_sched_setscheduler_denied",
        "sibling_sched_setattr_denied",
        "sibling_getpriority_denied",
        "sibling_setpriority_denied",
        "sibling_ioprio_get_denied",
        "sibling_ioprio_set_denied",
        "sysv_shmget_denied",
        "sysv_shmat_denied",
        "sysv_semget_denied",
        "sysv_msgget_denied",
        "process_madvise_denied",
        "process_mrelease_denied",
        "only_control_unix_fd",
        "global_tmp_list_denied",
        "global_tmp_create_denied",
        "global_var_tmp_list_denied",
        "global_var_tmp_create_denied",
        "global_shm_list_denied",
        "global_shm_create_denied",
    )
)
assert result["first"]["socket_fd_count"] == 1
assert result["first"]["non_unix_socket_fd_count"] == 0
assert result["second"]["socket_fd_count"] == 1
assert result["second"]["non_unix_socket_fd_count"] == 0
assert result["cross_session"]["socket_fd_count"] == 1
assert result["cross_session"]["non_unix_socket_fd_count"] == 0
assert result["sandbox_parent_listener_accepted_later_call"] is True
assert result["ingress"]["loopback_listener_only"] is True
assert result["ingress"]["remote_execute_denied"] is True
assert result["ingress"]["remote_list_sessions_denied"] is True
assert result["ingress"]["remote_delete_session_denied"] is True
assert result["security_gate_passed"] is True
PY
