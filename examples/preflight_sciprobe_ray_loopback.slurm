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
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Account comes from SBATCH_ACCOUNT in the submit environment.
#SBATCH --partition=cpu
#SBATCH --time=00:20:00
#SBATCH --exclusive

set -euo pipefail
umask 077

: "${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"
: "${CODE_DIR:?CODE_DIR is required}"

srun \
  --ntasks=1 \
  --cpus-per-task=8 \
  --exact \
  --kill-on-bad-exit=1 \
  --no-container-mount-home \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CODE_DIR}:/workspace/RL:ro" \
  --container-workdir=/workspace/RL \
  bash -lc '
    set -euo pipefail
    export PATH=/opt/uv/bin:/opt/nemo_rl_venv/bin:/root/.local/bin:${PATH}
    export PYTHONPATH=/workspace/RL:${PYTHONPATH:-}
    export UV_PYTHON_INSTALL_DIR=/opt/uv-python
    export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
    export RAY_ADDRESS="ray://192.0.2.10:10001"
    export RAY_REDIS_ADDRESS="192.0.2.11:6379"
    export RAY_API_SERVER_ADDRESS="http://192.0.2.12:8265"
    export RAY_DASHBOARD_ADDRESS="http://192.0.2.13:8265"
    export RAY_AGENT_ADDRESS="192.0.2.14:52365"
    unset RAY_REDIS_ADDRESS
    unset RAY_API_SERVER_ADDRESS
    unset RAY_DASHBOARD_ADDRESS
    unset RAY_AGENT_ADDRESS
    export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0
    export RAY_ADDRESS="127.0.0.1:1200"
    [[ "${RAY_ADDRESS}" == "127.0.0.1:1200" ]]
    for forbidden in \
      RAY_REDIS_ADDRESS \
      RAY_API_SERVER_ADDRESS \
      RAY_DASHBOARD_ADDRESS \
      RAY_AGENT_ADDRESS; do
      [[ -z "${!forbidden+x}" ]]
    done
    ray_tmp_dir="/tmp/sciprobe-ray-${SLURM_JOB_ID}"
    ray_auth_dir="/tmp/sciprobe-ray-auth-${SLURM_JOB_ID}"
    ray_auth_token_path="${ray_auth_dir}/token"
    cleanup() {
      original_rc=$?
      trap - EXIT
      ray stop --force >/dev/null 2>&1 || true
      scan_rc=0
      if [[ -f "${ray_auth_token_path}" && -d "${ray_tmp_dir}" ]]; then
        uv run --locked --no-sync python - \
          "${ray_auth_token_path}" "${ray_tmp_dir}" <<'"'"'PY'"'"' || scan_rc=$?
import json
import sys
from pathlib import Path

from examples.validate_sciprobe_ray_loopback import _scan_token_in_tree

token_path = Path(sys.argv[1]).resolve(strict=True)
result = _scan_token_in_tree(token_path, token_path.read_bytes(), Path(sys.argv[2]))
print("RAY_TOKEN_FINAL_LOG_SCAN " + json.dumps(result, sort_keys=True))
PY
      fi
      rm -f "${ray_auth_token_path}"
      rmdir "${ray_auth_dir}" 2>/dev/null || true
      rm -rf "${ray_tmp_dir}"
      if [[ "${scan_rc}" -ne 0 ]]; then
        exit "${scan_rc}"
      fi
      exit "${original_rc}"
    }
    trap cleanup EXIT
    ray stop --force >/dev/null 2>&1 || true
    rm -rf "${ray_tmp_dir}"
    rm -rf "${ray_auth_dir}"
    python - "${ray_auth_dir}" "${ray_auth_token_path}" <<'"'"'PY'"'"'
import os
import secrets
import stat
import sys
from pathlib import Path

auth_dir = Path(sys.argv[1])
token_path = Path(sys.argv[2])
auth_dir.mkdir(mode=0o700)
fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="ascii") as stream:
    stream.write(secrets.token_hex(32))
    stream.flush()
    os.fsync(stream.fileno())
if stat.S_IMODE(auth_dir.stat().st_mode) != 0o700:
    raise RuntimeError("Ray auth directory mode is not 0700")
if stat.S_IMODE(token_path.stat().st_mode) != 0o600:
    raise RuntimeError("Ray auth token mode is not 0600")
PY
    unset RAY_AUTH_TOKEN
    export RAY_AUTH_MODE=token
    export RAY_AUTH_TOKEN_PATH="${ray_auth_token_path}"
    ray start --head \
      --temp-dir="${ray_tmp_dir}" \
      --disable-usage-stats \
      --num-cpus=8 \
      --node-ip-address=127.0.0.1 \
      --port=1200 \
      --ray-client-server-port=1201 \
      --dashboard-port=8265 \
      --dashboard-host=127.0.0.1 \
      --include-dashboard=True \
      --min-worker-port=2000 \
      --max-worker-port=2999 \
      --node-manager-port=1302 \
      --object-manager-port=1304 \
      --runtime-env-agent-port=1306 \
      --dashboard-agent-grpc-port=1308 \
      --dashboard-agent-listen-port=1312 \
      --metrics-export-port=1310
    ray status
    ray_session_log_dir="$(readlink -f "${ray_tmp_dir}/session_latest/logs")"
    [[ -d "${ray_session_log_dir}" ]]
    uv run --locked --no-sync python examples/validate_sciprobe_ray_loopback.py \
      --require-worker \
      --require-token-auth \
      --ray-log-dir "${ray_session_log_dir}"
  '
