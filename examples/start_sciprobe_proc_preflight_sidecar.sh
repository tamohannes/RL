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

: "${SCIPROBE_SIDECAR_MODE:?}"
SCIPROBE_SANDBOX_START_COMMAND="${SCIPROBE_SANDBOX_START_COMMAND:-/start-with-nginx.sh}"

case "${SCIPROBE_SIDECAR_MODE}" in
  direct)
    "${SCIPROBE_SANDBOX_START_COMMAND}" &
    ;;
  unshare-pid)
    command -v unshare >/dev/null
    unshare --pid --fork --mount-proc --kill-child \
      "${SCIPROBE_SANDBOX_START_COMMAND}" &
    ;;
  *)
    echo "unknown sidecar mode" >&2
    exit 2
    ;;
esac
sandbox_pid=$!
cleanup() {
  kill -KILL "${sandbox_pid}" 2>/dev/null || true
  wait "${sandbox_pid}" 2>/dev/null || true
}
trap cleanup EXIT

deadline=$((SECONDS + 300))
while ! (echo > /dev/tcp/127.0.0.1/6000) 2>/dev/null; do
  if ! kill -0 "${sandbox_pid}" 2>/dev/null; then
    echo "sandbox exited before readiness" >&2
    wait "${sandbox_pid}"
    exit 1
  fi
  if (( SECONDS > deadline )); then
    echo "sandbox readiness timeout" >&2
    exit 1
  fi
  sleep 1
done
if [[ "${SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK:-}" == "1" ]]; then
  seccomp_ready_file="${SANDBOX_PORTS_DIR:?}/sciprobe-seccomp-preflight.ready"
  if [[ ! -r "${seccomp_ready_file}" ]] \
    || ! grep -Fqx "[SCIPROBE_SANDBOX_SECCOMP_PREFLIGHT_OK]" "${seccomp_ready_file}"; then
    echo "sandbox service opened without a valid seccomp preflight attestation" >&2
    exit 1
  fi
  echo "[SCIPROBE_SANDBOX_SECCOMP_PREFLIGHT_OK]"
fi
echo "[SCIPROBE_SANDBOX_READY]"
wait "${sandbox_pid}"
