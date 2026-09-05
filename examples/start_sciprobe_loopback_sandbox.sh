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

# NeMo-Skills' stock sandbox image exposes both nginx and its uWSGI workers on
# every node interface.  The SciProbe canary is single-node and exclusive, so
# keep the complete execution API on loopback instead.  Patch copies in /tmp;
# the image files remain unchanged and no credential enters the sandbox.
if [[ "${SANDBOX_FORCE_SINGLE_NODE:-0}" != "1" ]]; then
  echo "SciProbe loopback sandbox requires SANDBOX_FORCE_SINGLE_NODE=1" >&2
  exit 2
fi
if [[ "${NUM_WORKERS:-}" != "1" ]]; then
  echo "SciProbe loopback sandbox requires NUM_WORKERS=1" >&2
  exit 2
fi

# Validate the exact sandbox image and interpreter before opening the service.
# This wrapper is the hardened SciProbe entrypoint, so an unset requirement must
# fail instead of silently starting an unvalidated service. The container-local
# attestation survives stdout redirection and lets the trusted sidecar prove that
# this exact process passed the gate before it reports readiness.
if [[ "${SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK:-}" != "1" ]]; then
  echo "SciProbe loopback sandbox requires SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK=1" >&2
  exit 2
fi
: "${SANDBOX_PORTS_DIR:?SANDBOX_PORTS_DIR is required}"
# Follow the interpreter indirection when the image uses one.
#
# This runs before the seccomp preflight on purpose. The preflight asserts
# that numpy, pandas and stateful IPython still work under the filter, and
# those have to be checked against the interpreter that will actually run
# model code. Selecting it afterwards validated the image default instead,
# which on a science image is a different and barer interpreter: the checks
# numpy_pandas_csv_allowed and ipython_stateful_cells_allowed both failed
# for a missing package rather than for anything the filter did.
#
# The stock image puts the uWSGI bind template directly in /start-with-nginx.sh.
# A science-enriched image replaces that file with a short wrapper that prepends
# its own environment to PATH and execs /start-with-nginx.real.sh, because
# start-with-nginx launches uwsgi by bare name. Prepending the PATH is what
# swaps the interpreter the sandbox executes model code under, and it swaps the
# whole thing rather than grafting packages onto the stock one, so there is no
# mixed-ABI site-packages anywhere.
#
# Without this the patch below reads the 164-byte wrapper, finds no bind
# template and exits on "unexpected NeMo-Skills uWSGI bind template".
if [[ -r /start-with-nginx.real.sh ]]; then
  if [[ -x /app/kernel_env/bin/uwsgi ]]; then
    export PATH=/app/kernel_env/bin:${PATH}
  else
    echo "image has /start-with-nginx.real.sh but no /app/kernel_env/bin/uwsgi" >&2
    exit 2
  fi
  base_start=/start-with-nginx.real.sh
else
  base_start=/start-with-nginx.sh
fi

seccomp_validator=/workspace/validate-sciprobe-sandbox-seccomp.py
seccomp_hook_dir=/workspace/sciprobe-seccomp-hook
seccomp_ready_file="${SANDBOX_PORTS_DIR}/sciprobe-seccomp-preflight.ready"
if [[ ! -r "${seccomp_validator}" ]]; then
  echo "SciProbe sandbox seccomp validator is not mounted" >&2
  exit 2
fi
if [[ ! -r "${seccomp_hook_dir}/sitecustomize.py" ]]; then
  echo "SciProbe sandbox seccomp hook is not mounted" >&2
  exit 2
fi
mkdir -p "${SANDBOX_PORTS_DIR}"
chmod 700 "${SANDBOX_PORTS_DIR}"
rm -f -- "${seccomp_ready_file}"
python3 "${seccomp_validator}" --hook-dir "${seccomp_hook_dir}"
printf '%s\n' '[SCIPROBE_SANDBOX_SECCOMP_PREFLIGHT_OK]' > "${seccomp_ready_file}"
chmod 400 "${seccomp_ready_file}"
echo "[SCIPROBE_SANDBOX_SECCOMP_PREFLIGHT_OK]"

base_nginx=/etc/nginx/nginx.conf.template
patched_start=/tmp/sciprobe-start-with-nginx.sh
patched_nginx=/tmp/sciprobe-nginx.conf.template

python3 - "${base_start}" "${patched_start}" "${base_nginx}" "${patched_nginx}" <<'PY'
import sys
from pathlib import Path

base_start, patched_start, base_nginx, patched_nginx = map(Path, sys.argv[1:])

start_text = base_start.read_text(encoding="utf-8")
worker_public = "http-socket = 0.0.0.0:${WORKER_PORT}"
worker_loopback = "http-socket = 127.0.0.1:${WORKER_PORT}"
if start_text.count(worker_public) != 1:
    raise SystemExit("unexpected NeMo-Skills uWSGI bind template")
start_text = start_text.replace(worker_public, worker_loopback)

nginx_text = base_nginx.read_text(encoding="utf-8")
nginx_public = "listen ${NGINX_PORT};"
nginx_loopback = "listen 127.0.0.1:${NGINX_PORT};"
if nginx_text.count(nginx_public) != 1:
    raise SystemExit("unexpected NeMo-Skills nginx bind template")
nginx_text = nginx_text.replace(nginx_public, nginx_loopback)

template_path = str(patched_nginx)
if start_text.count(str(base_nginx)) != 1:
    raise SystemExit("unexpected NeMo-Skills nginx template reference")
start_text = start_text.replace(str(base_nginx), template_path)

patched_start.write_text(start_text, encoding="utf-8")
patched_nginx.write_text(nginx_text, encoding="utf-8")
PY

chmod 700 "${patched_start}"
exec "${patched_start}"
