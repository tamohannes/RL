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
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Account comes from SBATCH_ACCOUNT in the submit environment.
#SBATCH --partition=cpu
#SBATCH --time=00:15:00

set -euo pipefail
umask 077
export PATH="/cm/local/apps/slurm/current/bin:/cm/local/apps/python3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

: "${CODE_DIR:?CODE_DIR is required}"
: "${SANDBOX_CONTAINER:?SANDBOX_CONTAINER is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

mkdir -p "${OUTPUT_DIR}"
chmod 700 "${OUTPUT_DIR}"

sandbox_exports="NEMO_SKILLS_SANDBOX_BLOCK_NETWORK=1,SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK=1,PYTHONPATH=/workspace/sciprobe-seccomp-hook"
sandbox_env_names="NEMO_SKILLS_SANDBOX_BLOCK_NETWORK,SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK,PYTHONPATH"
srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --container-image="${SANDBOX_CONTAINER}" \
  --container-mounts="${CODE_DIR}/examples/validate_sciprobe_sandbox_seccomp.py:/workspace/validate-sciprobe-sandbox-seccomp.py:ro,${CODE_DIR}/examples/sandbox_seccomp_hook:/workspace/sciprobe-seccomp-hook:ro,${OUTPUT_DIR}:/workspace/output" \
  --no-container-mount-home \
  --export="${sandbox_exports}" \
  --container-env="${sandbox_env_names}" \
  /bin/bash -lc '
    set -euo pipefail
    python3 -c "import IPython, numpy, pandas; print(\"sandbox_notebook_dependencies=ok\")"
    python3 /workspace/validate-sciprobe-sandbox-seccomp.py \
      --hook-dir /workspace/sciprobe-seccomp-hook \
      | tee /workspace/output/result.json
  '

test -s "${OUTPUT_DIR}/result.json"
python3 - "${OUTPUT_DIR}/result.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "ok"
assert result["ipython_stateful_cells_allowed"] is True
assert result["numpy_pandas_csv_allowed"] is True
assert result["filter_active"] is True
assert result["process_filter_active"] is True
assert result["landlock_active"] is True
print(json.dumps({"status": "ok", "sandbox_seccomp_preflight": True}, sort_keys=True))
PY
