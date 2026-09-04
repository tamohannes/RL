#!/usr/bin/env python3
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

"""Execute the ray.sub TERM cleanup regression proofs with only the stdlib."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAY_SUB = ROOT / "ray.sub"


def _run_term_script(contents: str, path: Path) -> int:
    path.write_text(contents, encoding="utf-8")
    return subprocess.run(["bash", str(path)], check=False).returncode


def main() -> None:
    ray_sub = RAY_SUB.read_text(encoding="utf-8")
    outer_start = ray_sub.index("_nrl_log_exit()")
    outer_end = ray_sub.index("\n\n", ray_sub.index("trap _nrl_log_exit EXIT"))
    outer = ray_sub[outer_start:outer_end]

    head_start = ray_sub.index("_nrl_head_exit()")
    head_end = ray_sub.index("\n\n", ray_sub.index("trap _nrl_head_exit EXIT"))
    head = ray_sub[head_start:head_end].replace("\\$", "$")

    with tempfile.TemporaryDirectory(prefix="sciprobe-ray-signals-") as tmp:
        tmp_path = Path(tmp)
        outer_rc = _run_term_script(
            "#!/usr/bin/env bash\n"
            "set -eu\n"
            "RAY_AUTH_MODE=disabled\n"
            "log_phase() { :; }\n"
            "_nrl_stop_background_sruns() { :; }\n"
            "_nrl_scan_ray_auth_logs() { :; }\n" + outer + "\nkill -TERM $$\nexit 0\n",
            tmp_path / "outer-term-cleanup.sh",
        )

        head_logs = tmp_path / "head-logs"
        head_logs.mkdir()
        head_rc = _run_term_script(
            "#!/usr/bin/env bash\n"
            "set -eu\n"
            f"LOG_DIR={head_logs}\n"
            "RAY_AUTH_MODE=disabled\n"
            "HEAD_SIDECAR_PIDS=()\n"
            "_nrl_scan_ray_auth_logs() { :; }\n" + head + "\nkill -TERM $$\nexit 0\n",
            tmp_path / "head-term-cleanup.sh",
        )

    if outer_rc != 143 or head_rc != 143:
        raise RuntimeError(
            f"TERM cleanup exit mismatch: outer={outer_rc}, head={head_rc}"
        )
    print(json.dumps({"head_term_exit": head_rc, "outer_term_exit": outer_rc}))


if __name__ == "__main__":
    main()
