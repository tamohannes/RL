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

"""Prove that the host-side /proc sentinel checks detect an accessible child."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_sciprobe_proc_positive_control.py OUTPUT")

    sentinel = os.environ["SCIPROBE_POSITIVE_CONTROL_SENTINEL"]
    private_file = Path(os.environ["SCIPROBE_POSITIVE_CONTROL_PRIVATE_FILE"])
    expected_private = private_file.read_bytes()
    child_env = os.environ.copy()
    child_env["SCIPROBE_PROC_ENV_SENTINEL"] = sentinel
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        env=child_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        environ_path = Path("/proc") / str(child.pid) / "environ"
        root_private_path = (
            Path("/proc") / str(child.pid) / "root" / private_file.relative_to("/")
        )
        for _ in range(50):
            try:
                environ = environ_path.read_bytes()
                private_contents = root_private_path.read_bytes()
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("positive-control child did not become readable")

        encoded_assignment = f"SCIPROBE_PROC_ENV_SENTINEL={sentinel}".encode()
        proof = {
            "target_proc_visible": environ_path.parent.exists(),
            "target_environ_readable": True,
            "target_env_sentinel_found": encoded_assignment in environ.split(b"\0"),
            "target_private_file_readable": private_contents == expected_private,
        }
        assert all(proof.values()), proof
        result = {
            "status": "ok",
            "variant": "bare-host-positive-control",
            "proof": proof,
        }
        output = Path(sys.argv[1])
        temporary_output = output.with_suffix(output.suffix + ".tmp")
        temporary_output.write_text(
            json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_output, output)
        print(json.dumps(result, sort_keys=True))
    finally:
        child.kill()
        child.wait()


if __name__ == "__main__":
    main()
