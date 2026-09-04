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

"""Create a private runtime copy of the one-row SciProbe canary dataset."""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from pathlib import Path

PROBE_ID = "q3:c013:d0"


def _ensure_private_store(path: Path, runtime_dir: Path) -> None:
    path = path.absolute()
    if path.parent != runtime_dir or path == runtime_dir / "train.jsonl":
        raise RuntimeError("capability store must be a separate runtime-private file")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        file_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or stat.S_IMODE(file_stat.st_mode) != 0o600
            or file_stat.st_uid != os.geteuid()
            or file_stat.st_nlink != 1
        ):
            raise RuntimeError("capability store must be a private mode-0600 file")
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capability-store", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError("signal canary source must contain exactly one JSON object")
    row = rows[0]
    if row.get("id") != PROBE_ID or row.get("probe_id") != PROBE_ID:
        raise RuntimeError("signal canary source has the wrong probe id")
    forbidden_fields = {
        "_sciprobe_verifier_capability",
        "sciprobe_capability",
    }
    if forbidden_fields.intersection(row):
        raise RuntimeError("signal dataset must not persist a verifier capability")

    output = args.output.absolute()
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output.parent, 0o700)
    _ensure_private_store(args.capability_store, output.parent)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    os.chmod(output, 0o600)

    persisted = json.loads(output.read_text(encoding="utf-8"))
    if persisted != row or forbidden_fields.intersection(persisted):
        raise RuntimeError("runtime dataset differs from the capability-free source")
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output),
                "probe_id": PROBE_ID,
                "rows": 1,
                "mode": oct(output.stat().st_mode & 0o777),
                "capability_store": str(args.capability_store.absolute()),
                "capability_store_mode": oct(
                    args.capability_store.stat().st_mode & 0o777
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
