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

"""Stage pinned SciProbe data into an idempotent private root."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

EXPECTED_DATA_SHA256 = (
    "16713f67f959a4c276baea508c1fb64fa54bf622f4e14b0b4def77d6c152a590"
)
EXPECTED_DATA_FILES = 25
EXPECTED_DATA_BYTES = 8137
EXPECTED_TOP_ENTRIES = {"data"}


def _validate_tree(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"data must be a real directory: {root}")
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"data contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"data contains a non-regular entry: {relative}")
        contents = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(contents)).encode("ascii"))
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
        files += 1
        total_bytes += len(contents)
    return digest.hexdigest(), files, total_bytes


def _validate_staged(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"staged root must be a real directory: {root}")
    top_entries = {path.name for path in root.iterdir()}
    if top_entries != EXPECTED_TOP_ENTRIES:
        raise RuntimeError(
            f"staged root must contain only data, got {sorted(top_entries)}"
        )
    data_hash, files, total_bytes = _validate_tree(root / "data")
    expected = (EXPECTED_DATA_SHA256, EXPECTED_DATA_FILES, EXPECTED_DATA_BYTES)
    actual = (data_hash, files, total_bytes)
    if actual != expected:
        raise RuntimeError(f"data tree drift: got {actual}, expected {expected}")


def _validate_source(source: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise RuntimeError(f"source root must be a real directory: {source}")
    data_hash, files, total_bytes = _validate_tree(source / "data")
    if (data_hash, files, total_bytes) != (
        EXPECTED_DATA_SHA256,
        EXPECTED_DATA_FILES,
        EXPECTED_DATA_BYTES,
    ):
        raise RuntimeError("source data does not match the pinned probe")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    destination = args.destination.absolute()
    _validate_source(source)

    if destination.exists() or destination.is_symlink():
        _validate_staged(destination)
        print(
            f"validated existing staged probe: {destination} "
            f"({EXPECTED_DATA_FILES} files, {EXPECTED_DATA_BYTES} bytes)"
        )
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.stage-", dir=destination.parent
    ) as temp_name:
        staged = Path(temp_name) / destination.name
        staged.mkdir()
        shutil.copytree(source / "data", staged / "data")
        _validate_staged(staged)
        os.rename(staged, destination)

    _validate_staged(destination)
    print(
        f"staged pinned probe: {destination} "
        f"({EXPECTED_DATA_FILES} files, {EXPECTED_DATA_BYTES} bytes)"
    )


if __name__ == "__main__":
    main()
