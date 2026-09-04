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
"""Write a run-local sciprobe_checks config carrying this bank's probes.

The committed component config declares one probe, because a config file is a poor
place to keep a few hundred generated entries and because the set changes per run.
The bank already emits its own `probes:` map, so a run splices that map into a copy
of the extensions tree and points NEMO_GYM_EXTRA_ROOTS at the copy. The committed
config stays a single readable example and each run gets exactly its own bank.

Only the entries for probes actually in this run's train.jsonl are kept, so the
verifier's allowlist is the training set rather than everything the bank happens to
hold. A probe absent from the allowlist is refused by the server, which is the
behaviour we want if a row and the map ever disagree.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

_ENTRY = re.compile(r'^  ("(?P<id>[^"]+)"):\s*$')


def parse_probe_blocks(text: str) -> dict[str, list[str]]:
    """Split the bank's probes yaml into one block of lines per probe id.

    Deliberately textual. The values include an OmegaConf ${oc.env:...} interpolation
    that a YAML round-trip would quote, and a quoted interpolation does not resolve.
    """
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.strip() in ("", "probes:"):
            continue
        m = _ENTRY.match(line)
        if m:
            current = m.group("id")
            blocks[current] = [line]
        elif current is not None:
            blocks[current].append(line)
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extensions", type=Path, required=True, help="source extensions tree")
    ap.add_argument("--bank-probes", type=Path, required=True, help="bank sciprobe_checks.probes.yaml")
    ap.add_argument("--train-path", type=Path, required=True, help="rows this run will train on")
    ap.add_argument("--output", type=Path, required=True, help="run-local extensions tree to create")
    a = ap.parse_args()

    wanted = []
    for line in a.train_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            wanted.append(json.loads(line)["probe_id"])
    if not wanted:
        raise SystemExit(f"no rows in {a.train_path}")

    blocks = parse_probe_blocks(a.bank_probes.read_text(encoding="utf-8"))
    missing = [p for p in wanted if p not in blocks]
    if missing:
        raise SystemExit(f"probes in train.jsonl but not in the bank map: {missing[:5]}")

    if a.output.exists():
        shutil.rmtree(a.output)
    shutil.copytree(a.extensions, a.output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    target = a.output / "resources_servers/sciprobe_checks/configs/sciprobe_checks.yaml"
    lines = target.read_text(encoding="utf-8").splitlines()

    # Replace the committed `probes:` mapping, re-indented from the bank's 2 spaces
    # to the 6 the component config nests it at.
    start = next(i for i, l in enumerate(lines) if l.strip() == "probes:")
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or len(lines[end]) - len(lines[end].lstrip()) > indent):
        end += 1

    shift = " " * (indent + 2 - 2)
    rendered = [lines[start]]
    for probe_id in wanted:
        for line in blocks[probe_id]:
            rendered.append(shift + line if line.strip() else line)

    target.write_text("\n".join(lines[:start] + rendered + lines[end:]) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "probes": len(wanted), "config": str(target)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
