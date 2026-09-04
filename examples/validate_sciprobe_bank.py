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
"""Preflight for a SciProbe training bank, before any GPU is allocated.

Everything a bank run needs that is cheap to check: the rows and the verifier map
agree, nothing that reveals an answer reaches the policy, the hashes the verifier
will recompute match what shipped, and the configured checker interpreter can
actually execute a probe's checks.py.

The last one is the point. A checker that cannot import its dependencies used to
surface as reward 0, which GRPO reads as "the model failed" and which makes the
probe permanently unsolvable. Failing here costs seconds; failing on the cluster
costs a model load and a vLLM spin-up first.

This replaces the q3-specific signal validators for bank runs. Those are pinned to
one probe and to a gold-in-the-yaml verifier design that no longer exists.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

RUNNER = (
    Path(__file__).parent
    / "nemo_gym_extensions/resources_servers/sciprobe_checks/grader_runner.py"
)


def _fail(message: str) -> None:
    print(json.dumps({"status": "failed", "reason": message}))
    raise SystemExit(1)


def _load_rows(train_path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(train_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            _fail(f"{train_path}:{n} is not valid JSON: {error}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--train-path", type=Path, required=True)
    ap.add_argument(
        "--checker-python",
        required=True,
        help="interpreter the verifier will use to run checks.py",
    )
    ap.add_argument("--probe-container-root", default="/workspace/sciprobe-probe")
    ap.add_argument(
        "--execute-sample",
        type=int,
        default=2,
        help="how many probes to actually grade as a smoke test (0 disables)",
    )
    a = ap.parse_args()

    rows = _load_rows(a.train_path)
    if not rows:
        _fail(f"no rows in {a.train_path}")

    # 1. rows are well formed and self-consistent
    ids = []
    for row in rows:
        for key in ("id", "probe_id", "agent_ref", "responses_create_params", "verifier_type"):
            if key not in row:
                _fail(f"row {row.get('id', '?')} missing {key}")
        if row["id"] != row["probe_id"]:
            _fail(f"row id {row['id']} != probe_id {row['probe_id']}")
        if row["verifier_type"] != "sciprobe_checks":
            _fail(f"row {row['id']} has verifier_type {row['verifier_type']}")
        system = row["responses_create_params"]["input"][0]["content"]
        # Each rollout must be sent to its own probe directory. One shared chdir is
        # what makes every rollout see every probe's data once a bank has more than
        # one entry.
        want = f"{a.probe_container_root}/{row['probe_id']}"
        if want not in system:
            _fail(f"row {row['id']} does not chdir into its own probe dir ({want})")
        ids.append(row["probe_id"])
    if len(set(ids)) != len(ids):
        _fail("duplicate probe ids in the training rows")

    # 2. the policy must never see the grader or an answer
    policy = a.bank / "policy"
    grader = a.bank / "grader"
    for name in ("checks.py", "gold.json", "reference.py", "wrong_reference.py"):
        found = sorted(p for p in policy.rglob(name))
        if found:
            _fail(f"{name} is reachable from the policy tree: {found[0]}")
    for probe_id in ids:
        if not (policy / probe_id / "data").is_dir():
            _fail(f"policy data missing for {probe_id}")
        if not (grader / probe_id / "checks.py").is_file():
            _fail(f"grader checks.py missing for {probe_id}")

    # 3. hashes must match what the verifier will recompute, or every grade fails closed
    sys.path.insert(0, str(RUNNER.parent))
    import grader_runner as gr  # noqa: E402

    manifests = {
        m["probe_id"]: m
        for m in json.loads((a.bank / "BANK.json").read_text(encoding="utf-8"))["manifests"]
    }
    for probe_id in ids:
        m = manifests.get(probe_id)
        if m is None:
            _fail(f"{probe_id} is not in BANK.json")
        root = grader / probe_id
        if gr._sha256(root / "checks.py") != m["checks_sha256"]:
            _fail(f"{probe_id}: checks.py hash does not match the manifest")
        tree, files, total = gr._data_tree_sha256(root / "data")
        if (tree, files, total) != (m["data_tree_sha256"], m["data_files"], m["data_bytes"]):
            _fail(f"{probe_id}: data tree hash does not match the manifest")

    # 4. the configured interpreter can really run a checker
    executed = []
    for probe_id in ids[: max(a.execute_sample, 0)]:
        m = manifests[probe_id]
        proc = subprocess.run(
            [
                sys.executable, str(RUNNER),
                "--probe-root", str(grader / probe_id),
                "--checks-sha256", m["checks_sha256"],
                "--data-tree-sha256", m["data_tree_sha256"],
                "--checker-python", a.checker_python,
            ],
            input=json.dumps({k: None for k in m["answer_keys"]}),
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
            _fail(f"{probe_id}: checker could not run under {a.checker_python}: {tail[0][:300]}")
        out = json.loads(proc.stdout)
        # A deliberately empty answer must be rejected, not crash and not pass.
        if out.get("status") not in {"ok", "answer_rejected"}:
            _fail(f"{probe_id}: unexpected checker status {out.get('status')}")
        if out.get("status") == "ok" and all(v for _, v in out["checks"]):
            _fail(f"{probe_id}: an all-null answer passed every check")
        executed.append(probe_id)

    print(
        json.dumps(
            {
                "status": "ok",
                "rows": len(rows),
                "probes": len(set(ids)),
                "checker_python": a.checker_python,
                "executed_smoke": executed,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
