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

"""Prove the signal canary uses only its pinned structured gold."""

from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

from nemo_gym_extensions.resources_servers.sciprobe_checks.reward_contract import (
    canonical_answer_sha256,
    structured_gold_checks,
)

GOLD_SHA256 = "45f51cc52d4093ee60d941fc093653b0497f9b076b0de5b6b8175a0f945df36c"
GOLD = {
    "n_samples": 4,
    "total_aligned_reads": 111766,
    "total_modified_reads": 69429,
    "mean_modified_pct": 62.19,
    "max_modified_pct": 67.75,
    "max_modified_sample": "S4",
}


def _assert_no_dynamic_checker_path(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_imports = {"importlib", "pickle", "subprocess"}
    forbidden_calls = {"__import__", "eval", "exec"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert (
                not {alias.name.partition(".")[0] for alias in node.names}
                & forbidden_imports
            )
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").partition(".")[0] not in forbidden_imports
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_calls
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "checks.py" not in node.value


def main() -> None:
    assert canonical_answer_sha256(GOLD) == GOLD_SHA256
    result = structured_gold_checks(GOLD, GOLD)
    assert result and all(passed for _, passed in result)
    assert not all(
        passed
        for _, passed in structured_gold_checks(
            {**GOLD, "total_modified_reads": 42337}, GOLD
        )
    )
    assert not all(
        passed
        for _, passed in structured_gold_checks({**GOLD, "n_samples": True}, GOLD)
    )

    root = Path(__file__).resolve().parent
    verifier_files = (
        root / "nemo_gym_extensions/resources_servers/sciprobe_checks/app.py",
        root
        / "nemo_gym_extensions/resources_servers/sciprobe_checks/reward_contract.py",
    )
    for verifier_file in verifier_files:
        _assert_no_dynamic_checker_path(verifier_file)

    with tempfile.TemporaryDirectory(prefix="sciprobe_no_checker_") as directory:
        marker = Path(directory) / "dynamic-checker-executed"
        neighbor = Path(directory) / "checks.py"
        neighbor.write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\nraise SystemExit(91)\n",
            encoding="utf-8",
        )
        assert all(passed for _, passed in structured_gold_checks(GOLD, GOLD))
        assert not marker.exists()

    print(
        json.dumps(
            {
                "status": "ok",
                "gold_sha256": GOLD_SHA256,
                "typed_gold_checks": len(result),
                "dynamic_checker_code_absent": True,
                "malicious_neighbor_not_executed": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
