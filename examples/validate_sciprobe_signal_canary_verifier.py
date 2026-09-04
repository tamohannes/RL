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

"""Exercise the strict output contract around pinned structured gold."""

from __future__ import annotations

import json
from typing import Any

from nemo_gym_extensions.resources_servers.sciprobe_checks.reward_contract import (
    parse_candidate,
    passes_contract,
    structured_gold_checks,
)

ANSWER_KEYS = [
    "n_samples",
    "total_aligned_reads",
    "total_modified_reads",
    "mean_modified_pct",
    "max_modified_pct",
    "max_modified_sample",
]
GOLD = {
    "n_samples": 4,
    "total_aligned_reads": 111766,
    "total_modified_reads": 69429,
    "mean_modified_pct": 62.19,
    "max_modified_pct": 67.75,
    "max_modified_sample": "S4",
}


def _evaluate(text: str) -> dict[str, Any]:
    parse_status, answer, exact_key_set = parse_candidate(text, ANSWER_KEYS)
    gold_checks = structured_gold_checks(answer, GOLD)
    return {
        "parse_status": parse_status,
        "exact_key_set": exact_key_set,
        "gold_checks_pass": bool(gold_checks)
        and all(passed for _, passed in gold_checks),
        "reward_pass": passes_contract(
            parse_status,
            exact_key_set,
            gold_checks,
        ),
    }


def main() -> None:
    cases = {
        "invalid_json": "this is not JSON",
        "json_string_wrapper": json.dumps(json.dumps(GOLD, sort_keys=True)),
        "missing_key": json.dumps(
            {key: value for key, value in GOLD.items() if key != "n_samples"}
        ),
        "extra_key": json.dumps({**GOLD, "extra": 1}),
        "wrong_value": json.dumps({**GOLD, "total_modified_reads": 42337}),
        "bool_as_count": json.dumps({**GOLD, "n_samples": True}),
        "exact_gold": json.dumps(GOLD, sort_keys=True),
    }
    outcomes = {name: _evaluate(text) for name, text in cases.items()}
    assert outcomes["invalid_json"]["parse_status"] == "invalid_json"
    assert outcomes["json_string_wrapper"]["parse_status"] == "not_object"
    assert outcomes["missing_key"]["exact_key_set"] is False
    assert outcomes["extra_key"]["exact_key_set"] is False
    assert outcomes["wrong_value"]["gold_checks_pass"] is False
    assert outcomes["bool_as_count"]["gold_checks_pass"] is False
    assert outcomes["exact_gold"] == {
        "parse_status": "ok",
        "exact_key_set": True,
        "gold_checks_pass": True,
        "reward_pass": True,
    }
    assert all(
        not outcome["reward_pass"]
        for name, outcome in outcomes.items()
        if name != "exact_gold"
    )
    print(json.dumps({"status": "ok", "cases": outcomes}, sort_keys=True))


if __name__ == "__main__":
    main()
