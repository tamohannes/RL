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

"""Validate the 32-rollout reward profile for the isolated overfit task."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from nemo_gym_extensions.resources_servers.sciprobe_overfit_checks.reward_contract import (
    score_stateful_choice_trace,
)

PROBE_ID = "stateful-choice-overfit-v1"
TOOL_NAME = "stateful_python_code_exec"
STATE_NAME = "carry"
STATE_VALUE = 17
EXPECTED_SECOND_OUTPUT = "55"
CHOICES = ["A", "B"]
REWARDED_CHOICE = "B"
EXPECTED_ROLLOUTS = 32


def _unwrap_singletons(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _numeric(value: Any, label: str) -> float:
    value = _unwrap_singletons(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(f"{label} is not numeric: {value!r}")
    number = float(value)
    assert math.isfinite(number), f"{label} is non-finite"
    return number


def _table_step(path: Path) -> int:
    marker = "full_result_"
    name = path.name.lower()
    marker_position = name.find(marker)
    assert marker_position >= 0, f"cannot find {marker!r} in {path.name!r}"
    suffix = name[marker_position + len(marker) :]
    step_text, separator, _ = suffix.partition("_")
    assert separator and step_text.isdigit(), f"cannot parse table step from {path}"
    return int(step_text)


def _is_split_table(path: Path, split: str) -> bool:
    lowered_parts = [part.lower() for part in path.parts]
    if split in lowered_parts:
        return True
    marker = f"{split}_"
    return any(part.startswith(marker) for part in lowered_parts)


def _load_validation_full_results(
    log_root: Path,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[Path]]]:
    results_by_step: dict[int, list[dict[str, Any]]] = {}
    paths_by_step: dict[int, list[Path]] = {}
    for path in sorted(log_root.rglob("*.table.json")):
        if "full_result_" not in path.name.lower():
            continue
        if not _is_split_table(path, "validation"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload.get("columns") == ["Full result"], (
            f"{path}: expected the one-column Full result schema"
        )
        rows = payload.get("data")
        assert isinstance(rows, list), f"{path}: table data is not a list"
        step = _table_step(path)
        parsed_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            assert isinstance(row, list) and len(row) == 1, (
                f"{path}: malformed row {row_index}"
            )
            value = row[0]
            if isinstance(value, str):
                value = json.loads(value)
            assert isinstance(value, dict), (
                f"{path}: row {row_index} Full result is not an object"
            )
            parsed_rows.append(value)
        results_by_step.setdefault(step, []).extend(parsed_rows)
        paths_by_step.setdefault(step, []).append(path)
    return results_by_step, paths_by_step


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        assert isinstance(row, dict), f"{path}:{line_number}: row is not an object"
        rows.append(row)
    return rows


def _validate_token_audit(result: dict[str, Any], label: str) -> dict[str, int]:
    audit = result.get("_nemo_rl_token_audit")
    assert isinstance(audit, dict), f"{label}: missing raw token audit"
    assert audit.get("version") == 1, f"{label}: unknown token-audit version"
    turns = audit.get("turns")
    assert isinstance(turns, list) and turns, f"{label}: token audit has no turns"

    seen: list[int] = []
    generation_tokens = 0
    output_item_indexes: list[int] = []
    for turn_index, turn in enumerate(turns):
        assert isinstance(turn, dict), f"{label}: audit turn {turn_index} is invalid"
        output_item_index = turn.get("output_item_index")
        assert isinstance(output_item_index, int) and not isinstance(
            output_item_index, bool
        )
        output_item_indexes.append(output_item_index)
        prompt = turn.get("prompt_token_ids")
        generation = turn.get("generation_token_ids")
        logprobs = turn.get("generation_logprobs")
        assert isinstance(prompt, list) and all(
            isinstance(token, int) and not isinstance(token, bool) for token in prompt
        ), f"{label}: invalid prompt IDs in audit turn {turn_index}"
        assert (
            isinstance(generation, list)
            and generation
            and all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in generation
            )
        ), f"{label}: invalid generation IDs in audit turn {turn_index}"
        assert isinstance(logprobs, list) and len(logprobs) == len(generation), (
            f"{label}: generation ID/logprob length mismatch in turn {turn_index}"
        )
        for value in logprobs:
            _numeric(value, f"{label}: audit turn {turn_index} logprob")
        assert prompt[: len(seen)] == seen, (
            f"{label}: audit turn {turn_index} breaks cumulative prompt continuity"
        )
        seen.extend(prompt[len(seen) :])
        seen.extend(generation)
        generation_tokens += len(generation)
    assert output_item_indexes == sorted(set(output_item_indexes)), (
        f"{label}: audit output-item indexes are not unique and ordered"
    )
    return {
        "turns": len(turns),
        "tokens": len(seen),
        "generation_tokens": generation_tokens,
    }


def _validate_result(result: dict[str, Any], label: str) -> dict[str, Any]:
    assert result.get("id") == PROBE_ID, f"{label}: wrong id"
    assert result.get("probe_id") == PROBE_ID, f"{label}: wrong probe_id"
    assert result.get("verifier_type") == "sciprobe_overfit_checks", (
        f"{label}: wrong verifier"
    )
    row_index = result.get("_rowidx")
    assert isinstance(row_index, int) and not isinstance(row_index, bool), (
        f"{label}: missing integer _rowidx"
    )
    response = result.get("response")
    assert isinstance(response, dict), f"{label}: response is missing"
    assert response.get("status") == "completed", f"{label}: response did not complete"
    assert response.get("error") is None, f"{label}: response contains an error"
    output = response.get("output")
    assert isinstance(output, list), f"{label}: response.output is not a list"

    local_reward, choice, checks = score_stateful_choice_trace(
        response,
        tool_name=TOOL_NAME,
        state_name=STATE_NAME,
        state_value=STATE_VALUE,
        expected_second_output=EXPECTED_SECOND_OUTPUT,
        choices=CHOICES,
        rewarded_choice=REWARDED_CHOICE,
    )
    assert checks and all(bool(passed) for _, passed in checks), (
        f"{label}: invalid fixed stateful trace: "
        f"{[name for name, passed in checks if not passed]}"
    )
    assert choice in CHOICES, f"{label}: invalid final choice {choice!r}"
    reward = _numeric(result.get("reward"), f"{label}: reward")
    assert reward in {0.0, 1.0}, f"{label}: reward is not binary"
    assert reward == local_reward, (
        f"{label}: verifier reward differs from local contract"
    )
    expected_reward = 1.0 if choice == REWARDED_CHOICE else 0.0
    assert reward == expected_reward, f"{label}: reward does not select B"
    audit_summary = _validate_token_audit(result, label)
    return {
        "row_index": row_index,
        "choice": choice,
        "reward": reward,
        "audit": audit_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    log_root = run_root / "logs"
    assert log_root.is_dir(), f"missing log root: {log_root}"
    val_paths = sorted(log_root.rglob("val_data_step0.jsonl"))
    assert len(val_paths) == 1, (
        f"expected one val_data_step0.jsonl, found {len(val_paths)}"
    )
    val_rows = _load_jsonl(val_paths[0])
    assert len(val_rows) == EXPECTED_ROLLOUTS, (
        f"expected {EXPECTED_ROLLOUTS} validation rows, found {len(val_rows)}"
    )

    results_by_step, paths_by_step = _load_validation_full_results(log_root)
    assert set(results_by_step) == {0}, (
        f"profile run must contain only validation full-result step 0; "
        f"found {sorted(results_by_step)}"
    )
    assert len(paths_by_step[0]) == 1, (
        f"expected one validation full-result table, found {len(paths_by_step[0])}"
    )
    results = results_by_step[0]
    assert len(results) == EXPECTED_ROLLOUTS, (
        f"expected {EXPECTED_ROLLOUTS} validation full results, found {len(results)}"
    )
    summaries = [
        _validate_result(result, f"validation step 0 rollout {index}")
        for index, result in enumerate(results)
    ]
    by_index = {summary["row_index"]: summary for summary in summaries}
    assert len(by_index) == EXPECTED_ROLLOUTS, "duplicate validation _rowidx values"
    assert set(by_index) == set(range(EXPECTED_ROLLOUTS)), (
        f"validation _rowidx values are {sorted(by_index)}"
    )

    logged_rewards: dict[int, float] = {}
    for position, row in enumerate(val_rows):
        row_index = row.get("idx")
        assert isinstance(row_index, int) and not isinstance(row_index, bool), (
            f"validation JSONL row {position}: invalid idx"
        )
        reward = _numeric(
            row.get("rewards"), f"validation JSONL row {position}: reward"
        )
        assert reward in {0.0, 1.0}
        assert reward == by_index[row_index]["reward"], (
            f"validation JSONL row {position}: reward differs from full result"
        )
        logged_rewards[row_index] = reward
    assert set(logged_rewards) == set(range(EXPECTED_ROLLOUTS))

    choices = [by_index[index]["choice"] for index in sorted(by_index)]
    rewards = [logged_rewards[index] for index in sorted(logged_rewards)]
    assert set(choices) == set(CHOICES), (
        f"reward profile lacks both valid choices: {choices}"
    )
    assert set(rewards) == {0.0, 1.0}, (
        f"reward profile lacks both binary classes: {rewards}"
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "run_root": str(run_root),
                "rollouts": EXPECTED_ROLLOUTS,
                "validation_jsonl": str(val_paths[0]),
                "full_result_table": str(paths_by_step[0][0]),
                "choice_counts": {choice: choices.count(choice) for choice in CHOICES},
                "reward_mean": sum(rewards) / len(rewards),
                "token_audit_turns": [
                    summary["audit"]["turns"] for summary in summaries
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
