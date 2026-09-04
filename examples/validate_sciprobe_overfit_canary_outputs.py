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

"""Validate exact tokens and learning in the isolated Lightning overfit canary."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any

from nemo_gym_extensions.resources_servers.sciprobe_overfit_checks.reward_contract import (
    score_stateful_choice_trace,
)
from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

PROBE_ID = "stateful-choice-overfit-v1"
TOOL_NAME = "stateful_python_code_exec"
STATE_NAME = "carry"
STATE_VALUE = 17
EXPECTED_SECOND_OUTPUT = "55"
CHOICES = ["A", "B"]
REWARDED_CHOICE = "B"
REQUIRED_TENSORBOARD_TAGS = {
    "train/loss",
    "train/grad_norm",
    "train/global_valid_toks",
    "train/lr",
    "train/advantages/max",
    "train/advantages/min",
    "train/token_mult_prob_error",
}


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


def _float32(value: float) -> float:
    """Match the default float32 tensor conversion in Gym postprocessing."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _integer(value: Any, label: str) -> int:
    value = _unwrap_singletons(value)
    assert isinstance(value, int) and not isinstance(value, bool), (
        f"{label} is not an integer: {value!r}"
    )
    return value


def _one_sequence(row: dict[str, Any], field: str, label: str) -> list[Any]:
    value = row.get(field)
    assert isinstance(value, list) and len(value) == 1, (
        f"{label}: {field} must have one batch dimension"
    )
    sequence = value[0]
    assert isinstance(sequence, list), f"{label}: {field} is not a sequence"
    return sequence


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


def _indexed_jsonl_paths(log_root: Path, prefix: str) -> dict[int, Path]:
    suffix = ".jsonl"
    paths: dict[int, Path] = {}
    for path in sorted(log_root.rglob(f"{prefix}*{suffix}")):
        name = path.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        step_text = name[len(prefix) : -len(suffix)]
        if not step_text.isdigit():
            continue
        step = int(step_text)
        assert step not in paths, f"duplicate {prefix}{step}{suffix} under {log_root}"
        paths[step] = path
    return paths


def _table_step(path: Path) -> int:
    marker = "full_result_"
    name = path.name.lower()
    marker_position = name.find(marker)
    assert marker_position >= 0, f"cannot find {marker!r} in {path.name!r}"
    suffix = name[marker_position + len(marker) :]
    step_text, separator, _ = suffix.partition("_")
    assert separator and step_text.isdigit(), f"cannot parse table step from {path}"
    return int(step_text)


def _table_split(path: Path) -> str:
    lowered_parts = [part.lower() for part in path.parts]
    matches: list[str] = []
    for split in ("train", "validation"):
        marker = f"{split}_"
        if split in lowered_parts or any(
            part.startswith(marker) for part in lowered_parts
        ):
            matches.append(split)
    assert len(matches) == 1, f"cannot uniquely classify table split for {path}"
    return matches[0]


def _load_full_result_tables(
    log_root: Path,
) -> tuple[
    dict[str, dict[int, list[list[dict[str, Any]]]]],
    dict[str, dict[int, list[Path]]],
]:
    results: dict[str, dict[int, list[list[dict[str, Any]]]]] = {
        "train": {},
        "validation": {},
    }
    sources: dict[str, dict[int, list[Path]]] = {
        "train": {},
        "validation": {},
    }
    for path in sorted(log_root.rglob("*.table.json")):
        if "full_result_" not in path.name.lower():
            continue
        split = _table_split(path)
        step = _table_step(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload.get("columns") == ["Full result"], (
            f"{path}: expected the one-column Full result schema"
        )
        rows = payload.get("data")
        assert isinstance(rows, list), f"{path}: table data is not a list"
        parsed_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            assert isinstance(row, list) and len(row) == 1, (
                f"{path}: malformed row {row_index}"
            )
            value = row[0]
            if isinstance(value, str):
                value = json.loads(value)
            assert isinstance(value, dict), (
                f"{path}: Full result row {row_index} is not an object"
            )
            parsed_rows.append(value)
        results[split].setdefault(step, []).append(parsed_rows)
        sources[split].setdefault(step, []).append(path)
    return results, sources


def _raw_audit_segments(
    result: dict[str, Any], label: str
) -> tuple[list[int], list[int], list[float], dict[str, int]]:
    audit = result.get("_nemo_rl_token_audit")
    assert isinstance(audit, dict), f"{label}: missing raw token audit"
    assert audit.get("version") == 1, f"{label}: unknown token-audit version"
    turns = audit.get("turns")
    assert isinstance(turns, list) and turns, f"{label}: token audit has no turns"

    seen: list[int] = []
    expected_mask: list[int] = []
    expected_logprobs: list[float] = []
    generation_tokens = 0
    output_item_indexes: list[int] = []
    for turn_index, turn in enumerate(turns):
        assert isinstance(turn, dict), f"{label}: audit turn {turn_index} is invalid"
        output_item_index = turn.get("output_item_index")
        assert isinstance(output_item_index, int) and not isinstance(
            output_item_index, bool
        ), f"{label}: audit turn {turn_index} has invalid output_item_index"
        output_item_indexes.append(output_item_index)
        prompt = turn.get("prompt_token_ids")
        generation = turn.get("generation_token_ids")
        logprobs = turn.get("generation_logprobs")
        assert isinstance(prompt, list) and all(
            isinstance(token, int) and not isinstance(token, bool) for token in prompt
        ), f"{label}: audit turn {turn_index} has invalid prompt IDs"
        assert (
            isinstance(generation, list)
            and generation
            and all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in generation
            )
        ), f"{label}: audit turn {turn_index} has invalid generation IDs"
        assert isinstance(logprobs, list) and len(logprobs) == len(generation), (
            f"{label}: audit turn {turn_index} generation IDs/logprobs differ"
        )
        typed_logprobs = [
            _float32(_numeric(value, f"{label}: audit turn {turn_index} logprob"))
            for value in logprobs
        ]
        assert prompt[: len(seen)] == seen, (
            f"{label}: audit turn {turn_index} breaks cumulative prompt continuity"
        )
        suffix = prompt[len(seen) :]
        seen.extend(suffix)
        expected_mask.extend([0] * len(suffix))
        expected_logprobs.extend([0.0] * len(suffix))
        seen.extend(generation)
        expected_mask.extend([1] * len(generation))
        expected_logprobs.extend(typed_logprobs)
        generation_tokens += len(generation)
    assert output_item_indexes == sorted(set(output_item_indexes)), (
        f"{label}: audit output-item indexes are not unique and ordered"
    )
    return (
        seen,
        expected_mask,
        expected_logprobs,
        {
            "turns": len(turns),
            "tokens": len(seen),
            "generation_tokens": generation_tokens,
        },
    )


def _validate_full_result(result: dict[str, Any], label: str) -> dict[str, Any]:
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
    response_status = response.get("status")
    assert response_status in {"completed", "incomplete"}, (
        f"{label}: unexpected response status {response_status!r}"
    )
    assert response.get("error") is None, f"{label}: response contains an error"
    incomplete_details = response.get("incomplete_details")
    if response_status == "completed":
        assert incomplete_details is None, (
            f"{label}: completed response has incomplete_details"
        )
    else:
        assert incomplete_details == {"reason": "max_output_tokens"}, (
            f"{label}: incomplete response was not an explicit output-token cap"
        )
    assert isinstance(response.get("output"), list), (
        f"{label}: response.output is not a list"
    )
    num_tool_calls = _integer(result.get("num_tool_calls"), f"{label}: tool calls")
    tool_timeout_count = _integer(
        result.get("tool_timeout_count"), f"{label}: tool timeout count"
    )
    tool_request_timeout_count = _integer(
        result.get("tool_request_timeout_count"),
        f"{label}: tool request timeout count",
    )

    local_reward, choice, checks = score_stateful_choice_trace(
        response,
        tool_name=TOOL_NAME,
        state_name=STATE_NAME,
        state_value=STATE_VALUE,
        expected_second_output=EXPECTED_SECOND_OUTPUT,
        choices=CHOICES,
        rewarded_choice=REWARDED_CHOICE,
    )
    assert checks, f"{label}: verifier returned no checks"
    failed_checks = [name for name, passed in checks if not passed]
    trace_valid = not failed_checks
    reward = _numeric(result.get("reward"), f"{label}: reward")
    assert reward in {0.0, 1.0}, f"{label}: reward is not binary"
    assert reward == local_reward, (
        f"{label}: verifier reward differs from local contract"
    )
    if response_status == "incomplete":
        assert reward == 0.0 and not trace_valid, (
            f"{label}: capped response must be an unrewarded invalid trace"
        )
    if reward == 1.0:
        assert trace_valid and choice == REWARDED_CHOICE, (
            f"{label}: rewarded rollout is not a valid B trace"
        )
    elif trace_valid:
        assert choice == "A", f"{label}: valid unrewarded rollout is not A"
    tokens, mask, logprobs, audit_summary = _raw_audit_segments(result, label)
    if response_status == "incomplete":
        responses_create_params = result.get("responses_create_params")
        assert isinstance(responses_create_params, dict), (
            f"{label}: missing responses_create_params"
        )
        max_output_tokens = responses_create_params.get("max_output_tokens")
        assert isinstance(max_output_tokens, int) and max_output_tokens > 0, (
            f"{label}: capped response has no positive max_output_tokens"
        )
        audit_turns = result["_nemo_rl_token_audit"]["turns"]
        assert len(audit_turns[-1]["generation_token_ids"]) == max_output_tokens, (
            f"{label}: incomplete response did not exactly hit max_output_tokens"
        )
    return {
        "row_index": row_index,
        "response_status": response_status,
        "choice": choice,
        "reward": reward,
        "trace_valid": trace_valid,
        "failed_checks": failed_checks,
        "num_tool_calls": num_tool_calls,
        "tool_timeout_count": tool_timeout_count,
        "tool_request_timeout_count": tool_request_timeout_count,
        "tokens": tokens,
        "mask": mask,
        "logprobs": logprobs,
        "audit": audit_summary,
    }


def _validate_train_step(
    *,
    step: int,
    results: list[dict[str, Any]],
    train_path: Path,
    expected_rollouts: int,
    pad_token_id: int,
    sequence_multiple: int,
    max_sequence_length: int,
    probability_error_threshold: float,
    require_valid_choice_contrast: bool,
) -> dict[str, Any]:
    assert len(results) == expected_rollouts, (
        f"train step {step}: expected {expected_rollouts} full results, "
        f"found {len(results)}"
    )
    summaries = [
        _validate_full_result(result, f"train step {step} rollout {position}")
        for position, result in enumerate(results)
    ]
    result_by_index = {summary["row_index"]: summary for summary in summaries}
    assert len(result_by_index) == expected_rollouts, (
        f"train step {step}: duplicate full-result _rowidx"
    )
    assert set(result_by_index) == set(range(expected_rollouts)), (
        f"train step {step}: full-result indexes are {sorted(result_by_index)}"
    )

    rows = _load_jsonl(train_path)
    assert len(rows) == expected_rollouts, (
        f"train step {step}: expected {expected_rollouts} JSONL rows, found {len(rows)}"
    )
    row_indexes: set[int] = set()
    unpadded_lengths: list[int] = []
    serialized_lengths: list[int] = []
    active_advantages: list[float] = []
    active_probability_errors: list[float] = []
    any_policy_delta = False
    rewards: list[float] = []
    choices: list[str | None] = []
    for position, row in enumerate(rows):
        label = f"train step {step} JSONL row {position}"
        row_index = _integer(row.get("idx"), f"{label}: idx")
        assert row_index in result_by_index, f"{label}: no matching full result"
        assert row_index not in row_indexes, f"{label}: duplicate idx"
        row_indexes.add(row_index)
        summary = result_by_index[row_index]

        reward_field = "filtered_rewards" if "filtered_rewards" in row else "rewards"
        reward = _numeric(row.get(reward_field), f"{label}: {reward_field}")
        assert reward == summary["reward"], f"{label}: reward/full-result mismatch"
        rewards.append(reward)
        choices.append(summary["choice"])
        sample_mask = _numeric(row.get("sample_loss_mask"), f"{label}: sample mask")
        assert sample_mask > 0, f"{label}: sample is masked"
        input_length = _integer(row.get("input_lengths"), f"{label}: input length")

        token_ids = _one_sequence(row, "token_ids", label)
        token_mask = _one_sequence(row, "token_loss_mask", label)
        advantages = _one_sequence(row, "advantages", label)
        generation_logprobs = _one_sequence(row, "generation_logprobs", label)
        prev_logprobs = _one_sequence(row, "prev_logprobs", label)
        serialized_length = len(token_ids)
        assert serialized_length > 0 and serialized_length <= max_sequence_length
        assert serialized_length % sequence_multiple == 0, (
            f"{label}: serialized length {serialized_length} is not divisible by "
            f"{sequence_multiple}"
        )
        assert all(
            len(sequence) == serialized_length
            for sequence in (
                token_mask,
                advantages,
                generation_logprobs,
                prev_logprobs,
            )
        ), f"{label}: serialized field lengths differ"
        expected_tokens = summary["tokens"]
        expected_mask = summary["mask"]
        expected_logprobs = summary["logprobs"]
        assert input_length == len(expected_tokens), (
            f"{label}: input_lengths does not equal the unpadded audit length"
        )
        assert token_ids[:input_length] == expected_tokens, (
            f"{label}: token_ids differ from the exact raw token audit"
        )
        assert token_mask[:input_length] == expected_mask, (
            f"{label}: token_loss_mask differs from audit turn boundaries"
        )
        assert generation_logprobs[:input_length] == expected_logprobs, (
            f"{label}: generation_logprobs differ from the exact raw audit"
        )
        assert all(token == pad_token_id for token in token_ids[input_length:]), (
            f"{label}: token padding does not use tokenizer pad ID {pad_token_id}"
        )
        assert all(float(value) == 0.0 for value in token_mask[input_length:]), (
            f"{label}: padded token mask is nonzero"
        )
        assert all(
            float(value) == 0.0 for value in generation_logprobs[input_length:]
        ), f"{label}: padded generation logprobs are nonzero"

        for token_position, enabled in enumerate(token_mask[:input_length]):
            assert enabled in {0, 1, 0.0, 1.0, False, True}, (
                f"{label}: token mask is not binary at {token_position}"
            )
            if not bool(enabled):
                continue
            advantage = _numeric(
                advantages[token_position], f"{label}: advantage {token_position}"
            )
            generation_logprob = _numeric(
                generation_logprobs[token_position],
                f"{label}: generation logprob {token_position}",
            )
            prev_logprob = _numeric(
                prev_logprobs[token_position],
                f"{label}: previous logprob {token_position}",
            )
            active_advantages.append(advantage)
            delta = abs(prev_logprob - generation_logprob)
            any_policy_delta = any_policy_delta or delta != 0.0
            active_probability_errors.append(math.exp(delta))
        unpadded_lengths.append(input_length)
        serialized_lengths.append(serialized_length)

    assert row_indexes == set(range(expected_rollouts))
    assert set(rewards) == {0.0, 1.0}, (
        f"train step {step}: same-prompt group lacks mixed rewards"
    )
    valid_choice_counts = {
        choice: sum(
            summary["trace_valid"] and summary["choice"] == choice
            for summary in summaries
        )
        for choice in CHOICES
    }
    if require_valid_choice_contrast:
        assert all(valid_choice_counts[choice] > 0 for choice in CHOICES), (
            f"train step {step}: initial same-prompt group lacks a valid A/B "
            f"contrast: {valid_choice_counts}"
        )
    assert any(value > 0 for value in active_advantages), (
        f"train step {step}: no positive advantage"
    )
    assert any(value < 0 for value in active_advantages), (
        f"train step {step}: no negative advantage"
    )
    assert any_policy_delta, (
        f"train step {step}: prev_logprobs were not independently recomputed"
    )
    expected_serialized_length = (
        (max(unpadded_lengths) + sequence_multiple - 1) // sequence_multiple
    ) * sequence_multiple
    assert set(serialized_lengths) == {expected_serialized_length}, (
        f"train step {step}: serialized lengths {sorted(set(serialized_lengths))} "
        f"do not equal roundup(max unpadded length)={expected_serialized_length}"
    )
    token_mult_probability_error = sum(active_probability_errors) / len(
        active_probability_errors
    )
    assert token_mult_probability_error <= probability_error_threshold, (
        f"train step {step}: reconstructed token multiplicative probability error "
        f"{token_mult_probability_error} exceeds {probability_error_threshold}"
    )
    return {
        "jsonl": str(train_path),
        "reward_mean": sum(rewards) / len(rewards),
        "choice_counts": {choice: choices.count(choice) for choice in CHOICES},
        "invalid_choice_count": choices.count(None),
        "valid_choice_counts": valid_choice_counts,
        "invalid_trace_count": sum(not summary["trace_valid"] for summary in summaries),
        "response_status_counts": {
            status: sum(summary["response_status"] == status for summary in summaries)
            for status in ("completed", "incomplete")
        },
        "tool_timeout_count": sum(
            summary["tool_timeout_count"] for summary in summaries
        ),
        "tool_request_timeout_count": sum(
            summary["tool_request_timeout_count"] for summary in summaries
        ),
        "unpadded_length_min": min(unpadded_lengths),
        "unpadded_length_max": max(unpadded_lengths),
        "serialized_length": expected_serialized_length,
        "active_tokens": len(active_advantages),
        "advantage_min": min(active_advantages),
        "advantage_max": max(active_advantages),
        "token_mult_prob_error_reconstructed": token_mult_probability_error,
    }


def _validate_validation_step(
    *,
    step: int,
    results: list[dict[str, Any]],
    validation_path: Path,
    expected_rollouts: int,
) -> dict[str, Any]:
    assert len(results) == expected_rollouts, (
        f"validation step {step}: expected {expected_rollouts} full results, "
        f"found {len(results)}"
    )
    summaries = [
        _validate_full_result(result, f"validation step {step} rollout {position}")
        for position, result in enumerate(results)
    ]
    result_by_index = {summary["row_index"]: summary for summary in summaries}
    assert len(result_by_index) == expected_rollouts, (
        f"validation step {step}: duplicate full-result _rowidx"
    )
    assert set(result_by_index) == set(range(expected_rollouts)), (
        f"validation step {step}: full-result indexes are {sorted(result_by_index)}"
    )
    rows = _load_jsonl(validation_path)
    assert len(rows) == expected_rollouts, (
        f"validation step {step}: expected {expected_rollouts} JSONL rows, "
        f"found {len(rows)}"
    )
    rewards_by_index: dict[int, float] = {}
    for position, row in enumerate(rows):
        row_index = _integer(
            row.get("idx"), f"validation step {step} JSONL row {position}: idx"
        )
        assert row_index in result_by_index
        assert row_index not in rewards_by_index
        reward = _numeric(
            row.get("rewards"),
            f"validation step {step} JSONL row {position}: reward",
        )
        assert reward in {0.0, 1.0}
        assert reward == result_by_index[row_index]["reward"], (
            f"validation step {step} JSONL row {position}: reward mismatch"
        )
        rewards_by_index[row_index] = reward
    assert set(rewards_by_index) == set(range(expected_rollouts))
    choices = [result_by_index[index]["choice"] for index in sorted(result_by_index)]
    rewards = [rewards_by_index[index] for index in sorted(rewards_by_index)]
    return {
        "jsonl": str(validation_path),
        "reward_mean": sum(rewards) / len(rewards),
        "choice_counts": {choice: choices.count(choice) for choice in CHOICES},
        "invalid_choice_count": choices.count(None),
        "valid_trace_count": sum(summary["trace_valid"] for summary in summaries),
        "invalid_trace_count": sum(not summary["trace_valid"] for summary in summaries),
        "response_status_counts": {
            status: sum(summary["response_status"] == status for summary in summaries)
            for status in ("completed", "incomplete")
        },
        "tool_timeout_count": sum(
            summary["tool_timeout_count"] for summary in summaries
        ),
        "tool_request_timeout_count": sum(
            summary["tool_request_timeout_count"] for summary in summaries
        ),
    }


def _select_accepted_train_table(
    *,
    step: int,
    candidates: list[list[dict[str, Any]]],
    candidate_paths: list[Path],
    train_path: Path,
    expected_rollouts: int,
    pad_token_id: int,
    sequence_multiple: int,
    max_sequence_length: int,
    probability_error_threshold: float,
) -> dict[str, Any]:
    """Find the one dynamic-sampling table that exactly produced train data."""
    assert len(candidates) == len(candidate_paths) and candidates
    matches: list[tuple[Path, dict[str, Any]]] = []
    rejected: dict[str, str] = {}
    for candidate, candidate_path in zip(candidates, candidate_paths):
        try:
            summary = _validate_train_step(
                step=step,
                results=candidate,
                train_path=train_path,
                expected_rollouts=expected_rollouts,
                pad_token_id=pad_token_id,
                sequence_multiple=sequence_multiple,
                max_sequence_length=max_sequence_length,
                probability_error_threshold=probability_error_threshold,
                require_valid_choice_contrast=step == 1,
            )
        except AssertionError as error:
            rejected[str(candidate_path)] = str(error)
        else:
            matches.append((candidate_path, summary))
    assert len(matches) == 1, (
        f"train step {step}: expected exactly one full-result table to match "
        f"train_data, found {len(matches)}; rejected={json.dumps(rejected, sort_keys=True)}"
    )
    accepted_path, summary = matches[0]
    summary["accepted_full_result_table"] = str(accepted_path)
    summary["dynamic_sampling_table_candidates"] = len(candidates)
    return summary


def _load_tensorboard_metrics(
    log_root: Path, expected_steps: set[int]
) -> dict[str, Any]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    errors: dict[str, str] = {}
    for path in sorted(log_root.rglob("events.out.tfevents.*")):
        try:
            accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
            accumulator.Reload()
            tags = set(accumulator.Tags().get("scalars", []))
            if not REQUIRED_TENSORBOARD_TAGS.issubset(tags):
                continue
            metrics: dict[str, dict[int, float]] = {}
            complete = True
            for tag in sorted(REQUIRED_TENSORBOARD_TAGS):
                values: dict[int, float] = {}
                for event in accumulator.Scalars(tag):
                    value = float(event.value)
                    assert math.isfinite(value), f"{path}: {tag} is non-finite"
                    values[int(event.step)] = value
                if set(values) != expected_steps:
                    complete = False
                    break
                metrics[tag] = values
            if not complete:
                continue
            return {
                "path": str(path),
                "metrics": metrics,
                "available_scalar_tags": sorted(tags),
            }
        except Exception as error:
            errors[str(path)] = f"{type(error).__name__}: {error}"
    raise AssertionError(
        "no TensorBoard event file contains the complete six-step proof: "
        + json.dumps(errors, sort_keys=True)
    )


def _validate_safetensors(path: Path) -> dict[str, Any]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_header_length = handle.read(8)
        assert len(raw_header_length) == 8, f"truncated safetensors: {path}"
        (header_length,) = struct.unpack("<Q", raw_header_length)
        assert 2 < header_length <= file_size - 8, (
            f"invalid safetensors header length: {path}"
        )
        header = json.loads(handle.read(header_length))
    assert isinstance(header, dict), f"safetensors header is not an object: {path}"
    payload_size = file_size - 8 - header_length
    tensor_count = 0
    maximum_end = 0
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        assert isinstance(metadata, dict), f"invalid tensor metadata {name}: {path}"
        offsets = metadata.get("data_offsets")
        assert isinstance(offsets, list) and len(offsets) == 2
        assert all(isinstance(value, int) for value in offsets)
        start, end = offsets
        assert 0 <= start <= end <= payload_size
        assert isinstance(metadata.get("shape"), list)
        assert isinstance(metadata.get("dtype"), str)
        maximum_end = max(maximum_end, end)
        tensor_count += 1
    assert tensor_count > 0, f"no tensors in {path}"
    assert maximum_end == payload_size, f"incomplete safetensors payload: {path}"
    return {"path": str(path), "bytes": file_size, "tensors": tensor_count}


def _validate_checkpoint(run_root: Path, expected_step: int) -> dict[str, Any]:
    checkpoint_root = run_root / "checkpoints"
    status_path = checkpoint_root / "latest_checkpoint_status.json"
    assert status_path.is_file(), f"missing checkpoint status: {status_path}"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert isinstance(status, dict)
    assert status.get("last_checkpoint_step") == expected_step, (
        f"final checkpoint step is {status.get('last_checkpoint_step')!r}, "
        f"expected {expected_step}"
    )
    assert (
        _numeric(
            status.get("last_successful_ckpt_save_completion"),
            "checkpoint completion time",
        )
        > 0
    )
    step_root = checkpoint_root / f"step_{expected_step}"
    assert step_root.is_dir(), f"missing final checkpoint: {step_root}"
    tensor_paths = sorted(step_root.rglob("*.safetensors"))
    assert tensor_paths, f"no safetensors under {step_root}"
    return {
        "status_path": str(status_path),
        "step": expected_step,
        "safetensors": [_validate_safetensors(path) for path in tensor_paths],
    }


def _load_master_config(path: Path) -> MasterConfig:
    resolved = OmegaConf.to_container(load_config(str(path)), resolve=True)
    assert isinstance(resolved, dict), f"{path}: resolved config is not a mapping"
    return MasterConfig(**resolved)


def main() -> None:
    register_omegaconf_resolvers()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-max-steps", type=int, default=6)
    parser.add_argument("--expected-train-rollouts", type=int, default=16)
    parser.add_argument("--expected-validation-rollouts", type=int, default=32)
    args = parser.parse_args()

    assert args.expected_max_steps == 6, (
        "the overfit proof requires exactly six updates"
    )
    expected_train_steps = set(range(1, args.expected_max_steps + 1))
    expected_validation_steps = set(range(0, args.expected_max_steps + 1))
    run_root = args.run_root.resolve()
    log_root = run_root / "logs"
    assert log_root.is_dir(), f"missing log root: {log_root}"

    master_config = _load_master_config(args.config.resolve())
    assert master_config.grpo.max_num_steps == args.expected_max_steps
    assert master_config.grpo.num_generations_per_prompt == args.expected_train_rollouts
    assert (
        master_config.grpo.val_num_generations_per_prompt
        == args.expected_validation_rollouts
    )
    assert master_config.grpo.num_prompts_per_step == 1
    assert master_config.grpo.use_dynamic_sampling is True
    assert master_config.grpo.async_grpo is not None
    assert master_config.grpo.async_grpo.enabled is False
    train_repeat = int(master_config.data["train"]["repeat"])
    assert train_repeat >= master_config.grpo.max_num_steps * (
        master_config.grpo.dynamic_sampling_max_gen_batches + 1
    )
    # The Gym runner accepts only null here and derives both runtime values
    # from the one-row validation dataset before entering grpo_train().
    assert master_config.grpo.max_val_samples is None
    assert master_config.grpo.val_batch_size is None
    assert master_config.grpo.val_at_start is True
    assert master_config.grpo.val_at_end is True
    assert master_config.grpo.val_period == 1
    assert master_config.grpo.stop_at_validation_metric is None
    assert master_config.grpo.stop_at_validation_threshold is None
    assert master_config.grpo.seq_logprob_error_threshold == 2
    assert master_config.loss_fn.force_on_policy_ratio is False
    assert master_config.loss_fn.use_importance_sampling_correction is False

    assert master_config.data["default"]["processor"] == "nemo_gym_data_processor"
    assert master_config.data["default"]["env_name"] == "nemo_gym"
    assert master_config.data["use_multiple_dataloader"] is False
    assert master_config.data["num_workers"] == 1
    assert master_config.policy["sequence_packing"]["enabled"] is False
    assert master_config.policy["dynamic_batching"]["enabled"] is False
    generation = master_config.policy["generation"]
    assert generation["top_p"] == 1.0
    assert generation["top_k"] is None
    assert generation["val_top_p"] == 1.0
    assert generation["val_top_k"] is None
    assert master_config.checkpointing["metric_name"] is None
    assert master_config.checkpointing["save_optimizer"] is False
    assert master_config.checkpointing["model_save_format"] == "safetensors"

    sequence_multiple = int(master_config.policy["make_sequence_length_divisible_by"])
    assert sequence_multiple > 0
    max_sequence_length = int(master_config.policy["max_total_sequence_length"])
    probability_error_threshold = master_config.grpo.seq_logprob_error_threshold
    assert isinstance(probability_error_threshold, (int, float)) and not isinstance(
        probability_error_threshold, bool
    ), "grpo.seq_logprob_error_threshold must be configured"
    probability_error_threshold = float(probability_error_threshold)
    assert probability_error_threshold >= 1.0

    tokenizer_config = dict(master_config.policy["tokenizer"])
    tokenizer_config["name"] = str(Path(args.model).resolve())
    tokenizer = get_tokenizer(tokenizer_config)
    pad_token_id = tokenizer.pad_token_id
    assert isinstance(pad_token_id, int) and not isinstance(pad_token_id, bool), (
        "Lightning tokenizer has no integer pad token ID"
    )

    full_results, table_paths = _load_full_result_tables(log_root)
    assert set(full_results["train"]) == expected_train_steps, (
        f"train full-result steps are {sorted(full_results['train'])}"
    )
    assert set(full_results["validation"]) == expected_validation_steps, (
        f"validation full-result steps are {sorted(full_results['validation'])}"
    )
    for split, expected_steps in (
        ("train", expected_train_steps),
        ("validation", expected_validation_steps),
    ):
        assert set(table_paths[split]) == expected_steps
        for step in expected_steps:
            assert table_paths[split][step], f"{split} step {step}: no result table"
            if split == "validation":
                assert len(table_paths[split][step]) == 1, (
                    f"validation step {step}: expected one full-result table, "
                    f"found {len(table_paths[split][step])}"
                )

    train_paths = _indexed_jsonl_paths(log_root, "train_data_step")
    validation_paths = _indexed_jsonl_paths(log_root, "val_data_step")
    assert set(train_paths) == expected_train_steps, (
        f"train JSONL steps are {sorted(train_paths)}"
    )
    assert set(validation_paths) == expected_validation_steps, (
        f"validation JSONL steps are {sorted(validation_paths)}"
    )

    train_summaries = {
        step: _select_accepted_train_table(
            step=step,
            candidates=full_results["train"][step],
            candidate_paths=table_paths["train"][step],
            train_path=train_paths[step],
            expected_rollouts=args.expected_train_rollouts,
            pad_token_id=pad_token_id,
            sequence_multiple=sequence_multiple,
            max_sequence_length=max_sequence_length,
            probability_error_threshold=probability_error_threshold,
        )
        for step in sorted(expected_train_steps)
    }
    validation_summaries = {
        step: _validate_validation_step(
            step=step,
            results=full_results["validation"][step][0],
            validation_path=validation_paths[step],
            expected_rollouts=args.expected_validation_rollouts,
        )
        for step in sorted(expected_validation_steps)
    }
    initial_validation = validation_summaries[0]
    final_validation = validation_summaries[args.expected_max_steps]
    assert initial_validation["reward_mean"] < 1.0, (
        "initial validation is already saturated at reward 1"
    )
    assert final_validation["reward_mean"] > initial_validation["reward_mean"], (
        "validation reward did not increase from step 0 to step 6"
    )
    assert final_validation["reward_mean"] >= 0.75, (
        "step-6 validation did not reach the intended obvious-overfit threshold"
    )

    tensorboard = _load_tensorboard_metrics(log_root, expected_train_steps)
    metrics = tensorboard["metrics"]
    for step in sorted(expected_train_steps):
        assert math.isfinite(metrics["train/loss"][step])
        assert metrics["train/grad_norm"][step] > 0, f"step {step}: grad norm is zero"
        assert metrics["train/global_valid_toks"][step] > 0, (
            f"step {step}: global valid-token denominator is zero"
        )
        assert metrics["train/lr"][step] > 0, f"step {step}: learning rate is zero"
        assert metrics["train/advantages/max"][step] > 0
        assert metrics["train/advantages/min"][step] < 0
        assert (
            metrics["train/token_mult_prob_error"][step] <= probability_error_threshold
        ), (
            f"step {step}: TensorBoard token_mult_prob_error exceeds "
            f"{probability_error_threshold}"
        )
    checkpoint = _validate_checkpoint(run_root, args.expected_max_steps)

    print(
        json.dumps(
            {
                "status": "ok",
                "run_root": str(run_root),
                "updates": args.expected_max_steps,
                "model": str(Path(args.model).resolve()),
                "config": str(args.config.resolve()),
                "pad_token_id": pad_token_id,
                "sequence_multiple": sequence_multiple,
                "probability_error_threshold": probability_error_threshold,
                "train": train_summaries,
                "validation": validation_summaries,
                "validation_reward_shift": (
                    final_validation["reward_mean"] - initial_validation["reward_mean"]
                ),
                "tensorboard": tensorboard,
                "checkpoint": checkpoint,
                "full_result_tables": {
                    split: {
                        step: [str(path) for path in paths]
                        for step, paths in sorted(step_paths.items())
                    }
                    for split, step_paths in table_paths.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
