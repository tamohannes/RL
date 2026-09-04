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

"""Validate reward signal, optimizer work, and artifacts from the signal canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any

PROBE_ID = "q3:c013:d0"
TOOL_NAME = "stateful_python_code_exec"
GOLD_SHA256 = "45f51cc52d4093ee60d941fc093653b0497f9b076b0de5b6b8175a0f945df36c"
ANSWER_KEYS = {
    "n_samples",
    "total_aligned_reads",
    "total_modified_reads",
    "mean_modified_pct",
    "max_modified_pct",
    "max_modified_sample",
}
GOLD = {
    "n_samples": 4,
    "total_aligned_reads": 111766,
    "total_modified_reads": 69429,
    "mean_modified_pct": 62.19,
    "max_modified_pct": 67.75,
    "max_modified_sample": "S4",
}
REQUIRED_TENSORBOARD_TAGS = {
    "train/loss",
    "train/grad_norm",
    "train/lr",
    "train/global_valid_toks",
    "train/advantages/max",
    "train/advantages/min",
}


def _load_full_results(log_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    results: list[dict[str, Any]] = []
    sources: list[Path] = []
    for path in sorted(log_root.rglob("*.table.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        columns = payload.get("columns")
        rows = payload.get("data")
        if not isinstance(columns, list) or not isinstance(rows, list):
            continue
        if "Full result" not in columns:
            continue
        column_index = columns.index("Full result")
        for row in rows:
            if not isinstance(row, list) or column_index >= len(row):
                raise AssertionError(f"malformed W&B table row in {path}")
            value = row[column_index]
            if isinstance(value, str):
                value = json.loads(value)
            if not isinstance(value, dict):
                raise AssertionError(f"full_result is not an object in {path}")
            results.append(value)
        sources.append(path)
    return results, sources


def _parse_arguments(item: dict[str, Any]) -> dict[str, Any]:
    arguments = item.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise AssertionError("function-call arguments are not a JSON object")
    return arguments


def _last_assistant_text(output_items: list[dict[str, Any]]) -> str:
    messages = [
        item
        for item in output_items
        if item.get("type") == "message" and item.get("role") == "assistant"
    ]
    if not messages:
        raise AssertionError("response contains no assistant message")
    chunks: list[str] = []
    content = messages[-1].get("content")
    if isinstance(content, str):
        chunks.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "\n".join(chunks).strip()


def _validate_tool_pairs(output_items: list[dict[str, Any]], index: int) -> int:
    calls = [
        (position, item)
        for position, item in enumerate(output_items)
        if item.get("type") == "function_call"
    ]
    outputs = [
        (position, item)
        for position, item in enumerate(output_items)
        if item.get("type") == "function_call_output"
    ]
    if not calls:
        raise AssertionError(f"rollout {index}: no tool call")
    call_ids: list[str] = []
    for position, item in calls:
        assert item.get("name") == TOOL_NAME, (
            f"rollout {index}: unexpected tool {item.get('name')!r}"
        )
        _parse_arguments(item)
        call_id = item.get("call_id")
        assert isinstance(call_id, str) and call_id
        call_ids.append(call_id)
        matching_positions = [
            output_position
            for output_position, output in outputs
            if output.get("call_id") == call_id
        ]
        assert len(matching_positions) == 1, (
            f"rollout {index}: call {call_id!r} has {len(matching_positions)} outputs"
        )
        assert position < matching_positions[0], (
            f"rollout {index}: output precedes call {call_id!r}"
        )
    assert len(set(call_ids)) == len(call_ids), f"rollout {index}: duplicate call IDs"
    assert len(outputs) == len(calls), f"rollout {index}: orphan tool output"
    return len(calls)


def _same_typed_value(candidate: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or isinstance(candidate, bool):
        return type(candidate) is type(expected) and candidate == expected
    if isinstance(expected, int):
        return isinstance(candidate, int) and candidate == expected
    if isinstance(expected, float):
        return isinstance(candidate, (int, float)) and candidate == expected
    return type(candidate) is type(expected) and candidate == expected


def _structured_gold_checks(answer: Any) -> list[list[Any]]:
    if not isinstance(answer, dict):
        return [["answer_is_object", False]]
    return [
        [f"{key}_matches_gold", _same_typed_value(answer.get(key), expected)]
        for key, expected in GOLD.items()
    ]


def _validate_result(result: dict[str, Any], index: int) -> dict[str, Any]:
    assert result.get("id") == PROBE_ID, f"rollout {index}: wrong id"
    assert result.get("probe_id") == PROBE_ID, f"rollout {index}: wrong probe_id"
    assert result.get("verifier_type") == "sciprobe_checks"
    row_index = result.get("_rowidx")
    assert isinstance(row_index, int), f"rollout {index}: missing integer _rowidx"
    assert result.get("_ng_task_index") == 0
    assert result.get("_ng_rollout_index") == row_index
    assert result.get("_ng_attempt_index") == 0

    response = result.get("response")
    assert isinstance(response, dict), f"rollout {index}: missing response"
    assert response.get("status") == "completed", (
        f"rollout {index}: response status={response.get('status')!r}"
    )
    assert response.get("error") is None, f"rollout {index}: response error"
    assert response.get("incomplete_details") is None, (
        f"rollout {index}: incomplete response"
    )
    output_items = response.get("output")
    assert isinstance(output_items, list) and all(
        isinstance(item, dict) for item in output_items
    ), f"rollout {index}: response.output is malformed"
    assert output_items and output_items[-1].get("type") == "message", (
        f"rollout {index}: response does not end in a message"
    )
    num_tool_calls = _validate_tool_pairs(output_items, index)
    assert int(result.get("num_tool_calls", -1)) == num_tool_calls
    assert int(result.get("tool_timeout_count", -1)) == 0
    assert int(result.get("tool_request_timeout_count", -1)) == 0

    create_params = result.get("responses_create_params")
    assert isinstance(create_params, dict)
    assert create_params.get("max_output_tokens") == 8192

    final_text = _last_assistant_text(output_items)
    assert final_text, f"rollout {index}: empty final answer"
    try:
        final_answer = json.loads(final_text)
    except json.JSONDecodeError:
        local_parse_status = "invalid_json"
        final_answer = final_text
        local_exact_key_set = False
    else:
        local_parse_status = "ok" if isinstance(final_answer, dict) else "not_object"
        local_exact_key_set = (
            isinstance(final_answer, dict) and set(final_answer) == ANSWER_KEYS
        )

    assert result.get("delegated_response") is None
    check_results = _structured_gold_checks(final_answer)
    assert all(
        isinstance(row, list)
        and len(row) == 2
        and isinstance(row[0], str)
        and isinstance(row[1], bool)
        for row in check_results
    )
    assert check_results, f"rollout {index}: gold comparator returned no checks"
    reward = float(result.get("reward"))
    assert reward in {0.0, 1.0}, f"rollout {index}: reward={reward}"
    expected_reward = (
        1.0
        if local_parse_status == "ok"
        and local_exact_key_set
        and check_results
        and all(passed for _, passed in check_results)
        else 0.0
    )
    assert reward == expected_reward
    if reward == 1.0:
        assert isinstance(final_answer, dict)
        assert set(final_answer) == ANSWER_KEYS
        assert all(passed for _, passed in check_results)

    usage = response.get("usage")
    output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
    assert isinstance(output_tokens, int) and output_tokens >= 0
    return {
        "row_index": row_index,
        "reward": reward,
        "num_tool_calls": num_tool_calls,
        "output_tokens": output_tokens,
        "question": result.get("question"),
        "create_params": create_params,
    }


def _contiguous_true_spans(mask: list[Any]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for position, value in enumerate(mask):
        assert value in {0, 1, 0.0, 1.0, False, True}, (
            f"token_loss_mask[{position}] is not binary: {value!r}"
        )
        if bool(value) and start is None:
            start = position
        elif not bool(value) and start is not None:
            spans.append((start, position))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _one_sequence(row: dict[str, Any], field: str, index: int) -> list[Any]:
    value = row.get(field)
    assert isinstance(value, list) and len(value) == 1, (
        f"train row {index}: {field} must have one sequence"
    )
    sequence = value[0]
    assert isinstance(sequence, list), f"train row {index}: {field} is not nested"
    return sequence


def _validate_train_data(
    log_root: Path,
    reward_by_index: dict[int, float],
    *,
    max_sequence_length: int,
    generation_cap: int,
) -> dict[str, Any]:
    paths = sorted(log_root.rglob("train_data_step1.jsonl"))
    assert len(paths) == 1, (
        f"expected one train_data_step1.jsonl under {log_root}, found {len(paths)}"
    )
    rows = [
        json.loads(line)
        for line in paths[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 8, f"expected 8 train rows, found {len(rows)}"

    active_advantages: list[float] = []
    span_lengths: list[int] = []
    active_tokens = 0
    sequence_lengths: list[int] = []
    rewards: list[float] = []
    indexes: list[int] = []
    for position, row in enumerate(rows):
        assert isinstance(row, dict)
        index = row.get("idx")
        assert isinstance(index, int), f"train row {position}: missing idx"
        indexes.append(index)
        reward_values = row.get("rewards")
        assert isinstance(reward_values, list) and len(reward_values) == 1
        reward = float(reward_values[0])
        assert reward == reward_by_index[index]
        rewards.append(reward)
        sample_mask = row.get("sample_loss_mask")
        assert isinstance(sample_mask, list) and len(sample_mask) == 1
        assert float(sample_mask[0]) > 0
        input_lengths = row.get("input_lengths")
        assert isinstance(input_lengths, list) and len(input_lengths) == 1
        assert isinstance(input_lengths[0], int) and input_lengths[0] > 0

        token_ids = _one_sequence(row, "token_ids", position)
        token_mask = _one_sequence(row, "token_loss_mask", position)
        advantages = _one_sequence(row, "advantages", position)
        generation_logprobs = _one_sequence(row, "generation_logprobs", position)
        prev_logprobs = _one_sequence(row, "prev_logprobs", position)
        sequence_length = len(token_ids)
        sequence_lengths.append(sequence_length)
        assert 0 < sequence_length <= max_sequence_length
        assert input_lengths[0] <= sequence_length
        assert all(
            len(sequence) == sequence_length
            for sequence in (
                token_mask,
                advantages,
                generation_logprobs,
                prev_logprobs,
            )
        )
        assert all(isinstance(token, int) for token in token_ids)

        spans = _contiguous_true_spans(token_mask)
        assert spans, f"train row {position}: no trainable model tokens"
        for start, end in spans:
            span_length = end - start
            assert span_length < generation_cap, (
                f"train row {position}: model turn hit/exceeded {generation_cap} "
                f"tokens ({span_length})"
            )
            span_lengths.append(span_length)
        for token_position, enabled in enumerate(token_mask):
            if not bool(enabled):
                continue
            active_tokens += 1
            advantage = float(advantages[token_position])
            generation_logprob = float(generation_logprobs[token_position])
            prev_logprob = float(prev_logprobs[token_position])
            assert math.isfinite(advantage)
            assert math.isfinite(generation_logprob)
            assert math.isfinite(prev_logprob)
            active_advantages.append(advantage)

    assert set(indexes) == set(range(8)), f"train indexes are {sorted(indexes)}"
    assert any(value > 0 for value in active_advantages), (
        "train data contains no positive advantage"
    )
    assert any(value < 0 for value in active_advantages), (
        "train data contains no negative advantage"
    )
    return {
        "path": str(paths[0]),
        "indexes": indexes,
        "rewards": rewards,
        "sequence_lengths": sequence_lengths,
        "active_tokens": active_tokens,
        "advantage_max": max(active_advantages),
        "advantage_min": min(active_advantages),
        "turn_span_lengths": span_lengths,
        "longest_turn_span": max(span_lengths),
        "near95_turns": sum(length >= 0.95 * generation_cap for length in span_lengths),
        "maxhit_turns": sum(length >= generation_cap for length in span_lengths),
    }


def _load_tensorboard_metrics(log_root: Path) -> dict[str, Any]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_paths = sorted(log_root.rglob("events.out.tfevents.*"))
    assert event_paths, f"no TensorBoard event files under {log_root}"
    errors: dict[str, str] = {}
    for path in event_paths:
        try:
            accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
            accumulator.Reload()
            scalar_tags = set(accumulator.Tags().get("scalars", []))
            if not REQUIRED_TENSORBOARD_TAGS.issubset(scalar_tags):
                continue
            metrics: dict[str, dict[str, float | int]] = {}
            for tag in sorted(REQUIRED_TENSORBOARD_TAGS):
                events = accumulator.Scalars(tag)
                assert events, f"{path}: tag {tag} has no scalar events"
                event = max(events, key=lambda item: (item.step, item.wall_time))
                value = float(event.value)
                assert math.isfinite(value), f"{path}: {tag} is non-finite"
                metrics[tag] = {"step": int(event.step), "value": value}
            assert {item["step"] for item in metrics.values()} == {1}, (
                f"{path}: required training metrics are not all at step 1"
            )
            return {
                "path": str(path),
                "metrics": metrics,
                "available_scalar_tags": sorted(scalar_tags),
            }
        except Exception as error:
            errors[str(path)] = f"{type(error).__name__}: {error}"
    raise AssertionError(
        "no TensorBoard event file contains the complete training proof; "
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
        raw_header = handle.read(header_length)
    header = json.loads(raw_header)
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
    mtp_tensor_names = [
        name
        for name in header
        if name != "__metadata__" and (name.startswith("mtp.") or ".mtp." in name)
    ]
    assert not mtp_tensor_names, f"unexpected trainer-only MTP tensors: {path}"
    assert tensor_count > 0, f"no tensors in {path}"
    assert maximum_end == payload_size, f"incomplete safetensors payload: {path}"
    return {
        "path": str(path),
        "bytes": file_size,
        "header_bytes": header_length,
        "tensor_count": tensor_count,
        "mtp_tensor_count": len(mtp_tensor_names),
    }


def _validate_checkpoint(run_root: Path) -> dict[str, Any]:
    checkpoint_root = run_root / "checkpoints"
    status_path = checkpoint_root / "latest_checkpoint_status.json"
    assert status_path.is_file(), f"missing checkpoint status: {status_path}"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert isinstance(status, dict)
    assert status.get("last_checkpoint_step") == 1
    assert float(status.get("last_successful_ckpt_save_completion", 0)) > 0
    step_root = checkpoint_root / "step_1"
    assert step_root.is_dir(), f"missing step_1 checkpoint: {step_root}"
    tensor_paths = sorted(step_root.rglob("*.safetensors"))
    assert tensor_paths, f"no safetensors under {step_root}"
    return {
        "status_path": str(status_path),
        "status": status,
        "safetensors": [_validate_safetensors(path) for path in tensor_paths],
    }


def _validate_secrets_absent_from_logs(
    run_root: Path,
    secrets: dict[str, str],
) -> dict[str, Any]:
    encoded = {name: value.encode("utf-8") for name, value in secrets.items()}
    max_secret_length = max(len(value) for value in encoded.values())
    candidates: set[Path] = set()
    for directory_name in ("logs", "nemo-gym", "ray"):
        directory = run_root / directory_name
        if not directory.is_dir():
            continue
        candidates.update(
            path
            for path in directory.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
    for pattern in ("slurm-*.out", "slurm-*.err"):
        candidates.update(
            path
            for path in run_root.glob(pattern)
            if path.is_file() and not path.is_symlink()
        )

    scanned_bytes = 0
    for path in sorted(candidates):
        carry = b""
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                scanned_bytes += len(chunk)
                payload = carry + chunk
                for name, secret in encoded.items():
                    assert secret not in payload, f"{name} leaked into log {path}"
                carry = payload[-(max_secret_length - 1) :]
    return {
        "files": len(candidates),
        "bytes": scanned_bytes,
        "secret_values_checked": sorted(secrets),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=12288)
    parser.add_argument("--generation-cap", type=int, default=8192)
    args = parser.parse_args()

    encoded_gold = json.dumps(
        GOLD,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert hashlib.sha256(encoded_gold).hexdigest() == GOLD_SHA256

    run_root = args.run_root.resolve()
    log_root = run_root / "logs"
    assert log_root.is_dir(), f"missing log root: {log_root}"
    results, table_paths = _load_full_results(log_root)
    assert table_paths, f"no full-result W&B tables under {log_root}"
    assert len(results) == args.expected_rollouts, (
        f"expected {args.expected_rollouts} rollouts, found {len(results)}"
    )
    serialized_results = json.dumps(results, sort_keys=True)
    for forbidden_marker in (
        "_sciprobe_verifier_capability",
        "X-SciProbe-Rollout-Capability",
        "X-SciProbe-Trusted-Ingress",
        "SCIPROBE_TRUSTED_INGRESS_TOKEN",
        "SCIPROBE_POLICY_GENERATION_TOKEN",
        "candidate_sha256",
        "check_results",
        "checks_sha256",
        "data_bytes",
        "data_files",
        "data_tree_sha256",
        "exact_key_set",
        "extracted_answer",
        "failed_checks",
        "grader_status",
        "gold_sha256",
        "parse_status",
        "source_ref",
    ):
        assert forbidden_marker not in serialized_results
    secret_values: dict[str, str] = {}
    for secret_env in (
        "SCIPROBE_CAPABILITY_SIGNING_KEY",
        "SCIPROBE_VERIFIER_TOKEN",
        "SCIPROBE_TRUSTED_INGRESS_TOKEN",
        "SCIPROBE_POLICY_GENERATION_TOKEN",
    ):
        secret = os.environ.get(secret_env, "")
        assert len(secret) >= 32, f"{secret_env} is not configured"
        secret_values[secret_env] = secret
        assert secret not in serialized_results, (
            f"{secret_env} leaked into full results"
        )
    secret_log_scan = _validate_secrets_absent_from_logs(
        run_root,
        secret_values,
    )
    summaries = [
        _validate_result(result, index) for index, result in enumerate(results)
    ]
    row_indexes = [summary["row_index"] for summary in summaries]
    assert set(row_indexes) == set(range(args.expected_rollouts)), row_indexes
    rewards = [summary["reward"] for summary in summaries]
    assert set(rewards) == {0.0, 1.0}, (
        f"same-probe group lacks mixed binary rewards: {rewards}"
    )
    questions = {summary["question"] for summary in summaries}
    create_params = {
        json.dumps(summary["create_params"], sort_keys=True) for summary in summaries
    }
    assert len(questions) == 1 and len(create_params) == 1, (
        "rollouts do not form one repeated-prompt group"
    )
    reward_by_index = {summary["row_index"]: summary["reward"] for summary in summaries}

    train_data = _validate_train_data(
        log_root,
        reward_by_index,
        max_sequence_length=args.max_sequence_length,
        generation_cap=args.generation_cap,
    )
    tensorboard = _load_tensorboard_metrics(log_root)
    tb_metrics = tensorboard["metrics"]
    assert abs(float(tb_metrics["train/loss"]["value"])) > 0, "train/loss is zero"
    assert tb_metrics["train/grad_norm"]["value"] > 0
    assert tb_metrics["train/lr"]["value"] > 0
    assert tb_metrics["train/global_valid_toks"]["value"] > 0
    assert tb_metrics["train/advantages/max"]["value"] > 0
    assert tb_metrics["train/advantages/min"]["value"] < 0
    assert math.isclose(
        float(tb_metrics["train/global_valid_toks"]["value"]),
        float(train_data["active_tokens"]),
        rel_tol=0,
        abs_tol=0.5,
    )
    assert math.isclose(
        float(tb_metrics["train/advantages/max"]["value"]),
        float(train_data["advantage_max"]),
        rel_tol=1e-5,
        abs_tol=1e-6,
    )
    assert math.isclose(
        float(tb_metrics["train/advantages/min"]["value"]),
        float(train_data["advantage_min"]),
        rel_tol=1e-5,
        abs_tol=1e-6,
    )
    checkpoint = _validate_checkpoint(run_root)

    print(
        json.dumps(
            {
                "status": "ok",
                "run_root": str(run_root),
                "full_result_tables": [str(path) for path in table_paths],
                "rollouts": len(results),
                "same_probe_id": PROBE_ID,
                "rollout_indexes": sorted(row_indexes),
                "rewards": [reward_by_index[index] for index in sorted(row_indexes)],
                "pass_at_1": sum(rewards) / len(rewards),
                "tool_calls_per_rollout": [
                    summary["num_tool_calls"] for summary in summaries
                ],
                "response_output_tokens": [
                    summary["output_tokens"] for summary in summaries
                ],
                "generation_health": {
                    "per_turn_cap": args.generation_cap,
                    "turn_span_lengths": train_data["turn_span_lengths"],
                    "longest_turn_span": train_data["longest_turn_span"],
                    "near95": train_data["near95_turns"],
                    "maxhit": train_data["maxhit_turns"],
                    "empty_final": 0,
                },
                "train_data": train_data,
                "tensorboard": tensorboard,
                "checkpoint": checkpoint,
                "secret_log_scan": secret_log_scan,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
