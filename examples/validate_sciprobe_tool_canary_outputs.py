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

"""Validate exact NeMo Gym traces from the SciProbe stateful-tool canary."""

from __future__ import annotations

import argparse
import ast
import json
import math
import struct
from pathlib import Path
from typing import Any

FIRST_ASSIGNMENT = 'state_token = sum(ord(ch) for ch in "SciProbe")'
FIRST_ASSIGNMENT_AST = ast.dump(
    ast.parse(FIRST_ASSIGNMENT).body[0], include_attributes=False
)
SECOND_CODE = "state_token"
EXPECTED_VALUE = "791"
TOOL_NAME = "stateful_python_code_exec"


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


def _message_text(output_items: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for item in output_items:
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
    return "\n".join(chunks)


def _matches_first_assignment(code: Any) -> bool:
    """Accept the required assignment plus an optional display of the variable."""
    if not isinstance(code, str):
        return False
    try:
        statements = ast.parse(code).body
    except SyntaxError:
        return False
    if not statements:
        return False
    if ast.dump(statements[0], include_attributes=False) != FIRST_ASSIGNMENT_AST:
        return False
    return all(
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Name)
        and statement.value.id == SECOND_CODE
        for statement in statements[1:]
    )


def _validate_safetensors(path: Path) -> dict[str, Any]:
    """Validate a safetensors header and every declared byte range with stdlib only."""
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_header_length = handle.read(8)
        if len(raw_header_length) != 8:
            raise AssertionError(f"truncated safetensors prefix: {path}")
        (header_length,) = struct.unpack("<Q", raw_header_length)
        if header_length <= 2 or 8 + header_length > file_size:
            raise AssertionError(
                f"invalid safetensors header length {header_length}: {path}"
            )
        raw_header = handle.read(header_length)
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssertionError(f"invalid safetensors JSON header: {path}") from error
    if not isinstance(header, dict):
        raise AssertionError(f"safetensors header is not an object: {path}")

    data_size = file_size - 8 - header_length
    tensor_count = 0
    maximum_end = 0
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, dict):
            raise AssertionError(f"invalid tensor metadata for {name!r}: {path}")
        offsets = metadata.get("data_offsets")
        shape = metadata.get("shape")
        dtype = metadata.get("dtype")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or not isinstance(shape, list)
            or not isinstance(dtype, str)
        ):
            raise AssertionError(f"incomplete tensor metadata for {name!r}: {path}")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise AssertionError(f"invalid tensor byte range for {name!r}: {path}")
        maximum_end = max(maximum_end, end)
        tensor_count += 1
    if tensor_count == 0:
        raise AssertionError(f"safetensors file contains no tensors: {path}")
    if maximum_end != data_size:
        raise AssertionError(
            f"safetensors payload ends at {maximum_end}, file has {data_size}: {path}"
        )
    return {
        "path": str(path),
        "bytes": file_size,
        "header_bytes": header_length,
        "tensor_count": tensor_count,
    }


def _validate_result(result: dict[str, Any], index: int) -> dict[str, Any]:
    response = result.get("response")
    if not isinstance(response, dict):
        raise AssertionError(f"rollout {index}: missing response object")
    incomplete_details = response.get("incomplete_details")
    incomplete_reason = (
        incomplete_details.get("reason")
        if isinstance(incomplete_details, dict)
        else None
    )
    cap_hit = incomplete_reason in {"max_output_tokens", "max_length"}
    if cap_hit:
        raise AssertionError(
            f"rollout {index}: generation hit its output cap ({incomplete_reason})"
        )
    output_items = response.get("output")
    if not isinstance(output_items, list) or not all(
        isinstance(item, dict) for item in output_items
    ):
        raise AssertionError(f"rollout {index}: response.output is not an object list")

    calls = [
        (position, item)
        for position, item in enumerate(output_items)
        if item.get("type") == "function_call"
    ]
    call_outputs = [
        (position, item)
        for position, item in enumerate(output_items)
        if item.get("type") == "function_call_output"
    ]
    parsed_calls = [
        {
            "position": position,
            "call_id": item.get("call_id"),
            "name": item.get("name"),
            "arguments": _parse_arguments(item),
        }
        for position, item in calls
    ]

    first_assignment_matches = (
        len(parsed_calls) == 2
        and parsed_calls[0]["name"] == TOOL_NAME
        and parsed_calls[1]["name"] == TOOL_NAME
        and _matches_first_assignment(parsed_calls[0]["arguments"].get("code"))
    )
    second_call_exact = (
        len(parsed_calls) == 2
        and parsed_calls[1]["arguments"].get("code") == SECOND_CODE
    )
    second_output = None
    if len(parsed_calls) >= 2:
        for _, item in call_outputs:
            if item.get("call_id") == parsed_calls[1]["call_id"]:
                second_output = item.get("output")
                break
    reused_state = (
        isinstance(second_output, str) and second_output.strip() == EXPECTED_VALUE
    )
    output_positions = {
        item.get("call_id"): position for position, item in call_outputs
    }
    ordered_call_outputs = (
        len(parsed_calls) == 2
        and parsed_calls[0]["call_id"] in output_positions
        and parsed_calls[1]["call_id"] in output_positions
        and parsed_calls[0]["position"]
        < output_positions[parsed_calls[0]["call_id"]]
        < parsed_calls[1]["position"]
        < output_positions[parsed_calls[1]["call_id"]]
    )
    response_id = response.get("id")
    response_scoped_continuity = (
        isinstance(response_id, str)
        and bool(response_id)
        and first_assignment_matches
        and second_call_exact
        and ordered_call_outputs
        and reused_state
    )

    reward = float(result["reward"])
    if not math.isfinite(reward):
        raise AssertionError(f"rollout {index}: non-finite reward")
    reported_calls = int(result.get("num_tool_calls", len(parsed_calls)))
    if reported_calls != len(parsed_calls):
        raise AssertionError(
            f"rollout {index}: num_tool_calls={reported_calls}, trace has {len(parsed_calls)}"
        )
    internal_timeouts = int(result.get("tool_timeout_count", 0))
    request_timeouts = int(result.get("tool_request_timeout_count", 0))
    if internal_timeouts or request_timeouts:
        raise AssertionError(
            f"rollout {index}: tool timeouts={internal_timeouts}+{request_timeouts}"
        )

    final_text = _message_text(output_items)
    if response_scoped_continuity and reward != 1.0:
        raise AssertionError(
            f"rollout {index}: stateful trace received reward {reward}"
        )

    usage = response.get("usage")
    output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
    if not isinstance(output_tokens, int) or output_tokens < 0:
        raise AssertionError(f"rollout {index}: missing response output-token count")
    create_params = result.get("responses_create_params")
    output_cap = (
        create_params.get("max_output_tokens")
        if isinstance(create_params, dict)
        else None
    )
    if not isinstance(output_cap, int) or output_cap <= 0:
        raise AssertionError(f"rollout {index}: missing positive output cap")

    return {
        "index": index,
        "reward": reward,
        "response_status": response.get("status"),
        "incomplete_reason": incomplete_reason,
        "output_cap_hit": cap_hit,
        "output_tokens": output_tokens,
        "output_cap": output_cap,
        "num_tool_calls": len(parsed_calls),
        "response_id": response_id,
        "first_assignment_matches": first_assignment_matches,
        "second_call_exact": second_call_exact,
        "ordered_call_outputs": ordered_call_outputs,
        "second_output": second_output,
        "reused_state": reused_state,
        "final_answer_present": "\\boxed{791}" in final_text,
        "visible_final_empty": not bool(final_text.strip()),
        "response_scoped_continuity": response_scoped_continuity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int, default=4)
    parser.add_argument(
        "--allow-constant-rewards",
        action="store_true",
        help="Validate mechanics-only artifacts even when GRPO had no learning signal.",
    )
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    log_root = run_root / "logs"
    assert log_root.is_dir(), f"missing log root: {log_root}"

    results, table_paths = _load_full_results(log_root)
    assert table_paths, f"no NeMo Gym full-result W&B table under {log_root}"
    assert len(results) == args.expected_rollouts, (
        f"expected {args.expected_rollouts} training rollouts, found {len(results)}"
    )

    summaries = [
        _validate_result(result, index) for index, result in enumerate(results)
    ]
    proof_count = sum(row["response_scoped_continuity"] for row in summaries)
    assert proof_count > 0, "no rollout proved response-scoped state reuse"
    rewards = [row["reward"] for row in summaries]
    reward_variance_present = len(set(rewards)) > 1
    if not args.allow_constant_rewards:
        assert reward_variance_present, (
            f"reward group has no variance: {rewards}; GRPO had no learning signal"
        )

    checkpoint_root = run_root / "checkpoints"
    checkpoint_entries = (
        sorted(path.name for path in checkpoint_root.iterdir())
        if checkpoint_root.is_dir()
        else []
    )
    assert checkpoint_entries, f"no checkpoint artifacts under {checkpoint_root}"
    status_path = checkpoint_root / "latest_checkpoint_status.json"
    assert status_path.is_file(), f"missing checkpoint status: {status_path}"
    checkpoint_status = json.loads(status_path.read_text(encoding="utf-8"))
    assert isinstance(checkpoint_status, dict), "checkpoint status is not an object"
    tensor_paths = sorted((checkpoint_root / "step_1").rglob("*.safetensors"))
    assert tensor_paths, "step_1 contains no safetensors shards"
    tensor_shards = [_validate_safetensors(path) for path in tensor_paths]

    output_caps = {row["output_cap"] for row in summaries}
    assert len(output_caps) == 1, f"inconsistent output caps: {sorted(output_caps)}"
    output_cap = output_caps.pop()
    output_token_counts = [row["output_tokens"] for row in summaries]
    maximum_output_tokens = max(output_token_counts)
    near95_count = sum(value >= 0.95 * output_cap for value in output_token_counts)
    maxhit_count = sum(value == output_cap for value in output_token_counts)
    empty_visible_final_count = sum(row["visible_final_empty"] for row in summaries)

    print(
        json.dumps(
            {
                "status": "ok",
                "run_root": str(run_root),
                "full_result_tables": [str(path) for path in table_paths],
                "rollouts": len(summaries),
                "rewards": rewards,
                "reward_variance_present": reward_variance_present,
                "response_scoped_stateful_proofs": proof_count,
                "generation_health": {
                    "output_cap": output_cap,
                    "output_tokens_per_rollout": output_token_counts,
                    "max_output_tokens_observed": maximum_output_tokens,
                    "near95_count": near95_count,
                    "maxhit_count": maxhit_count,
                    "empty_visible_final_count": empty_visible_final_count,
                },
                "checkpoint_entries": checkpoint_entries,
                "checkpoint_status": checkpoint_status,
                "safetensors_shards": tensor_shards,
                "trace_summaries": summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
