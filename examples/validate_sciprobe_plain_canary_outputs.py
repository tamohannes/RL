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

"""Validate structured outputs from the prompt-only SciProbe GRPO canary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from safetensors import safe_open


def _unwrap_singletons(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _scalar(row: dict[str, Any], key: str, index: int) -> float:
    value = _unwrap_singletons(row[key])
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(f"rollout {index}: {key} is not numeric")
    value = float(value)
    if not math.isfinite(value):
        raise AssertionError(f"rollout {index}: {key} is not finite")
    return value


def _vector(row: dict[str, Any], key: str, index: int) -> list[Any]:
    value = row[key]
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise AssertionError(f"rollout {index}: {key} is not a vector")
    return value


def _assert_finite_vector(values: list[Any], label: str) -> None:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AssertionError(f"{label} contains a non-numeric value")
        if not math.isfinite(float(value)):
            raise AssertionError(f"{label} contains a non-finite value")


def _validate_checkpoint_loadability(checkpoint_root: Path) -> dict[str, Any]:
    status_path = checkpoint_root / "latest_checkpoint_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    step = int(status["last_checkpoint_step"])
    step_dir = checkpoint_root / f"step_{step}"
    shard_paths = sorted(
        (step_dir / "policy" / "weights" / "model").glob("*.safetensors")
    )
    assert shard_paths, f"no safetensor shards under {step_dir}"

    shard_summaries: list[dict[str, Any]] = []
    for path in shard_paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            assert keys, f"{path}: empty safetensor index"
            candidates = []
            for key in keys:
                shape = list(handle.get_slice(key).get_shape())
                candidates.append((math.prod(shape), key, shape))
            _, smallest_key, smallest_shape = min(candidates)
            tensor = handle.get_tensor(smallest_key)
            assert list(tensor.shape) == smallest_shape
            shard_summaries.append(
                {
                    "file": path.name,
                    "tensor_keys": len(keys),
                    "loaded_tensor": smallest_key,
                    "loaded_shape": smallest_shape,
                }
            )

    return {
        "last_checkpoint_step": step,
        "checkpoint_dir": step_dir.name,
        "safetensor_shards": len(shard_paths),
        "shards": shard_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help="Only safe-open and minimally load every checkpoint shard.",
    )
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    checkpoint_root = run_root / "checkpoints"
    checkpoint_loadability = _validate_checkpoint_loadability(checkpoint_root)
    print(
        json.dumps({"checkpoint_loadability": checkpoint_loadability}, sort_keys=True)
    )
    if args.checkpoint_only:
        return
    log_root = run_root / "logs"
    data_paths = sorted(log_root.rglob("train_data_step1.jsonl"))
    assert len(data_paths) == 1, (
        f"expected one train_data_step1.jsonl under {log_root}, found {len(data_paths)}"
    )
    rows = [
        json.loads(line)
        for line in data_paths[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == args.expected_rollouts, (
        f"expected {args.expected_rollouts} training rollouts, found {len(rows)}"
    )

    rewards: list[float] = []
    advantages: list[float] = []
    sequence_lengths: list[int] = []
    generation_lengths: list[int] = []
    required = {
        "rewards",
        "input_lengths",
        "token_ids",
        "token_loss_mask",
        "sample_loss_mask",
        "advantages",
        "generation_logprobs",
        "prev_logprobs",
    }
    for index, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            raise AssertionError(f"rollout {index}: missing fields {sorted(missing)}")

        reward = _scalar(row, "rewards", index)
        rewards.append(reward)
        input_length = int(_scalar(row, "input_lengths", index))
        token_ids = _vector(row, "token_ids", index)
        token_mask = _vector(row, "token_loss_mask", index)
        generation_logprobs = _vector(row, "generation_logprobs", index)
        prev_logprobs = _vector(row, "prev_logprobs", index)
        advantage_values = _vector(row, "advantages", index)

        assert token_ids, f"rollout {index}: empty token_ids"
        assert len(token_ids) <= args.max_sequence_length, (
            f"rollout {index}: {len(token_ids)} tokens exceed cap {args.max_sequence_length}"
        )
        assert 0 < input_length <= len(token_ids), (
            f"rollout {index}: invalid input length {input_length}/{len(token_ids)}"
        )
        assert len(token_mask) == len(token_ids), (
            f"rollout {index}: token/mask lengths differ"
        )
        assert len(generation_logprobs) == len(prev_logprobs), (
            f"rollout {index}: generation/previous logprob lengths differ"
        )
        assert len(generation_logprobs) <= len(token_ids), (
            f"rollout {index}: more generation logprobs than tokens"
        )
        _assert_finite_vector(token_mask, f"rollout {index} token_loss_mask")
        _assert_finite_vector(
            generation_logprobs, f"rollout {index} generation_logprobs"
        )
        _assert_finite_vector(prev_logprobs, f"rollout {index} prev_logprobs")
        _assert_finite_vector(advantage_values, f"rollout {index} advantages")

        advantages.extend(float(value) for value in advantage_values)
        sequence_lengths.append(len(token_ids))
        generation_lengths.append(len(generation_logprobs))

    assert len(set(rewards)) > 1, (
        f"reward group has no variance: {rewards}; GRPO would have no learning signal"
    )
    assert any(abs(value) > 1e-8 for value in advantages), (
        "all advantages are zero despite reward variance"
    )

    checkpoint_entries = (
        sorted(path.name for path in checkpoint_root.iterdir())
        if checkpoint_root.is_dir()
        else []
    )
    assert checkpoint_entries, f"no checkpoint artifacts under {checkpoint_root}"

    print(
        json.dumps(
            {
                "status": "ok",
                "run_root": str(run_root),
                "train_data": str(data_paths[0]),
                "rollouts": len(rows),
                "rewards": rewards,
                "reward_variance_present": True,
                "nonzero_advantages": sum(abs(value) > 1e-8 for value in advantages),
                "sequence_lengths": sequence_lengths,
                "generation_logprob_lengths": generation_lengths,
                "checkpoint_entries": checkpoint_entries,
                "checkpoint_loadability": checkpoint_loadability,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
