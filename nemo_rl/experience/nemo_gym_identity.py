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

"""Canonical NeMo Gym rollout identity stamping."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

NEMO_GYM_TASK_INDEX_KEY = "_ng_task_index"
NEMO_GYM_ROLLOUT_INDEX_KEY = "_ng_rollout_index"
NEMO_GYM_ATTEMPT_INDEX_KEY = "_ng_attempt_index"


def _nonnegative_int(value: Any, field: str) -> int:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def infer_nemo_gym_rollouts_per_prompt(
    input_indices: Sequence[Any] | None,
    *,
    batch_size: int,
) -> int:
    """Infer a repeated prompt-group width from consecutive dataset indices."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if input_indices is None:
        return 1
    if len(input_indices) != batch_size:
        raise ValueError("NeMo-Gym input-index count must match the rollout-row count")

    first_index = _nonnegative_int(input_indices[0], "input_batch.idx")
    group_size = 1
    while group_size < batch_size:
        next_index = _nonnegative_int(input_indices[group_size], "input_batch.idx")
        if next_index != first_index:
            break
        group_size += 1
    return group_size


def stamp_nemo_gym_rollout_identity(
    rows: list[dict[str, Any]],
    *,
    input_indices: Sequence[Any] | None,
    rollouts_per_prompt: int,
    attempt_index: int,
) -> None:
    """Stamp canonical task, rollout, attempt, and transport row indices."""
    if rollouts_per_prompt <= 0:
        raise ValueError("rollouts_per_prompt must be greater than zero")
    if len(rows) % rollouts_per_prompt != 0:
        raise ValueError("NeMo-Gym row count must be divisible by rollouts_per_prompt")
    attempt_index = _nonnegative_int(attempt_index, "attempt_index")
    if input_indices is not None and len(input_indices) != len(rows):
        raise ValueError("NeMo-Gym input-index count must match the rollout-row count")

    seen_task_indices: set[int] = set()
    for group_start in range(0, len(rows), rollouts_per_prompt):
        group_rows = rows[group_start : group_start + rollouts_per_prompt]
        if input_indices is not None:
            group_input_index = _nonnegative_int(
                input_indices[group_start], "input_batch.idx"
            )
            for row_index in range(group_start, group_start + rollouts_per_prompt):
                row_input_index = _nonnegative_int(
                    input_indices[row_index], "input_batch.idx"
                )
                if row_input_index != group_input_index:
                    raise ValueError(
                        "Every rollout in a prompt group must have the same input index"
                    )
        else:
            group_input_index = group_start // rollouts_per_prompt

        existing_task_values = [row.get(NEMO_GYM_TASK_INDEX_KEY) for row in group_rows]
        if any(value is not None for value in existing_task_values) and any(
            value is None for value in existing_task_values
        ):
            raise ValueError(
                "Every rollout in a prompt group must carry the task index together"
            )
        existing_task_indices = {
            _nonnegative_int(value, NEMO_GYM_TASK_INDEX_KEY)
            for value in existing_task_values
            if value is not None
        }
        if len(existing_task_indices) > 1:
            raise ValueError("Every rollout in a prompt group must have one task index")
        task_index = (
            existing_task_indices.pop() if existing_task_indices else group_input_index
        )
        if task_index in seen_task_indices:
            raise ValueError("Every prompt group must have a unique task index")
        seen_task_indices.add(task_index)

        for rollout_index, row in enumerate(group_rows):
            row_index = group_start + rollout_index
            existing_rollout_index = row.get(NEMO_GYM_ROLLOUT_INDEX_KEY)
            if (
                existing_rollout_index is not None
                and _nonnegative_int(existing_rollout_index, NEMO_GYM_ROLLOUT_INDEX_KEY)
                != rollout_index
            ):
                raise ValueError("NeMo-Gym rollout index conflicts with row order")
            existing_attempt_index = row.get(NEMO_GYM_ATTEMPT_INDEX_KEY)
            if existing_attempt_index is not None:
                _nonnegative_int(existing_attempt_index, NEMO_GYM_ATTEMPT_INDEX_KEY)

            row[NEMO_GYM_TASK_INDEX_KEY] = task_index
            row[NEMO_GYM_ROLLOUT_INDEX_KEY] = rollout_index
            row[NEMO_GYM_ATTEMPT_INDEX_KEY] = attempt_index
            row["_rowidx"] = row_index
