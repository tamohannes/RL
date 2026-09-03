from __future__ import annotations

import pytest

from nemo_rl.experience.nemo_gym_identity import (
    infer_nemo_gym_rollouts_per_prompt,
    stamp_nemo_gym_rollout_identity,
)


class ScalarIndex:
    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value


def test_stamp_nemo_gym_rollout_identity_is_canonical_across_groups() -> None:
    rows = [{} for _ in range(3)] + [{"_ng_task_index": 42} for _ in range(3)]

    stamp_nemo_gym_rollout_identity(
        rows,
        input_indices=[ScalarIndex(value) for value in (10, 10, 10, 20, 20, 20)],
        rollouts_per_prompt=3,
        attempt_index=2,
    )

    assert [row["_ng_task_index"] for row in rows] == [10, 10, 10, 42, 42, 42]
    assert [row["_ng_rollout_index"] for row in rows] == [0, 1, 2, 0, 1, 2]
    assert [row["_ng_attempt_index"] for row in rows] == [2] * 6
    assert [row["_rowidx"] for row in rows] == list(range(6))

    # A collection retry reuses the rows and advances only the attempt index.
    stamp_nemo_gym_rollout_identity(
        rows,
        input_indices=[10, 10, 10, 20, 20, 20],
        rollouts_per_prompt=3,
        attempt_index=3,
    )
    assert [row["_ng_attempt_index"] for row in rows] == [3] * 6


def test_infer_nemo_gym_rollouts_per_prompt_from_repeated_indices() -> None:
    assert infer_nemo_gym_rollouts_per_prompt([4, 4, 8, 8], batch_size=4) == 2
    assert infer_nemo_gym_rollouts_per_prompt([4, 8], batch_size=2) == 1
    assert infer_nemo_gym_rollouts_per_prompt(None, batch_size=2) == 1


def test_stamp_nemo_gym_rollout_identity_rejects_conflicting_rollout() -> None:
    with pytest.raises(ValueError, match="rollout index conflicts"):
        stamp_nemo_gym_rollout_identity(
            [{"_ng_rollout_index": 7}],
            input_indices=[0],
            rollouts_per_prompt=1,
            attempt_index=0,
        )


@pytest.mark.parametrize(
    "rows,input_indices,error",
    [
        (
            [{"_ng_task_index": 1}, {"_ng_task_index": 2}],
            [10, 10],
            "one task index",
        ),
        (
            [{"_ng_task_index": 1}, {}],
            [10, 10],
            "carry the task index together",
        ),
        ([{}, {}], [10, 11], "same input index"),
        ([{}, {}], [10, 10], "unique task index"),
    ],
)
def test_stamp_nemo_gym_rollout_identity_rejects_ambiguous_task_groups(
    rows: list[dict[str, int]],
    input_indices: list[int],
    error: str,
) -> None:
    group_size = 2 if "unique" not in error else 1
    with pytest.raises(ValueError, match=error):
        stamp_nemo_gym_rollout_identity(
            rows,
            input_indices=input_indices,
            rollouts_per_prompt=group_size,
            attempt_index=0,
        )


@pytest.mark.parametrize("attempt_index", [True, -1, 0.5])
def test_stamp_nemo_gym_rollout_identity_rejects_invalid_attempt(
    attempt_index: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="attempt_index"):
        stamp_nemo_gym_rollout_identity(
            [{}],
            input_indices=[0],
            rollouts_per_prompt=1,
            attempt_index=attempt_index,  # type: ignore[arg-type]
        )
