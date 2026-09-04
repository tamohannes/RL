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

"""Dependency-free deployment checks for Gym identity and raw token audit."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from nemo_rl.environments.nemo_gym import NemoGym
from nemo_rl.experience.nemo_gym_identity import (
    infer_nemo_gym_rollouts_per_prompt,
    stamp_nemo_gym_rollout_identity,
)


class _ScalarIndex:
    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value


class _Tokenizer:
    def batch_decode(self, batch: list[list[int]]) -> list[str]:
        return [" ".join(map(str, token_ids)) for token_ids in batch]


def _expect_error(
    action: Callable[[], None],
    error_types: tuple[type[Exception], ...],
    message_part: str,
) -> None:
    try:
        action()
    except error_types as error:
        assert message_part in str(error)
    else:
        raise AssertionError(f"expected {error_types} containing {message_part!r}")


def _validate_identity() -> None:
    rows: list[dict[str, Any]] = [{} for _ in range(3)] + [
        {"_ng_task_index": 42} for _ in range(3)
    ]
    input_indices = [_ScalarIndex(value) for value in (10, 10, 10, 20, 20, 20)]
    assert infer_nemo_gym_rollouts_per_prompt(input_indices, batch_size=6) == 3

    stamp_nemo_gym_rollout_identity(
        rows,
        input_indices=input_indices,
        rollouts_per_prompt=3,
        attempt_index=2,
    )
    assert [row["_ng_task_index"] for row in rows] == [10, 10, 10, 42, 42, 42]
    assert [row["_ng_rollout_index"] for row in rows] == [0, 1, 2, 0, 1, 2]
    assert [row["_ng_attempt_index"] for row in rows] == [2] * 6
    assert [row["_rowidx"] for row in rows] == list(range(6))

    stamp_nemo_gym_rollout_identity(
        rows,
        input_indices=[10, 10, 10, 20, 20, 20],
        rollouts_per_prompt=3,
        attempt_index=3,
    )
    assert [row["_ng_attempt_index"] for row in rows] == [3] * 6
    assert infer_nemo_gym_rollouts_per_prompt([4, 4, 8, 8], batch_size=4) == 2
    assert infer_nemo_gym_rollouts_per_prompt([4, 8], batch_size=2) == 1
    assert infer_nemo_gym_rollouts_per_prompt(None, batch_size=2) == 1

    _expect_error(
        lambda: stamp_nemo_gym_rollout_identity(
            [{"_ng_rollout_index": 7}],
            input_indices=[0],
            rollouts_per_prompt=1,
            attempt_index=0,
        ),
        (ValueError,),
        "rollout index conflicts",
    )
    _expect_error(
        lambda: stamp_nemo_gym_rollout_identity(
            [{}, {}],
            input_indices=[10, 11],
            rollouts_per_prompt=2,
            attempt_index=0,
        ),
        (ValueError,),
        "same input index",
    )


def _raw_result() -> dict[str, Any]:
    return {
        "response": {
            "output": [
                {
                    "type": "message",
                    "prompt_token_ids": [11, 12],
                    "generation_token_ids": [13],
                    "generation_log_probs": [-0.125],
                },
                {
                    "type": "message",
                    "prompt_token_ids": [11, 12, 13, 14, 15],
                    "generation_token_ids": [16, 17],
                    "generation_log_probs": [-0.25, -0.5],
                },
            ]
        },
        "responses_create_params": {"input": []},
    }


def _postprocess(retain_token_audit: bool) -> dict[str, Any]:
    mock_self = type(
        "_MockSelf",
        (),
        {"cfg": {"retain_token_audit": retain_token_audit}},
    )()
    return (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            mock_self,
            {},
            _raw_result(),
            _Tokenizer(),
        )
    )


def _validate_token_audit() -> None:
    expected_audit = {
        "version": 1,
        "turns": [
            {
                "output_item_index": 0,
                "prompt_token_ids": [11, 12],
                "generation_token_ids": [13],
                "generation_logprobs": [-0.125],
            },
            {
                "output_item_index": 1,
                "prompt_token_ids": [11, 12, 13, 14, 15],
                "generation_token_ids": [16, 17],
                "generation_logprobs": [-0.25, -0.5],
            },
        ],
    }

    disabled = _postprocess(False)
    enabled = _postprocess(True)
    assert "_nemo_rl_token_audit" not in disabled["full_result"]
    assert enabled["full_result"]["_nemo_rl_token_audit"] == expected_audit
    assert (
        json.loads(json.dumps(enabled["full_result"]))["_nemo_rl_token_audit"]
        == expected_audit
    )

    expected_message_ids = [[11, 12], [13], [14, 15], [16, 17]]
    for result in (disabled, enabled):
        assert [
            message["token_ids"].tolist() for message in result["message_log"]
        ] == expected_message_ids
        assert result["message_log"][1]["generation_logprobs"].tolist() == [-0.125]
        assert result["message_log"][3]["generation_logprobs"].tolist() == [
            -0.25,
            -0.5,
        ]


def main() -> None:
    _validate_identity()
    _validate_token_audit()
    print(
        json.dumps(
            {
                "canonical_rollout_identity": True,
                "raw_token_audit_disabled_absent": True,
                "raw_token_audit_enabled_exact": True,
                "status": "ok",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
