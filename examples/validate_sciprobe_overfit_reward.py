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

"""Exercise both reward classes and every stateful-trace gate locally."""

from __future__ import annotations

import copy
import json

from nemo_gym_extensions.resources_servers.sciprobe_overfit_checks.reward_contract import (
    score_stateful_choice_trace,
)

TOOL_NAME = "stateful_python_code_exec"
FIRST_CODE = 'carry = 17\n"stored"'
SECOND_CODE = "carry * 3 + 4"


def _response(choice: str) -> dict:
    return {
        "output": [
            {
                "type": "function_call",
                "name": TOOL_NAME,
                "call_id": "call-1",
                "arguments": json.dumps({"code": FIRST_CODE}),
            },
            {"type": "function_call_output", "call_id": "call-1", "output": "stored"},
            {
                "type": "function_call",
                "name": TOOL_NAME,
                "call_id": "call-2",
                "arguments": {"code": SECOND_CODE},
            },
            {"type": "function_call_output", "call_id": "call-2", "output": "55"},
            {"type": "message", "role": "assistant", "content": [{"text": choice}]},
        ]
    }


def _score(response: dict) -> tuple[float, str | None, list[list[object]]]:
    return score_stateful_choice_trace(
        response,
        tool_name=TOOL_NAME,
        state_name="carry",
        state_value=17,
        expected_second_output="55",
        choices=["A", "B"],
        rewarded_choice="B",
    )


def main() -> None:
    rewarded = _score(_response("B"))
    unrewarded = _score(_response("A"))
    assert rewarded[0] == 1.0 and rewarded[1] == "B"
    assert unrewarded[0] == 0.0 and unrewarded[1] == "A"
    assert all(passed for _, passed in rewarded[2])
    assert all(passed for _, passed in unrewarded[2])

    semantic_variant = _response("B")
    semantic_variant["output"][0]["arguments"] = json.dumps(
        {"code": 'carry=17\nprint("stored")'}
    )
    semantic_variant["output"][2]["arguments"] = {"code": "4 + 3 * carry"}
    semantic_reward = _score(semantic_variant)
    assert semantic_reward[0] == 1.0
    assert all(passed for _, passed in semantic_reward[2])

    invalid_cases = {}
    mutations = {
        "wrong_first_code": lambda value: value["output"][0].update(
            {"arguments": json.dumps({"code": "carry = 18\n'stored'"})}
        ),
        "wrong_second_code": lambda value: value["output"][2].update(
            {"arguments": {"code": "55"}}
        ),
        "wrong_second_output": lambda value: value["output"][3].update(
            {"output": "54"}
        ),
        "extra_final_text": lambda value: value["output"][4].update(
            {"content": [{"text": "A."}]}
        ),
        "duplicate_call_id": lambda value: value["output"][2].update(
            {"call_id": "call-1"}
        ),
    }
    for name, mutate in mutations.items():
        candidate = copy.deepcopy(_response("B"))
        mutate(candidate)
        reward, _, checks = _score(candidate)
        assert reward == 0.0 and not all(passed for _, passed in checks), name
        invalid_cases[name] = [check for check, passed in checks if not passed]
    print(
        json.dumps(
            {
                "status": "ok",
                "rewarded_choice": "B",
                "unrewarded_choice": "A",
                "invalid_cases": invalid_cases,
                "llm_judge": False,
                "verifier_executes_model_code": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
