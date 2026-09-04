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

"""Pure checks for the fixed stateful-path, binary-choice overfit task."""

from __future__ import annotations

import ast
import json
from typing import Any


def _field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _output_items(response: Any) -> list[Any]:
    output = _field(response, "output")
    return output if isinstance(output, list) else []


def _arguments(item: Any) -> dict[str, Any] | None:
    arguments = _field(item, "arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return arguments if isinstance(arguments, dict) else None


def _parse_module(code: Any) -> ast.Module | None:
    """Parse model-authored Python without executing it."""
    if not isinstance(code, str):
        return None
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _is_state_name(node: ast.AST, state_name: str) -> bool:
    return (
        isinstance(node, ast.Name)
        and node.id == state_name
        and isinstance(node.ctx, ast.Load)
    )


def _is_harmless_display(node: ast.AST, state_name: str) -> bool:
    """Allow literal/name display expressions and ``print`` of those values."""
    if isinstance(node, ast.Constant) or _is_state_name(node, state_name):
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
        and not node.keywords
        and all(
            isinstance(argument, ast.Constant) or _is_state_name(argument, state_name)
            for argument in node.args
        )
    )


def _initializes_state(code: Any, state_name: str, state_value: int) -> bool:
    module = _parse_module(code)
    if module is None:
        return False
    assignments = 0
    for statement in module.body:
        if isinstance(statement, ast.Assign):
            if (
                len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
                or statement.targets[0].id != state_name
                or not isinstance(statement.value, ast.Constant)
                or isinstance(statement.value.value, bool)
                or statement.value.value != state_value
            ):
                return False
            assignments += 1
        elif isinstance(statement, ast.Expr):
            if not _is_harmless_display(statement.value, state_name):
                return False
        else:
            return False
    return assignments == 1


def _evaluate_numeric_expression(
    node: ast.AST, state_name: str, state_value: int
) -> tuple[int | float, bool] | None:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value, False
    if _is_state_name(node, state_name):
        return state_value, True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _evaluate_numeric_expression(node.operand, state_name, state_value)
        if operand is None:
            return None
        value, reads_state = operand
        return (+value if isinstance(node.op, ast.UAdd) else -value), reads_state
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
    ):
        left = _evaluate_numeric_expression(node.left, state_name, state_value)
        right = _evaluate_numeric_expression(node.right, state_name, state_value)
        if left is None or right is None:
            return None
        left_value, left_reads_state = left
        right_value, right_reads_state = right
        try:
            if isinstance(node.op, ast.Add):
                value = left_value + right_value
            elif isinstance(node.op, ast.Sub):
                value = left_value - right_value
            elif isinstance(node.op, ast.Mult):
                value = left_value * right_value
            elif isinstance(node.op, ast.Div):
                value = left_value / right_value
            elif isinstance(node.op, ast.FloorDiv):
                value = left_value // right_value
            else:
                value = left_value % right_value
        except (ArithmeticError, OverflowError):
            return None
        return value, left_reads_state or right_reads_state
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
        and len(node.args) == 1
        and not node.keywords
    ):
        return _evaluate_numeric_expression(node.args[0], state_name, state_value)
    return None


def _reads_state_and_produces(
    code: Any,
    state_name: str,
    state_value: int,
    expected_output: str,
) -> bool:
    module = _parse_module(code)
    if module is None or not module.body:
        return False
    if not all(isinstance(statement, ast.Expr) for statement in module.body):
        return False
    evaluated = _evaluate_numeric_expression(
        module.body[-1].value, state_name, state_value
    )
    if evaluated is None:
        return False
    value, reads_state = evaluated
    return reads_state and str(value) == expected_output


def _last_assistant_text(response: Any) -> str:
    for item in reversed(_output_items(response)):
        if _field(item, "type") != "message" or _field(item, "role") != "assistant":
            continue
        content = _field(item, "content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks = [
                _field(part, "text")
                for part in content
                if isinstance(_field(part, "text"), str)
            ]
            return "\n".join(chunks).strip()
    return ""


def score_stateful_choice_trace(
    response: Any,
    *,
    tool_name: str,
    state_name: str,
    state_value: int,
    expected_second_output: str,
    choices: list[str],
    rewarded_choice: str,
) -> tuple[float, str | None, list[list[Any]]]:
    """Score a fixed two-call trace plus one learnable final-token choice."""
    if len(choices) != 2 or len(set(choices)) != 2 or rewarded_choice not in choices:
        raise ValueError("choices must contain two distinct values including reward")
    if any(not isinstance(choice, str) or not choice for choice in choices):
        raise ValueError("choices must be non-empty strings")
    output = _output_items(response)
    calls = [
        (position, item)
        for position, item in enumerate(output)
        if _field(item, "type") == "function_call"
    ]
    call_outputs = [
        (position, item)
        for position, item in enumerate(output)
        if _field(item, "type") == "function_call_output"
    ]
    arguments = [_arguments(item) for _, item in calls]
    call_ids = [_field(item, "call_id") for _, item in calls]
    output_by_call_id = {
        _field(item, "call_id"): (position, item)
        for position, item in call_outputs
        if isinstance(_field(item, "call_id"), str)
    }

    exactly_two_calls = len(calls) == 2 and len(call_outputs) == 2
    tool_names_match = exactly_two_calls and all(
        _field(item, "name") == tool_name for _, item in calls
    )
    unique_call_ids = (
        exactly_two_calls
        and all(isinstance(call_id, str) and call_id for call_id in call_ids)
        and len(set(call_ids)) == 2
    )
    ordered_outputs = False
    if unique_call_ids and all(call_id in output_by_call_id for call_id in call_ids):
        ordered_outputs = (
            calls[0][0]
            < output_by_call_id[call_ids[0]][0]
            < calls[1][0]
            < output_by_call_id[call_ids[1]][0]
        )
    first_code_initializes_state = (
        len(arguments) == 2
        and arguments[0] is not None
        and _initializes_state(
            arguments[0].get("code"), state_name=state_name, state_value=state_value
        )
    )
    second_code_reads_state = (
        len(arguments) == 2
        and arguments[1] is not None
        and _reads_state_and_produces(
            arguments[1].get("code"),
            state_name=state_name,
            state_value=state_value,
            expected_output=expected_second_output,
        )
    )
    second_output_matches = False
    if len(call_ids) == 2 and call_ids[1] in output_by_call_id:
        second_output = _field(output_by_call_id[call_ids[1]][1], "output")
        second_output_matches = (
            isinstance(second_output, str)
            and second_output.strip() == expected_second_output
        )
    final_text = _last_assistant_text(response)
    selected_choice = final_text if final_text in choices else None
    checks: list[list[Any]] = [
        ["exactly_two_tool_calls_and_outputs", exactly_two_calls],
        ["tool_names_match", tool_names_match],
        ["call_ids_are_unique", unique_call_ids],
        ["call_outputs_are_ordered", ordered_outputs],
        ["first_call_initializes_fixed_state", first_code_initializes_state],
        ["second_call_reads_state_and_computes_result", second_code_reads_state],
        ["second_output_matches_expected_result", second_output_matches],
        ["final_answer_is_allowed_choice", selected_choice is not None],
    ]
    trace_valid = all(bool(passed) for _, passed in checks)
    reward = 1.0 if trace_valid and selected_choice == rewarded_choice else 0.0
    return reward, selected_choice, checks
