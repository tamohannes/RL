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
"""Dependency-free parsing and reward-contract helpers."""

from __future__ import annotations

import json
from typing import Any


def parse_candidate(text: str, answer_keys: list[str]) -> tuple[str, Any, bool]:
    """Parse the literal final response without accepting wrappers or extra keys."""
    try:
        answer = json.loads(text)
    except json.JSONDecodeError:
        return "invalid_json", text, False
    if not isinstance(answer, dict):
        return "not_object", answer, False
    exact_key_set = frozenset(answer) == frozenset(answer_keys)
    return "ok", answer, exact_key_set


def passes_contract(
    parse_status: str,
    exact_key_set: bool,
    check_results: list[list[Any]],
) -> bool:
    """Combine the strict output contract with the exact source checker."""
    return (
        parse_status == "ok"
        and exact_key_set
        and bool(check_results)
        and all(passed for _, passed in check_results)
    )
