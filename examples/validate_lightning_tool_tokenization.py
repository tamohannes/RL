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

"""Fail-fast checks for Lightning's native multi-turn tool tokenization."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

PYTHON_TOOL_FLAT = {
    "type": "function",
    "name": "stateful_python_code_exec",
    "description": (
        "Execute Python code in a stateful Jupyter notebook environment. "
        "State is preserved across calls in this episode."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute.",
            }
        },
        "required": ["code"],
    },
    "strict": True,
}


def _nested_tool(flat_tool: dict[str, Any]) -> dict[str, Any]:
    function = {key: value for key, value in flat_tool.items() if key != "type"}
    return {"type": "function", "function": function}


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def _render(
    tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> tuple[str, list[int]]:
    kwargs = {
        "tools": tools,
        "add_generation_prompt": True,
        "enable_thinking": True,
        "truncate_history_thinking": False,
    }
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    direct_ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if isinstance(direct_ids, Mapping):
        direct_ids = direct_ids["input_ids"]
    if hasattr(direct_ids, "tolist"):
        direct_ids = direct_ids.tolist()
    if direct_ids and isinstance(direct_ids[0], list):
        direct_ids = direct_ids[0]
    direct_ids = list(direct_ids)
    encoded_ids = _token_ids(tokenizer, rendered)
    assert direct_ids == encoded_ids, (
        "apply_chat_template(tokenize=True) disagrees with encoding the rendered "
        "template without extra special tokens"
    )
    return rendered, direct_ids


def _assert_prefix(prefix: list[int], full: list[int], label: str) -> None:
    assert full[: len(prefix)] == prefix, f"non-contiguous token history at {label}"


def main() -> None:
    register_omegaconf_resolvers()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--config",
        default=(
            "examples/configs/recipes/llm/"
            "grpo-nemotron3.5-lightning-30ba3b-1n4g-automodel-"
            "sciprobe-tool-canary.yaml"
        ),
    )
    args = parser.parse_args()

    resolved_config = OmegaConf.to_container(load_config(args.config), resolve=True)
    master_config = MasterConfig(**resolved_config)
    assert master_config.logger["wandb_enabled"] is True
    assert master_config.logger["wandb"]["mode"] == "offline"
    assert master_config.logger["wandb"]["log_nemo_gym_full_result_tables"] is True
    data_path = Path(master_config.data["train"]["data_path"])
    rows = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1, f"expected one canary row in {data_path}"
    responses_create_params = rows[0]["responses_create_params"]
    initial_messages = responses_create_params["input"]
    tools = responses_create_params["tools"]
    assert tools == [PYTHON_TOOL_FLAT], "canary must use the native flat tool schema"
    assert (
        responses_create_params["max_output_tokens"]
        <= (master_config.policy["generation"]["max_new_tokens"])
    )

    model_path = Path(args.model).resolve()
    template_path = model_path / "chat_template.jinja"
    assert template_path.is_file(), f"missing native template: {template_path}"
    template_text = template_path.read_text(encoding="utf-8")
    template_sha256 = hashlib.sha256(template_text.encode("utf-8")).hexdigest()

    tokenizer = get_tokenizer(
        {
            "name": str(model_path),
            "chat_template": "default",
            "chat_template_kwargs": {
                "enable_thinking": True,
                "truncate_history_thinking": False,
            },
        }
    )
    assert isinstance(tokenizer.chat_template, str) and tokenizer.chat_template
    assert tokenizer.chat_template == template_text, (
        "the tokenizer did not load the model's standalone chat_template.jinja"
    )
    assert tokenizer.bos_token == "<s>"
    assert tokenizer.eos_token == "<|im_end|>"
    assert tokenizer.pad_token == "<|im_end|>"

    nested_tool = _nested_tool(tools[0])
    prompt0, ids0 = _render(tokenizer, initial_messages, tools)
    prompt0_nested, ids0_nested = _render(tokenizer, initial_messages, [nested_tool])
    assert prompt0 == prompt0_nested and ids0 == ids0_nested, (
        "flat Responses API and nested Chat Completions tool schemas render differently"
    )
    assert "<function>\n<name>stateful_python_code_exec</name>" in prompt0
    assert prompt0.endswith("<|im_start|>assistant\n<think>\n")

    assistant_call1 = {
        "role": "assistant",
        "reasoning_content": "I will establish the state first.",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "stateful_python_code_exec",
                    "arguments": {"code": 'carry = 17\n"stored"'},
                },
            }
        ],
    }
    after_call1 = initial_messages + [assistant_call1]
    rendered_call1 = tokenizer.apply_chat_template(
        after_call1,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
        truncate_history_thinking=False,
    )
    ids_call1 = _token_ids(tokenizer, rendered_call1)
    assert rendered_call1.startswith(prompt0)
    _assert_prefix(ids0, ids_call1, "first assistant tool call")

    after_result1 = after_call1 + [{"role": "tool", "content": "stored"}]
    prompt1, ids1 = _render(tokenizer, after_result1, tools)
    _assert_prefix(ids_call1, ids1, "first tool result")

    assistant_call2 = {
        "role": "assistant",
        "reasoning_content": "The second call must reuse the prior Python state.",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "stateful_python_code_exec",
                    "arguments": {"code": "carry * 3 + 4"},
                },
            }
        ],
    }
    after_call2 = after_result1 + [assistant_call2]
    rendered_call2 = tokenizer.apply_chat_template(
        after_call2,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
        truncate_history_thinking=False,
    )
    ids_call2 = _token_ids(tokenizer, rendered_call2)
    assert rendered_call2.startswith(prompt1)
    _assert_prefix(ids1, ids_call2, "second assistant tool call")

    after_result2 = after_call2 + [{"role": "tool", "content": "55"}]
    prompt2, ids2 = _render(tokenizer, after_result2, tools)
    _assert_prefix(ids_call2, ids2, "second tool result")

    print(
        json.dumps(
            {
                "status": "ok",
                "model": str(model_path),
                "config": args.config,
                "data_path": str(data_path),
                "template_sha256": template_sha256,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "prompt_token_counts": [len(ids0), len(ids1), len(ids2)],
                "tool_schema_forms_match": True,
                "multi_turn_prefix_contiguous": True,
                "offline_full_result_capture": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
