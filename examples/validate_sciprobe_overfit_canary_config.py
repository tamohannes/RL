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

"""Resolve and fail closed on the isolated Lightning overfit canary config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, open_dict

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

DEFAULT_CONFIG = Path("examples/configs/recipes/llm") / (
    "grpo-nemotron3.5-lightning-30ba3b-1n4g-automodel-sciprobe-overfit-canary.yaml"
)
PROBE_ID = "stateful-choice-overfit-v1"


def _resolve_gym_config(config: MasterConfig) -> dict[str, Any]:
    from nemo_gym.global_config import (
        GlobalConfigDictParser,
        GlobalConfigDictParserConfig,
    )

    gym = OmegaConf.create(config.env["nemo_gym"])
    extension_root = Path("examples/nemo_gym_extensions").resolve()
    with open_dict(gym):
        gym["config_paths"] = [
            str(extension_root / path) if (extension_root / path).is_file() else path
            for path in gym["config_paths"]
        ]
        gym["policy_model_name"] = "validation-model"
        gym["policy_api_key"] = "validation-key"
        gym["policy_base_url"] = "http://127.0.0.1:8000/v1"
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=gym,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    result = OmegaConf.to_container(resolved, resolve=True)
    assert isinstance(result, dict)
    return result


def main() -> None:
    register_omegaconf_resolvers()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    resolved = OmegaConf.to_container(load_config(str(args.config)), resolve=True)
    assert isinstance(resolved, dict)
    config = MasterConfig(**resolved)

    assert config.cluster["num_nodes"] == 1
    assert config.cluster["gpus_per_node"] == 4
    assert config.grpo.num_prompts_per_step == 1
    assert config.grpo.num_generations_per_prompt == 16
    assert config.grpo.max_num_epochs == 1
    assert config.grpo.max_num_steps == 6
    assert config.grpo.val_at_start is True
    assert config.grpo.val_at_end is True
    assert config.grpo.val_period == 1
    # run_grpo_nemo_gym.py requires max_val_samples to be null, then derives
    # max_val_samples=val_batch_size=len(validation_dataset)=1 at runtime.
    assert config.grpo.val_batch_size is None
    assert config.grpo.max_val_samples is None
    assert config.grpo.val_num_generations_per_prompt == 32
    assert config.grpo.stop_at_validation_metric is None
    assert config.grpo.stop_at_validation_threshold is None
    assert config.grpo.normalize_rewards is True
    assert config.grpo.use_leave_one_out_baseline is False
    assert config.grpo.use_dynamic_sampling is True
    assert config.grpo.async_grpo is not None
    assert config.grpo.async_grpo.enabled is False
    assert config.grpo.dynamic_sampling_max_gen_batches == 10
    assert config.grpo.batch_multiplier == 1
    assert config.grpo.skip_reference_policy_logprobs_calculation is True
    assert config.grpo.overlong_filtering is False
    assert config.grpo.reward_scaling.enabled is False
    assert config.grpo.reward_shaping.enabled is False
    assert config.grpo.seq_logprob_error_threshold == 2
    assert config.grpo.adv_estimator.name == "grpo"
    assert config.grpo.adv_estimator.normalize_rewards is True
    assert config.grpo.adv_estimator.use_leave_one_out_baseline is False

    assert config.loss_fn.reference_policy_kl_penalty == 0.0
    assert config.loss_fn.use_importance_sampling_correction is False
    assert config.loss_fn.force_on_policy_ratio is False
    assert config.checkpointing["enabled"] is True
    assert config.checkpointing["metric_name"] is None
    assert config.checkpointing["save_period"] == 6
    assert config.checkpointing["save_optimizer"] is False
    assert config.checkpointing["model_save_format"] == "safetensors"

    assert config.policy["train_global_batch_size"] == 16
    assert config.policy["train_micro_batch_size"] == 1
    assert config.policy["logprob_batch_size"] == 1
    assert config.policy["max_total_sequence_length"] == 4096
    assert config.policy["make_sequence_length_divisible_by"] == 64
    assert config.policy["sequence_packing"]["enabled"] is False
    assert config.policy["dynamic_batching"]["enabled"] is False
    assert config.policy["optimizer"]["kwargs"]["lr"] == 5.0e-6
    generation = config.policy["generation"]
    assert generation["backend"] == "vllm"
    assert generation["max_new_tokens"] == 512
    assert 0 < generation["max_new_tokens"] <= 1024
    assert generation["temperature"] == 1.3
    assert generation["top_p"] == 1.0
    assert generation["top_k"] is None
    assert generation["val_temperature"] == 1.3
    assert generation["val_top_p"] == 1.0
    assert generation["val_top_k"] is None
    assert generation["vllm_cfg"]["async_engine"] is True
    assert generation["vllm_cfg"]["expose_http_server"] is True

    train = config.data["train"]
    validation = config.data["validation"]
    assert config.data["default"]["processor"] == "nemo_gym_data_processor"
    assert config.data["default"]["env_name"] == "nemo_gym"
    assert config.data["use_multiple_dataloader"] is False
    assert config.data["num_workers"] == 1
    assert train["dataset_name"] == "NemoGymDataset"
    assert validation["dataset_name"] == "NemoGymDataset"
    assert train["repeat"] == 80
    assert train["repeat"] >= config.grpo.max_num_steps * (
        config.grpo.dynamic_sampling_max_gen_batches + 1
    )
    assert train["data_path"] == validation["data_path"]
    data_path = Path(train["data_path"])
    rows = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == row["probe_id"] == PROBE_ID
    assert row["verifier_type"] == "sciprobe_overfit_checks"
    params = row["responses_create_params"]
    assert params["max_output_tokens"] == 512
    assert len(params["tools"]) == 1
    assert params["tools"][0]["name"] == "stateful_python_code_exec"
    assert params["tools"][0]["type"] == "function"
    assert params["tools"][0]["strict"] is True
    assert len(params["input"]) == 2
    serialized_row = json.dumps(row, sort_keys=True)
    for forbidden in (
        "rewarded_choice",
        "Only B",
        "only B",
        "sciprobe-private",
        "expected_answer",
        "q3:c013:d0",
    ):
        assert forbidden not in serialized_row

    raw_gym = config.env["nemo_gym"]
    assert raw_gym["loopback_only"] is True
    assert raw_gym["retain_token_audit"] is True
    assert raw_gym["config_paths"] == [
        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
        "responses_api_agents/sciprobe_simple_agent/configs/sciprobe_simple_agent.yaml",
        "resources_servers/sciprobe_overfit_checks/configs/sciprobe_overfit_checks.yaml",
        "resources_servers/sciprobe_ns_tools/configs/sciprobe_ns_tools.yaml",
    ]
    gym = _resolve_gym_config(config)
    agent = gym["ns_tools_simple_agent"]["responses_api_agents"][
        "sciprobe_simple_agent"
    ]
    assert agent["max_steps"] == 6
    ns_tools = gym["ns_tools"]["resources_servers"]["sciprobe_ns_tools"]
    assert ns_tools["default_verifier"] == "sciprobe_overfit_checks"
    assert ns_tools["verifiers"] == {
        "sciprobe_overfit_checks": {
            "type": "resources_servers",
            "name": "sciprobe_overfit_checks",
        }
    }
    assert ns_tools["num_workers"] == 1

    verifier_path = Path(
        "examples/nemo_gym_extensions/resources_servers/"
        "sciprobe_overfit_checks/configs/sciprobe_overfit_checks.yaml"
    )
    verifier = OmegaConf.to_container(OmegaConf.load(verifier_path), resolve=True)
    assert isinstance(verifier, dict)
    server = verifier["sciprobe_overfit_checks"]["resources_servers"][
        "sciprobe_overfit_checks"
    ]
    definition = server["probes"][PROBE_ID]
    assert server["auth_token_env_var"] == "SCIPROBE_VERIFIER_TOKEN"
    assert server["auth_header_name"] == "X-SciProbe-Verifier-Token"
    assert definition == {
        "tool_name": "stateful_python_code_exec",
        "state_name": "carry",
        "state_value": 17,
        "expected_second_output": "55",
        "choices": ["A", "B"],
        "rewarded_choice": "B",
    }

    assert config.logger["wandb_enabled"] is True
    assert config.logger["tensorboard_enabled"] is True
    assert config.logger["wandb"]["mode"] == "offline"
    assert config.logger["wandb"]["log_nemo_gym_full_result_tables"] is True
    print(
        json.dumps(
            {
                "status": "ok",
                "config": str(args.config),
                "probe_id": PROBE_ID,
                "train_repeat": train["repeat"],
                "train_rollouts_per_step": config.grpo.num_generations_per_prompt,
                "validation_rollouts": config.grpo.val_num_generations_per_prompt,
                "rewarded_choice": definition["rewarded_choice"],
                "token_audit": raw_gym["retain_token_audit"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
