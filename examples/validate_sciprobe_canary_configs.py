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

"""Resolve and validate the SciProbe Lightning canary configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, open_dict

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

CONFIG_ROOT = Path("examples/configs/recipes/llm")
PLAIN_CONFIG = CONFIG_ROOT / (
    "grpo-nemotron3.5-lightning-30ba3b-1n4g-automodel-sciprobe-canary.yaml"
)
TOOL_CONFIG = CONFIG_ROOT / (
    "grpo-nemotron3.5-lightning-30ba3b-1n4g-automodel-sciprobe-tool-canary.yaml"
)
SIGNAL_CONFIG = CONFIG_ROOT / (
    "grpo-nemotron3.5-lightning-30ba3b-1n4g-automodel-sciprobe-signal-canary.yaml"
)
SIGNAL_PROBE_ID = "q3:c013:d0"
SIGNAL_GOLD_SHA256 = "45f51cc52d4093ee60d941fc093653b0497f9b076b0de5b6b8175a0f945df36c"
SIGNAL_DATA_TREE_SHA256 = (
    "16713f67f959a4c276baea508c1fb64fa54bf622f4e14b0b4def77d6c152a590"
)
SIGNAL_ANSWER_KEYS = {
    "n_samples",
    "total_aligned_reads",
    "total_modified_reads",
    "mean_modified_pct",
    "max_modified_pct",
    "max_modified_sample",
}
SIGNAL_EXPECTED_ANSWER = {
    "n_samples": 4,
    "total_aligned_reads": 111766,
    "total_modified_reads": 69429,
    "mean_modified_pct": 62.19,
    "max_modified_pct": 67.75,
    "max_modified_sample": "S4",
}


def _resolve_gym_config(config: MasterConfig) -> dict[str, Any]:
    """Resolve Gym component configs exactly as the runtime does, without starting servers."""
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


def _load(path: Path) -> MasterConfig:
    resolved = OmegaConf.to_container(load_config(str(path)), resolve=True)
    assert isinstance(resolved, dict), f"{path}: resolved config is not a mapping"
    return MasterConfig(**resolved)


def _validate_common(
    config: MasterConfig,
    path: Path,
    *,
    expected_generations: int = 4,
    expected_global_batch_size: int = 4,
) -> dict[str, Any]:
    assert config.cluster["num_nodes"] == 1, f"{path}: expected one node"
    assert config.cluster["gpus_per_node"] == 4, f"{path}: expected four GPUs"
    assert config.grpo.num_prompts_per_step == 1
    assert config.grpo.num_generations_per_prompt == expected_generations
    assert config.grpo.max_num_steps == 1
    assert config.policy["train_global_batch_size"] == expected_global_batch_size
    assert config.policy["train_micro_batch_size"] == 1
    assert config.policy["dtensor_cfg"]["expert_parallel_size"] == 4
    assert (
        config.policy["dtensor_cfg"]["env_vars"]["PYTORCH_CUDA_ALLOC_CONF"]
        == "expandable_segments:False"
    )
    assert config.policy["make_sequence_length_divisible_by"] == 64
    assert config.policy["generation"]["vllm_cfg"]["tensor_parallel_size"] == 4
    assert (
        config.policy["dtensor_cfg"]["automodel_kwargs"]["num_nextn_predict_layers"]
        == 0
    )
    assert config.loss_fn.force_on_policy_ratio is True
    assert config.loss_fn.use_importance_sampling_correction is False
    assert config.checkpointing["enabled"] is True
    assert config.checkpointing["save_period"] == 1

    data_path = Path(config.data["train"]["data_path"])
    assert data_path.is_file(), f"{path}: missing train data {data_path}"
    return {
        "config": str(path),
        "dataset": config.data["train"]["dataset_name"],
        "data_path": str(data_path),
        "max_total_sequence_length": config.policy["max_total_sequence_length"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-config", type=Path, default=SIGNAL_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    signal_config = args.signal_config
    register_omegaconf_resolvers()
    plain = _load(PLAIN_CONFIG)
    plain_summary = _validate_common(plain, PLAIN_CONFIG)
    assert plain.data["train"]["dataset_name"] == "ResponseDataset"
    assert plain.data["default"]["env_name"] == "math"
    assert plain.policy["max_total_sequence_length"] == 2048
    assert plain.logger["wandb_enabled"] is False

    tool = _load(TOOL_CONFIG)
    tool_summary = _validate_common(tool, TOOL_CONFIG)
    assert tool.data["train"]["dataset_name"] == "NemoGymDataset"
    assert tool.data["default"]["env_name"] == "nemo_gym"
    assert tool.data["validation"] is None
    assert tool.data["use_multiple_dataloader"] is False
    assert tool.policy["max_total_sequence_length"] == 4096
    assert tool.policy["max_total_sequence_length"] % 64 == 0
    assert tool.policy.get("is_vlm") in (None, False)
    assert tool.policy["generation"] is not None
    assert tool.policy["generation"]["backend"] == "vllm"
    assert tool.policy["generation"]["vllm_cfg"]["async_engine"] is True
    assert tool.policy["generation"]["vllm_cfg"]["expose_http_server"] is True
    assert tool.policy["generation"]["stop_strings"] is None
    assert tool.policy["generation"]["stop_token_ids"] is None
    assert tool.policy["generation"]["max_new_tokens"] == 2048
    assert tool.policy["generation"]["max_new_tokens"] % 64 == 0
    assert tool.grpo.max_val_samples is None
    assert tool.grpo.async_grpo.enabled is False
    assert tool.grpo.use_dynamic_sampling is False
    assert tool.grpo.reward_scaling.enabled is False
    assert tool.grpo.reward_shaping.enabled is False
    assert tool.env["should_use_nemo_gym"] is True
    assert "nemo_gym" in tool.env
    assert tool.env["nemo_gym"].get("is_trajectory_collection", False) is False
    assert tool.env["should_log_nemo_gym_responses"] is False
    assert tool.logger["wandb_enabled"] is True
    assert tool.logger["wandb"]["mode"] == "offline"
    assert tool.logger["wandb"]["log_nemo_gym_full_result_tables"] is True

    tool_rows = [
        json.loads(line)
        for line in Path(tool.data["train"]["data_path"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(tool_rows) == 1
    assert tool_rows[0]["responses_create_params"]["max_output_tokens"] == 2048

    signal = _load(signal_config)
    signal_summary = _validate_common(
        signal,
        signal_config,
        expected_generations=8,
        expected_global_batch_size=8,
    )
    assert signal.data["train"]["dataset_name"] == "NemoGymDataset"
    assert signal.data["default"]["env_name"] == "nemo_gym"
    assert signal.data["validation"] is None
    assert signal.data["use_multiple_dataloader"] is False
    assert signal.policy["max_total_sequence_length"] == 12288
    assert signal.policy["max_total_sequence_length"] % 64 == 0
    assert signal.policy["generation"]["max_new_tokens"] == 8192
    assert signal.policy["generation"]["temperature"] == 0.7
    assert signal.policy["generation"]["top_p"] == 1.0
    assert signal.policy["generation"]["top_k"] is None
    assert (
        signal.policy["generation"]["vllm_cfg"]["http_generation_api_key_env_var"]
        == "SCIPROBE_POLICY_GENERATION_TOKEN"
    )
    assert signal.grpo.max_val_samples is None
    assert signal.env["should_use_nemo_gym"] is True
    assert signal.env["should_log_nemo_gym_responses"] is False
    assert signal.logger["wandb_enabled"] is True
    assert signal.logger["tensorboard_enabled"] is True
    assert signal.logger["wandb"]["mode"] == "offline"
    assert signal.logger["wandb"]["log_nemo_gym_full_result_tables"] is True
    assert signal.checkpointing["checkpoint_dir"].endswith("signal-canary-r7")
    assert signal.logger["log_dir"].startswith(
        signal.checkpointing["checkpoint_dir"] + "/"
    )

    raw_gym = signal.env["nemo_gym"]
    assert raw_gym["config_paths"] == [
        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
        "responses_api_agents/sciprobe_simple_agent/configs/sciprobe_simple_agent.yaml",
        "resources_servers/sciprobe_checks/configs/sciprobe_checks.yaml",
        "resources_servers/sciprobe_ns_tools/configs/sciprobe_ns_tools.yaml",
    ]
    policy_model = raw_gym["policy_model"]["responses_api_models"]["vllm_model"]
    assert policy_model["api_key"] is None
    assert policy_model["api_key_env_var"] == "SCIPROBE_POLICY_GENERATION_TOKEN"
    assert (
        policy_model["trusted_ingress_token_env_var"]
        == "SCIPROBE_TRUSTED_INGRESS_TOKEN"
    )
    assert policy_model["trusted_ingress_header_name"] == ("X-SciProbe-Trusted-Ingress")
    gym = _resolve_gym_config(signal)
    ns_tools_implementations = gym["ns_tools"]["resources_servers"]
    assert set(ns_tools_implementations) == {"sciprobe_ns_tools"}
    ns_tools = ns_tools_implementations["sciprobe_ns_tools"]
    assert ns_tools["default_verifier"] == "sciprobe_checks"
    assert ns_tools["verifier_auth_token_env_var"] == "SCIPROBE_VERIFIER_TOKEN"
    assert ns_tools["verifier_auth_header_name"] == "X-SciProbe-Verifier-Token"
    assert ns_tools["capability_signing_key_env_var"] == (
        "SCIPROBE_CAPABILITY_SIGNING_KEY"
    )
    assert ns_tools["capability_header_name"] == ("X-SciProbe-Rollout-Capability")
    assert ns_tools["capability_store_path_env_var"] == "SCIPROBE_CAPABILITY_STORE_PATH"
    assert ns_tools["num_workers"] == 1
    assert ns_tools["verifiers"] == {
        "sciprobe_checks": {
            "type": "resources_servers",
            "name": "sciprobe_checks",
        }
    }
    assert (
        ns_tools["nemo_skills_tool_overrides"]["DirectPythonTool"]["exec_timeout_s"]
        == 30
    )

    agent = gym["ns_tools_simple_agent"]["responses_api_agents"][
        "sciprobe_simple_agent"
    ]
    assert agent["capability_signing_key_env_var"] == (
        "SCIPROBE_CAPABILITY_SIGNING_KEY"
    )
    assert agent["capability_header_name"] == ("X-SciProbe-Rollout-Capability")
    assert agent["capability_ttl_seconds"] == 300
    assert agent["trusted_ingress_token_env_var"] == "SCIPROBE_TRUSTED_INGRESS_TOKEN"
    assert agent["trusted_ingress_header_name"] == ("X-SciProbe-Trusted-Ingress")

    signal_rows = [
        json.loads(line)
        for line in Path(signal.data["train"]["data_path"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(signal_rows) == 1
    signal_row = signal_rows[0]
    assert signal_row["id"] == SIGNAL_PROBE_ID
    assert signal_row["probe_id"] == SIGNAL_PROBE_ID
    assert signal_row["verifier_type"] == "sciprobe_checks"
    assert "expected_answer" not in signal_row
    assert "_sciprobe_verifier_capability" not in signal_row
    assert signal_row["responses_create_params"]["max_output_tokens"] == 8192
    assert len(signal_row["responses_create_params"]["tools"]) == 1
    assert (
        signal_row["responses_create_params"]["tools"][0]["name"]
        == "stateful_python_code_exec"
    )
    system_prompt = signal_row["responses_create_params"]["input"][0]["content"]
    assert 'import os; os.chdir("/workspace/sciprobe-probe")' in system_prompt
    task_prompt = signal_row["responses_create_params"]["input"][1]["content"]
    answer_template = json.loads(task_prompt[task_prompt.rfind("{") :])
    assert set(answer_template) == SIGNAL_ANSWER_KEYS

    serialized_signal_row = json.dumps(signal_row, sort_keys=True)
    for forbidden in (
        "111766",
        "69429",
        "62.19",
        "67.75",
        '"S4"',
        "gold.json",
        "checks.py",
        "reference.py",
        "wrong_reference.py",
        "meta.json",
        "/workspace/sciprobe-private",
    ):
        assert forbidden not in serialized_signal_row

    verifier_config = OmegaConf.to_container(
        OmegaConf.load(
            "examples/nemo_gym_extensions/resources_servers/sciprobe_checks/"
            "configs/sciprobe_checks.yaml"
        ),
        resolve=True,
    )
    assert isinstance(verifier_config, dict)
    probe_definition = verifier_config["sciprobe_checks"]["resources_servers"][
        "sciprobe_checks"
    ]["probes"][SIGNAL_PROBE_ID]
    verifier_server = verifier_config["sciprobe_checks"]["resources_servers"][
        "sciprobe_checks"
    ]
    assert verifier_server["auth_token_env_var"] == "SCIPROBE_VERIFIER_TOKEN"
    assert verifier_server["auth_header_name"] == "X-SciProbe-Verifier-Token"
    assert probe_definition["expected_answer"] == SIGNAL_EXPECTED_ANSWER
    assert probe_definition["gold_sha256"] == SIGNAL_GOLD_SHA256
    assert probe_definition["data_tree_sha256"] == SIGNAL_DATA_TREE_SHA256
    assert set(probe_definition["answer_keys"]) == SIGNAL_ANSWER_KEYS

    print(
        json.dumps(
            {
                "status": "ok",
                "plain": plain_summary,
                "tool": tool_summary,
                "signal": signal_summary,
                "on_policy_ratio": True,
                "checkpoint_each_step": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
