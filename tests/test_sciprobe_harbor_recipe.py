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

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils import MasterConfig
from nemo_rl.algorithms.single_controller_utils.config import (
    validate_single_controller_config,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


RECIPE = Path(
    "examples/configs/recipes/llm/"
    "grpo-nemotron3.5-lightning-30ba3b-1n4g-automodel-"
    "sciprobe-harbor-canary.yaml"
)
LAUNCHER = Path("examples/launch_sciprobe_harbor_lightning_canary_aws_cmh.sh")


def test_harbor_recipe_is_an_eight_rollout_single_node_canary(monkeypatch) -> None:
    monkeypatch.setenv("SCIPROBE_HARBOR_BANK_DIR", "/workspace/sciprobe-bank")
    register_omegaconf_resolvers()
    config_dict = OmegaConf.to_container(load_config(RECIPE), resolve=True)
    assert isinstance(config_dict, dict)
    config = MasterConfig(**config_dict)
    validate_single_controller_config(config)

    assert config.grpo is not None
    assert config.grpo.async_grpo is None
    assert config.grpo.num_prompts_per_step == 1
    assert config.grpo.num_generations_per_prompt == 8
    assert config.grpo.max_num_steps == 1
    assert config.grpo.max_rollout_turns == 1
    assert config.policy["train_global_batch_size"] == 8
    assert config.cluster == {
        "gpus_per_node": 4,
        "num_nodes": 1,
        "master_port_range_low": 1400,
        "master_port_range_high": 1999,
        "segment_size": None,
    }

    assert config.data_plane["enabled"] is True
    assert config.checkpointing["save_data_plane"] is True

    failure = config.async_rl.rollout_failure
    assert failure.max_infra_attempts_per_prompt == 5
    assert failure.nemo_gym.max_row_attempts == 3
    assert failure.max_skipped_prompts == 0
    assert failure.max_consecutive_dropped_prompts == 0
    assert not hasattr(failure, "allow_drop")

    data = config.data
    assert data["train"]["data_path"] == "/workspace/sciprobe-bank/train.jsonl"
    assert data["train"]["dataset_name"] == "NemoGymDataset"
    assert data["validation"] is None
    assert data["default"]["processor"] == "nemo_gym_data_processor"
    assert data["default"]["env_name"] == "nemo_gym"
    assert "input_key" not in data["train"]

    env = config.env
    assert "math" not in env
    assert env["should_use_nemo_gym"] is True
    assert env["should_mask_flagged_samples"] is True
    gym = env["nemo_gym"]
    assert gym["skip_venv_if_present"] is True
    assert gym["config_paths"] == [
        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
        "responses_api_agents/harbor_agent/configs/harbor_agent_sciprobe.yaml",
    ]
    assert (
        gym["harbor_agent"]["responses_api_agents"]["harbor_agent"]["concurrency"] == 8
    )

    generation = config.policy["generation"]
    assert generation["colocated"] == {
        "enabled": False,
        "resources": {"gpus_per_node": 2, "num_nodes": 1},
    }
    assert generation["vllm_cfg"]["tensor_parallel_size"] == 2
    assert config.policy["dtensor_cfg"]["expert_parallel_size"] == 2
    assert (
        config.policy["dtensor_cfg"]["automodel_kwargs"]["num_nextn_predict_layers"]
        == 0
    )

    # SingleController reserves two GPUs for generation, leaving two training
    # ranks. Exercise the same even-shard guard used by policy presharding so a
    # future 3+1 edit cannot pass a schema-only config test.
    train_world_size = (
        config.cluster["gpus_per_node"]
        - generation["colocated"]["resources"]["gpus_per_node"]
    )
    dtensor = config.policy["dtensor_cfg"]
    model_parallel_size = (
        dtensor["tensor_parallel_size"] * dtensor["context_parallel_size"]
    )
    assert train_world_size % model_parallel_size == 0
    assert train_world_size % dtensor["expert_parallel_size"] == 0
    assert (
        generation["colocated"]["resources"]["gpus_per_node"]
        % generation["vllm_cfg"]["tensor_parallel_size"]
        == 0
    )
    data_parallel_size = train_world_size // model_parallel_size
    assert config.policy["train_global_batch_size"] % data_parallel_size == 0
    shards = BatchedDataDict(
        {"sample_id": torch.arange(config.policy["train_global_batch_size"])}
    ).shard_by_batch_size(shards=data_parallel_size)
    assert [len(shard["sample_id"]) for shard in shards] == [4, 4]


def test_harbor_launcher_dry_run_validates_and_mounts_bank(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    (bank / "tasks").mkdir(parents=True)
    (bank / "train.jsonl").write_text("", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    persistent = tmp_path / "persistent"
    persistent.mkdir()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    args_file = tmp_path / "sciprobe-args"
    sciprobe = bin_dir / "sciprobe"
    sciprobe.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" > "$SCIPROBE_ARGS_FILE"\n',
        encoding="utf-8",
    )
    sciprobe.chmod(sciprobe.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "SCIPROBE_ARGS_FILE": str(args_file),
            "RUN_ID": "test-r1",
            "BANK_DIR": str(bank),
            "MODEL_PATH": str(model),
            "CONTAINER": "example.invalid/nemo-rl:canary",
            "PERSISTENT_ROOT": str(persistent),
            "DRY_RUN": "true",
        }
    )
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert args_file.read_text(encoding="utf-8").strip() == (
        f"probes validate-harbor --bank {bank}"
    )
    assert f"{bank}:/workspace/sciprobe-bank:ro" in completed.stdout
    assert "export SCIPROBE_HARBOR_BANK_DIR=/workspace/sciprobe-bank" in (
        completed.stdout
    )
    assert "export SCIPROBE_HARBOR_TASKS_DIR=/workspace/sciprobe-bank/tasks" in (
        completed.stdout
    )
    assert "export SCIPROBE_SINGULARITY_CACHE_DIR=" in completed.stdout
    assert "export SCIPROBE_HARBOR_JOBS_DIR=" in completed.stdout
    assert (
        "/opt/gym_venvs/responses_api_agents/harbor_agent/.venv/bin/python -c"
        in completed.stdout
    )
    assert "HARBOR_AGENT_FAILURE_CLASS" in completed.stdout
    assert "SCIPROBE_VERIFIER_ISOLATION_PROFILE" in completed.stdout
    assert "harbor_agent_error" in completed.stdout
    assert "sciprobe-answer-json-v1" in completed.stdout
    assert "examples/run_grpo_single_controller.py" in completed.stdout
    assert "examples/nemo_gym/run_grpo_nemo_gym.py" not in completed.stdout
    assert "--nodes=1" in completed.stdout
    assert "--gres=gpu:4" in completed.stdout


def test_harbor_launcher_is_fresh_run_only() -> None:
    launch = LAUNCHER.read_text(encoding="utf-8")
    assert 'RUN_ID="${RUN_ID:?' in launch
    assert 'if [[ -e "${RUN_ROOT}" ]]' in launch
    for removed_legacy_path in (
        "SCIPROBE_CAPABILITY",
        "SCIPROBE_VERIFIER_TOKEN",
        "SANDBOX_CONTAINER",
        "materialize_sciprobe",
        "sciprobe_simple_agent",
        "sciprobe_ns_tools",
    ):
        assert removed_legacy_path not in launch
