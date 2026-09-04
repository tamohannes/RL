from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

BUILD_SCRIPT = Path("examples/build_nemo_rl_sciprobe_signal_sqsh.slurm")
PREFETCH_CONFIG = Path("examples/nemo_gym/prefetch_sciprobe_signal.yaml")


def test_signal_image_prefetches_exact_gym_servers_without_starting_them() -> None:
    source = BUILD_SCRIPT.read_text(encoding="utf-8")
    config = OmegaConf.to_container(OmegaConf.load(PREFETCH_CONFIG), resolve=True)
    assert isinstance(config, dict)
    config_paths = config["env"]["nemo_gym"]["config_paths"]

    assert config_paths == [
        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
        "responses_api_agents/sciprobe_simple_agent/configs/sciprobe_simple_agent.yaml",
        "resources_servers/sciprobe_checks/configs/sciprobe_checks.yaml",
        "resources_servers/sciprobe_ns_tools/configs/sciprobe_ns_tools.yaml",
    ]
    for config_path in config_paths:
        assert source.count(f"--config {config_path}") == 1

    assert '"${gym_cli}" env prefetch' in source
    assert '--search-dir "${NEMO_GYM_EXTRA_ROOTS}"' in source
    assert '+uv_venv_dir="${NEMO_GYM_VENV_DIR}"' in source
    assert "+skip_venv_if_present=true" in source
    for config_path in config_paths:
        server_dir = "/".join(config_path.split("/")[:2])
        assert source.count(server_dir) == 2
    assert 'test -x "${NEMO_GYM_VENV_DIR}/${server_venv}/.venv/bin/python"' in source
    assert "examples/nemo_gym/prefetch_venvs.py" not in source
    assert "NemoGym.options" not in source
    assert "ray start" not in source
