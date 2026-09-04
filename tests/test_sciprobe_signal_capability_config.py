from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config

RECIPE = Path(
    "examples/configs/recipes/llm/"
    "grpo-nemotron3.5-lightning-30ba3b-1n4g-automodel-sciprobe-signal-canary.yaml"
)
DATASET = Path(
    "examples/data/sciprobe/crispresso-editing-frequency-signal-canary.jsonl"
)
AGENT_CONFIG = Path(
    "examples/nemo_gym_extensions/responses_api_agents/sciprobe_simple_agent/"
    "configs/sciprobe_simple_agent.yaml"
)
NS_TOOLS_CONFIG = Path(
    "examples/nemo_gym_extensions/resources_servers/sciprobe_ns_tools/"
    "configs/sciprobe_ns_tools.yaml"
)
CHECKS_CONFIG = Path(
    "examples/nemo_gym_extensions/resources_servers/sciprobe_checks/"
    "configs/sciprobe_checks.yaml"
)
CHECKS_APP = Path(
    "examples/nemo_gym_extensions/resources_servers/sciprobe_checks/app.py"
)
LAUNCH_SCRIPT = Path("examples/launch_sciprobe_lightning_signal_canary.sh")
PREFIX_PREFLIGHT = Path(
    "examples/launch_sciprobe_signal_command_prefix_preflight.sh"
)
CONFIG_PREFLIGHT = Path("examples/preflight_sciprobe_canary_configs.slurm")
ISOLATION_VALIDATOR = Path("examples/validate_sciprobe_signal_canary_isolation.py")
PREFETCH_CONFIG = Path("examples/nemo_gym/prefetch_sciprobe_signal.yaml")
EXTENSION_ROOT = Path("examples/nemo_gym_extensions")
COMPONENT_REQUIREMENTS = (
    EXTENSION_ROOT / "responses_api_agents/sciprobe_simple_agent/requirements.txt",
    EXTENSION_ROOT / "resources_servers/sciprobe_ns_tools/requirements.txt",
    EXTENSION_ROOT / "resources_servers/sciprobe_checks/requirements.txt",
)


def test_signal_recipe_uses_post_generation_capability_agent() -> None:
    recipe = OmegaConf.to_container(OmegaConf.load(RECIPE), resolve=True)
    assert isinstance(recipe, dict)
    gym = recipe["env"]["nemo_gym"]
    assert gym["loopback_only"] is True
    assert gym["config_paths"] == [
        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
        "responses_api_agents/sciprobe_simple_agent/configs/sciprobe_simple_agent.yaml",
        "resources_servers/sciprobe_checks/configs/sciprobe_checks.yaml",
        "resources_servers/sciprobe_ns_tools/configs/sciprobe_ns_tools.yaml",
    ]
    assert gym["ns_tools"]["resources_servers"]["_delete_key"] == "ns_tools"
    policy_model = gym["policy_model"]["responses_api_models"]["vllm_model"]
    assert policy_model["api_key"] is None
    assert policy_model["api_key_env_var"] == "SCIPROBE_POLICY_GENERATION_TOKEN"
    assert (
        policy_model["trusted_ingress_token_env_var"]
        == "SCIPROBE_TRUSTED_INGRESS_TOKEN"
    )
    assert (
        recipe["policy"]["generation"]["vllm_cfg"]["http_generation_api_key_env_var"]
        == "SCIPROBE_POLICY_GENERATION_TOKEN"
    )

    agent_config = OmegaConf.to_container(OmegaConf.load(AGENT_CONFIG), resolve=True)
    agent = agent_config["ns_tools_simple_agent"]["responses_api_agents"][
        "sciprobe_simple_agent"
    ]
    assert agent["capability_signing_key_env_var"] == (
        "SCIPROBE_CAPABILITY_SIGNING_KEY"
    )
    assert agent["capability_header_name"] == "X-SciProbe-Rollout-Capability"
    assert agent["trusted_ingress_token_env_var"] == "SCIPROBE_TRUSTED_INGRESS_TOKEN"
    assert agent["trusted_ingress_header_name"] == "X-SciProbe-Trusted-Ingress"

    ns_tools_config = OmegaConf.to_container(
        OmegaConf.load(NS_TOOLS_CONFIG), resolve=True
    )
    ns_tools = ns_tools_config["ns_tools"]["resources_servers"]["sciprobe_ns_tools"]
    assert ns_tools["num_workers"] == 1
    assert ns_tools["capability_signing_key_env_var"] == (
        "SCIPROBE_CAPABILITY_SIGNING_KEY"
    )
    assert ns_tools["capability_header_name"] == ("X-SciProbe-Rollout-Capability")
    assert ns_tools["capability_store_path_env_var"] == "SCIPROBE_CAPABILITY_STORE_PATH"
    assert ns_tools["verifier_auth_token_env_var"] == "SCIPROBE_VERIFIER_TOKEN"

    prefetch = OmegaConf.to_container(OmegaConf.load(PREFETCH_CONFIG), resolve=True)
    assert isinstance(prefetch, dict)
    prefetch_gym = prefetch["env"]["nemo_gym"]
    assert prefetch_gym["config_paths"] == gym["config_paths"]
    assert prefetch_gym["ns_tools"]["resources_servers"]["_delete_key"] == "ns_tools"
    assert "sciprobe_ns_tools" in prefetch_gym["ns_tools"]["resources_servers"]


def test_signal_launch_enables_sandbox_network_block_and_fresh_secrets() -> None:
    launch = LAUNCH_SCRIPT.read_text(encoding="utf-8")
    assert (
        'SANDBOX_CONTAINER="${SANDBOX_CONTAINER:?set SANDBOX_CONTAINER '
        'to a pinned sandbox image}"' in launch
    )
    assert "nemo-skills-sandbox-latest.sqsh" not in launch
    assert (
        'SANDBOX_ENV_VARS="NEMO_SKILLS_SANDBOX_BLOCK_NETWORK=1,'
        "SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK=1,"
        "PYTHONPATH=/workspace/sciprobe-seccomp-hook,"
        'NUM_WORKERS=1,SANDBOX_FORCE_SINGLE_NODE=1"' in launch
    )
    sandbox_mount_line = next(
        line
        for line in launch.splitlines()
        if line.startswith("export SANDBOX_EXTRA_MOUNTS=")
    )
    main_mount_line = next(
        line for line in launch.splitlines() if line.startswith("export MOUNTS=")
    )
    assert (
        "examples/sandbox_seccomp_hook:/workspace/sciprobe-seccomp-hook:ro"
        in sandbox_mount_line
    )
    assert "sandbox_seccomp_hook" not in main_mount_line
    assert "validate_sciprobe_sandbox_seccomp.py" in launch
    assert (
        "examples/validate_sciprobe_sandbox_seccomp.py:"
        "/workspace/validate-sciprobe-sandbox-seccomp.py:ro" in sandbox_mount_line
    )
    assert (
        "uv run --locked --no-sync python "
        "examples/validate_sciprobe_sandbox_seccomp.py" not in launch
    )
    assert "examples/sandbox_seccomp_hook/sitecustomize.py" in launch
    assert (
        'SANDBOX_COMMAND="${SANDBOX_COMMAND:-unshare --pid --fork '
        "--mount-proc --kill-child "
        '/workspace/start-sciprobe-loopback-sandbox.sh}"' in launch
    )
    assert (
        "examples/start_sciprobe_loopback_sandbox.sh:"
        "/workspace/start-sciprobe-loopback-sandbox.sh:ro" in sandbox_mount_line
    )
    assert "--exclusive" in launch
    assert 'SCIPROBE_TRUSTED_INGRESS_TOKEN="$(openssl rand -hex 32)"' in launch
    assert 'SCIPROBE_POLICY_GENERATION_TOKEN="$(openssl rand -hex 32)"' in launch
    assert "SCIPROBE_PRIVATE_PROBE_ROOT" not in launch
    assert "/workspace/sciprobe-private" not in main_mount_line
    gym_actor_python = "/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python"
    auth_server_python = (
        "/opt/gym_venvs/resources_servers/sciprobe_ns_tools/.venv/bin/python"
    )
    assert f"test -x {gym_actor_python}" in launch
    assert (
        f"{gym_actor_python} \\\n    examples/validate_sciprobe_canary_configs.py"
        in launch
    )
    assert (
        f"{gym_actor_python} \\\n    examples/validate_sciprobe_no_replay.py" in launch
    )
    assert (
        f"{gym_actor_python} \\\n    examples/validate_sciprobe_canary_configs.py \\\n"
        "    --signal-config ${CONFIG_PATH}" in launch
    )
    assert f"test -x {auth_server_python}" in launch
    assert (
        f"{auth_server_python} \\\n    examples/validate_sciprobe_signal_canary_auth.py"
        in launch
    )
    policy_actor_python = (
        "/opt/ray_venvs/nemo_rl.models.policy.workers."
        "dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python"
    )
    assert f"test -x {policy_actor_python}" in launch
    assert (
        f"{policy_actor_python} \\\n    examples/validate_lightning_mtp_disabled.py"
        in launch
    )

    prefix_preflight = PREFIX_PREFLIGHT.read_text(encoding="utf-8")
    assert "sciprobe_rl_signal-prefix-preflight-" in prefix_preflight
    assert "SLURM_PARTITION=" in prefix_preflight
    assert ":-cpu}" in prefix_preflight
    assert "--gres" not in prefix_preflight
    assert "export SCIPROBE_REQUIRE_SANDBOX_SECCOMP_PREFLIGHT=1" in prefix_preflight
    assert (
        f"{gym_actor_python} \\\n    examples/validate_sciprobe_canary_configs.py \\\n"
        "    --signal-config ${CONFIG_PATH}" in prefix_preflight
    )
    assert f"test -x {auth_server_python}" in prefix_preflight
    assert (
        f"{auth_server_python} \\\n    examples/validate_sciprobe_signal_canary_auth.py"
        in prefix_preflight
    )
    assert (
        f"{policy_actor_python} \\\n    examples/validate_lightning_mtp_disabled.py"
        in prefix_preflight
    )
    for validator_command in (
        "examples/validate_sciprobe_ray_loopback.py",
        "examples/validate_sciprobe_canary_configs.py",
        "examples/validate_sciprobe_no_replay.py",
        "examples/validate_lightning_mtp_disabled.py",
        "examples/validate_lightning_tool_tokenization.py",
        "examples/validate_sciprobe_signal_canary_grader.py",
        "examples/validate_sciprobe_signal_canary_verifier.py",
        "examples/validate_sciprobe_signal_canary_auth.py",
        "examples/validate_sciprobe_signal_canary_isolation.py",
    ):
        assert validator_command in launch
        assert validator_command in prefix_preflight
    assert "[SCIPROBE_SIGNAL_PREFIX_PREFLIGHT_OK]" in prefix_preflight

    config_preflight = CONFIG_PREFLIGHT.read_text(encoding="utf-8")
    assert "export NEMO_RL_VENV_DIR=/opt/ray_venvs" in config_preflight
    assert '"${gym_actor_python}" -c "import nemo_gym, openai"' in config_preflight
    assert (
        '"${gym_actor_python}" examples/validate_sciprobe_canary_configs.py'
        in config_preflight
    )

    isolation = ISOLATION_VALIDATOR.read_text(encoding="utf-8")
    assert '"SCIPROBE_LANDLOCK_FILTER_ACTIVE"' in isolation
    assert "CROSS_SESSION_SANDBOX_CODE.replace(" in isolation
    assert "reader_session_id" in isolation
    assert '"sibling_signal_zero_denied"' in isolation
    assert '"sibling_prlimit_denied"' in isolation
    assert '"sibling_sched_setaffinity_denied"' in isolation
    assert '"sibling_setpriority_denied"' in isolation
    assert '"sysv_shmat_denied"' in isolation
    assert '"only_control_unix_fd"' in isolation
    assert '"open_fd_count"' in isolation
    assert isolation.count('["open_fd_count"] == 1') == 0
    assert isolation.count('["socket_fd_count"] == 1') == 3
    assert isolation.count('["non_unix_socket_fd_count"] == 0') == 3
    assert "assert cross_session[marker] is True" in isolation
    for ray_auth_name in (
        "RAY_AUTH_MODE",
        "RAY_AUTH_TOKEN",
        "RAY_AUTH_TOKEN_PATH",
    ):
        assert isolation.count(f'"{ray_auth_name}"') == 2


def test_signal_verifier_uses_only_provenance_pinned_hidden_checker() -> None:
    config = OmegaConf.to_container(OmegaConf.load(CHECKS_CONFIG), resolve=True)
    assert isinstance(config, dict)
    definition = config["sciprobe_checks"]["resources_servers"]["sciprobe_checks"][
        "probes"
    ]["q3:c013:d0"]
    assert definition["checks_sha256"] == (
        "38fdcd62becccf79f602aeabca8aaf318b731ea3e8c80df668895026ce6bde34"
    )
    assert definition["data_tree_sha256"] == (
        "16713f67f959a4c276baea508c1fb64fa54bf622f4e14b0b4def77d6c152a590"
    )
    assert "expected_answer" not in definition
    assert "gold_sha256" not in definition
    source = CHECKS_APP.read_text(encoding="utf-8")
    assert "_run_hidden_checker" in source
    assert 'RUNNER_PATH = Path(__file__).with_name("grader_runner.py")' in source
    assert "checks_sha256" in source
    assert "data_tree_sha256" in source
    assert CHECKS_APP.with_name("grader_runner.py").is_file()


def test_merged_gym_config_replaces_builtin_ns_tools_implementation() -> None:
    from nemo_gym.global_config import (
        GlobalConfigDictParser,
        GlobalConfigDictParserConfig,
    )

    recipe = load_config(RECIPE)
    assert (
        recipe.policy["dtensor_cfg"]["automodel_kwargs"]["num_nextn_predict_layers"]
        == 0
    )
    gym = OmegaConf.create(OmegaConf.to_container(recipe.env.nemo_gym, resolve=True))
    extension_root = Path("examples/nemo_gym_extensions").resolve()
    gym.config_paths = [
        str(extension_root / path) if (extension_root / path).is_file() else path
        for path in gym.config_paths
    ]
    gym.policy_model_name = "validation-model"
    gym.policy_api_key = "validation-key"
    gym.policy_base_url = "http://127.0.0.1:8000/v1"

    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=gym,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )

    implementations = resolved.ns_tools.resources_servers
    assert list(implementations) == ["sciprobe_ns_tools"]
    assert implementations.sciprobe_ns_tools.num_workers == 1
    assert (
        resolved.ns_tools_simple_agent.responses_api_agents[
            "sciprobe_simple_agent"
        ].resources_server.name
        == "ns_tools"
    )
    resolved_policy = resolved.policy_model.responses_api_models.vllm_model
    assert resolved_policy.api_key is None
    assert resolved_policy.api_key_env_var == "SCIPROBE_POLICY_GENERATION_TOKEN"
    assert (
        resolved_policy.trusted_ingress_token_env_var
        == "SCIPROBE_TRUSTED_INGRESS_TOKEN"
    )
    assert resolved.head_server.host == "127.0.0.1"
    parser = GlobalConfigDictParser()
    for instance in parser.filter_for_server_instance_configs(resolved):
        assert instance.get_inner_run_server_config_dict().host == "127.0.0.1"


def test_each_component_installs_shared_capability_module() -> None:
    package = tomllib.loads(
        (EXTENSION_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert package["tool"]["setuptools"]["py-modules"] == [
        "sciprobe_capability",
        "sciprobe_capability_store",
    ]
    for requirements in COMPONENT_REQUIREMENTS:
        assert "-e ../.." in requirements.read_text(encoding="utf-8").splitlines()


def test_runtime_materializer_never_persists_capability(
    tmp_path: Path,
) -> None:
    source_row = json.loads(DATASET.read_text(encoding="utf-8"))
    output = tmp_path / "runtime" / "train.jsonl"
    capability_store = output.parent / "capability-results.sqlite3"
    env = os.environ.copy()
    env["SCIPROBE_VERIFIER_CAPABILITY"] = "legacy-secret-must-not-persist"
    env["SCIPROBE_CAPABILITY_SIGNING_KEY"] = "ab" * 32
    env["SCIPROBE_TRUSTED_INGRESS_TOKEN"] = "cd" * 32
    env["SCIPROBE_POLICY_GENERATION_TOKEN"] = "ef" * 32

    completed = subprocess.run(
        [
            sys.executable,
            "examples/materialize_sciprobe_signal_runtime_dataset.py",
            "--source",
            str(DATASET),
            "--output",
            str(output),
            "--capability-store",
            str(capability_store),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    runtime_row = json.loads(output.read_text(encoding="utf-8"))
    assert runtime_row == source_row
    serialized = output.read_text(encoding="utf-8")
    assert "legacy-secret-must-not-persist" not in serialized
    assert "_sciprobe_verifier_capability" not in serialized
    assert "SCIPROBE_CAPABILITY_SIGNING_KEY" not in serialized
    assert "SCIPROBE_TRUSTED_INGRESS_TOKEN" not in serialized
    assert "SCIPROBE_POLICY_GENERATION_TOKEN" not in serialized
    assert env["SCIPROBE_TRUSTED_INGRESS_TOKEN"] not in serialized
    assert env["SCIPROBE_POLICY_GENERATION_TOKEN"] not in serialized
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert stat.S_IMODE(output.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(capability_store.stat().st_mode) == 0o600
    assert capability_store.read_bytes() == b""


def test_at_most_once_capability_preflight(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    store = runtime_dir / "capability-results.sqlite3"
    env = os.environ.copy()
    env["SCIPROBE_CAPABILITY_STORE_PATH"] = str(store)
    env["SCIPROBE_CAPABILITY_SIGNING_KEY"] = "ab" * 32
    env["SCIPROBE_VERIFIER_TOKEN"] = "0123456789abcdef" * 2
    env["PYTHONPATH"] = ":".join(
        (
            "examples",
            "examples/nemo_gym_extensions",
            "3rdparty/Gym-workspace/Gym",
            env.get("PYTHONPATH", ""),
        )
    )

    completed = subprocess.run(
        [sys.executable, "examples/validate_sciprobe_signal_canary_auth.py"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "ok"
    for required in (
        "completed_result_retry_cached",
        "concurrent_identical_retries_cached",
        "mismatched_body_rejected",
        "restart_shared_store",
        "ambiguous_pending_not_replayed",
        "expiry_cleanup",
        "capability_scrubbed",
        "backend_details_scrubbed",
    ):
        assert result[required] is True
    assert stat.S_IMODE(store.stat().st_mode) == 0o600
