from __future__ import annotations

from pathlib import Path

WRAPPER = Path("examples/start_sciprobe_loopback_sandbox.sh")
LAUNCHER = Path("examples/launch_sciprobe_lightning_signal_canary.sh")
PREFLIGHT = Path("examples/preflight_sciprobe_network_blocking.slurm")
VALIDATOR = Path("examples/validate_sciprobe_network_blocking.py")
RAY_SUB = Path("ray.sub")
SIDECAR = Path("examples/start_sciprobe_proc_preflight_sidecar.sh")


def test_secure_wrapper_fails_closed_and_patches_both_public_listeners() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert 'SANDBOX_FORCE_SINGLE_NODE:-0}" != "1"' in source
    assert 'NUM_WORKERS:-}" != "1"' in source
    assert 'worker_public = "http-socket = 0.0.0.0:${WORKER_PORT}"' in source
    assert 'worker_loopback = "http-socket = 127.0.0.1:${WORKER_PORT}"' in source
    assert 'nginx_public = "listen ${NGINX_PORT};"' in source
    assert 'nginx_loopback = "listen 127.0.0.1:${NGINX_PORT};"' in source
    assert "count(worker_public) != 1" in source
    assert "count(nginx_public) != 1" in source
    assert 'SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK:-}" != "1"' in source
    assert 'python3 "${seccomp_validator}" --hook-dir "${seccomp_hook_dir}"' in source
    assert "sciprobe-seccomp-preflight.ready" in source
    assert "[SCIPROBE_SANDBOX_SECCOMP_PREFLIGHT_OK]" in source


def test_real_canary_mounts_loopback_wrapper_on_an_exclusive_node() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    ray_sub = RAY_SUB.read_text(encoding="utf-8")
    assert "--exclusive" in source
    assert (
        "examples/start_sciprobe_loopback_sandbox.sh:"
        "/workspace/start-sciprobe-loopback-sandbox.sh:ro"
    ) in source
    assert (
        "examples/validate_sciprobe_sandbox_seccomp.py:"
        "/workspace/validate-sciprobe-sandbox-seccomp.py:ro"
    ) in source
    assert (
        "unshare --pid --fork --mount-proc --kill-child "
        "/workspace/start-sciprobe-loopback-sandbox.sh"
    ) in source
    sandbox_env = next(
        line
        for line in source.splitlines()
        if line.startswith("export SANDBOX_ENV_VARS=")
    )
    assert "NUM_WORKERS=1" in sandbox_env
    assert "SANDBOX_FORCE_SINGLE_NODE=1" in sandbox_env
    assert "export SCIPROBE_REQUIRE_SANDBOX_SECCOMP_PREFLIGHT=1" in source
    assert (
        'SANDBOX_CONTAINER_ARGS+=(--container-env="${SANDBOX_CONTAINER_ENV_NAMES}")'
        in ray_sub
    )
    assert 'sandbox_name="${sandbox_assignment%%=*}"' in ray_sub
    assert '--error "$SANDBOX_LOG_DIR/sandbox-stderr-%t.log"' in ray_sub
    assert "sandbox-[0-9]*.log" in ray_sub
    assert "sciprobe-seccomp-preflight.ready" in ray_sub
    assert "[SCIPROBE_SANDBOX_SECCOMP_PREFLIGHT_OK]" in ray_sub


def test_exact_image_preflight_checks_listener_and_remote_endpoint_denial() -> None:
    preflight = PREFLIGHT.read_text(encoding="utf-8")
    validator = VALIDATOR.read_text(encoding="utf-8")
    assert "#SBATCH --exclusive" in preflight
    assert (
        "SCIPROBE_SANDBOX_START_COMMAND=/workspace/start-loopback-sandbox.sh"
        in preflight
    )
    assert '--container-env="${sandbox_env_names}"' in preflight
    assert "NUM_WORKERS,SANDBOX_FORCE_SINGLE_NODE" in preflight
    assert "SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK,PYTHONPATH" in preflight
    assert preflight.count('grep -Fqx "[SCIPROBE_SANDBOX_SECCOMP_PREFLIGHT_OK]"') >= 2
    assert (
        "examples/validate_sciprobe_sandbox_seccomp.py:"
        "/workspace/validate-sciprobe-sandbox-seccomp.py:ro"
    ) in preflight
    assert "--remote-sandbox-host" in preflight
    assert '_proc_tcp_listeners(Path("/proc/net/tcp"), socket.AF_INET)' in validator
    assert '_proc_tcp_listeners(Path("/proc/net/tcp6"), socket.AF_INET6)' in validator
    assert '"POST",\n                f"{base_url}/execute"' in validator
    assert '"GET", f"{base_url}/sessions"' in validator
    assert (
        '"DELETE",\n                f"{base_url}/sessions/sciprobe-unauthorized"'
        in validator
    )
    assert 'expected == {("127.0.0.1", 6000), ("127.0.0.1", 6001)}' in validator
    assert 'Path(os.environ["SCIPROBE_SESSION_STATE_DIR"])' in validator
    assert 'extra_args={"request_id": request_id + "-reader"}' in validator
    assert "STATEFUL_PROOF_CHUNK_ITEMS = 1" in validator
    assert 'variable_name="sciprobe_security_proof"' in validator
    assert 'variable_name="sciprobe_cross_session_proof"' in validator
    assert '"sibling_read_denied"' in validator
    assert '"sibling_signal_zero_denied"' in validator
    assert '"sibling_prlimit_denied"' in validator
    assert '"sibling_sched_setaffinity_denied"' in validator
    assert '"sibling_setpriority_denied"' in validator
    assert '"sysv_shmat_denied"' in validator
    assert '"only_control_unix_fd"' in validator
    assert '"open_fd_count"' in validator
    assert "len(socket_fds) == 1" in validator
    assert 'cross_session.get("socket_fd_count") == 1' in validator
    assert 'proof.get("open_fd_count") == 1' not in validator
    assert 'followup.get("open_fd_count") == 1' not in validator
    assert 'cross_session.get("open_fd_count") == 1' not in validator
    assert 'result["cross_session"]["socket_fd_count"] == 1' in preflight
    assert 'result["cross_session"]["open_fd_count"] == 1' not in preflight
    assert '"global_tmp_create_denied"' in validator
    assert 'result["cross_session"][name] is True' in preflight


def test_model_sandbox_has_no_writable_host_control_bridge() -> None:
    ray_sub = RAY_SUB.read_text(encoding="utf-8")
    preflight = PREFLIGHT.read_text(encoding="utf-8")
    sidecar = SIDECAR.read_text(encoding="utf-8")

    assert "$SANDBOX_PORTS_DIR:$SANDBOX_PORTS_DIR" not in ray_sub
    assert 'SANDBOX_RUNTIME_DIR="/tmp/sciprobe-sandbox-${SLURM_JOB_ID}"' in ray_sub
    assert "[SCIPROBE_SANDBOX_READY]" in ray_sub
    assert "SANDBOX_PORTS_DIR=${control_dir}" not in preflight
    assert (
        "${control_dir}:${control_dir},${CODE_DIR}/examples/start_sciprobe"
        not in preflight
    )
    assert "SCIPROBE_SIDECAR_READY_FILE" not in preflight
    assert "SCIPROBE_SIDECAR_STOP_FILE" not in preflight
    assert "SCIPROBE_SIDECAR_READY_FILE" not in sidecar
    assert "SCIPROBE_SIDECAR_STOP_FILE" not in sidecar
    assert "sciprobe-seccomp-preflight.ready" in sidecar
