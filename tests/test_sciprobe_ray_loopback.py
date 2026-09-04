from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAY_SUB = ROOT / "ray.sub"
LAUNCHER = ROOT / "examples/launch_sciprobe_lightning_signal_canary.sh"
VALIDATOR = ROOT / "examples/validate_sciprobe_ray_loopback.py"
PREFLIGHT = ROOT / "examples/preflight_sciprobe_ray_loopback.slurm"


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sciprobe_ray_loopback", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_signal_launcher_enables_ray_loopback_and_runs_live_validator() -> None:
    launcher = LAUNCHER.read_text()
    assert 'RUN_ID="${RUN_ID:-r9}"' in launcher
    assert "export RAY_LOOPBACK_ONLY=1" in launcher
    assert "python examples/validate_sciprobe_ray_loopback.py" in launcher
    assert "--require-worker" in launcher
    assert (
        "ray_session_log_dir=\\$(readlink -f /tmp/ray/session_latest/logs)" in launcher
    )
    assert "--require-token-auth" in launcher
    assert '--ray-log-dir \\"\\${ray_session_log_dir}\\"' in launcher
    assert "--nodes=1" in launcher


def test_ray_sub_loopback_mode_is_one_node_and_binds_head_worker_dashboard() -> None:
    ray_sub = RAY_SUB.read_text()
    assert "RAY_LOOPBACK_ONLY=${RAY_LOOPBACK_ONLY:-0}" in ray_sub
    assert 'SLURM_JOB_NUM_NODES" -ne 1' in ray_sub
    assert '"${RAY_DEBUG:-}" == "legacy"' in ray_sub
    assert "external Ray debugger is forbidden" in ray_sub
    assert 'export RAY_ADDRESS="127.0.0.1:${PORT}"' in ray_sub
    for redirect in (
        "RAY_REDIS_ADDRESS",
        "RAY_API_SERVER_ADDRESS",
        "RAY_DASHBOARD_ADDRESS",
        "RAY_AGENT_ADDRESS",
    ):
        assert f"unset {redirect}" in ray_sub
    assert "export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0" in ray_sub
    assert 'head_node_ip="127.0.0.1"' in ray_sub
    assert 'WORKER_NODE_IP_ARG="--node-ip-address=127.0.0.1"' in ray_sub
    assert '--dashboard-host="$head_node_ip"' in ray_sub
    assert "$WORKER_NODE_IP_ARG" in ray_sub


def test_ray_sub_loopback_mode_uses_file_backed_native_auth() -> None:
    ray_sub = RAY_SUB.read_text()
    assert 'RAY_AUTH_DIR="$LOG_DIR/.ray-auth"' in ray_sub
    assert 'export RAY_AUTH_TOKEN_PATH="$RAY_AUTH_DIR/token"' in ray_sub
    assert "os.O_CREAT | os.O_EXCL" in ray_sub
    assert "auth_dir.mkdir(mode=0o700)" in ray_sub
    assert "0o600" in ray_sub
    assert "unset RAY_AUTH_TOKEN" in ray_sub
    assert "export RAY_AUTH_MODE=token" in ray_sub
    assert "--container-env=RAY_AUTH_MODE,RAY_AUTH_TOKEN_PATH" in ray_sub
    assert "_nrl_scan_ray_auth_logs" in ray_sub
    assert 'rm -f -- "$RAY_AUTH_TOKEN_PATH"' in ray_sub


def test_ray_sub_mounts_and_verifies_exact_per_job_log_canary() -> None:
    ray_sub = RAY_SUB.read_text()
    assert 'MOUNTS+=",$LOG_DIR:$LOG_DIR"' in ray_sub
    assert 'MOUNTS="$LOG_DIR:$LOG_DIR"' in ray_sub
    assert "_nrl_wait_for_shared_fs_canary()" in ray_sub
    assert '[[ "$observed" == "$expected_job_id" ]]' in ray_sub
    assert ray_sub.count("$(declare -f _nrl_wait_for_shared_fs_canary)") == 2
    assert '"$LOG_DIR/.shared_fs_canary" "$SLURM_JOB_ID" "head node" 30' in ray_sub
    assert (
        '"$LOG_DIR/.shared_fs_canary" "$SLURM_JOB_ID" "worker \\$SLURM_PROCID" 30'
        in ray_sub
    )


def test_ray_sub_stops_and_reaps_writers_before_final_secret_cleanup() -> None:
    ray_sub = RAY_SUB.read_text()
    cleanup_start = ray_sub.index("_nrl_log_exit()")
    cleanup_end = ray_sub.index("trap _nrl_log_exit", cleanup_start)
    cleanup = ray_sub[cleanup_start:cleanup_end]
    assert cleanup.index("_nrl_stop_background_sruns") < cleanup.index(
        "_nrl_scan_ray_auth_logs"
    )
    assert cleanup.index("_nrl_scan_ray_auth_logs") < cleanup.index(
        'rm -f -- "$RAY_AUTH_TOKEN_PATH"'
    )
    assert "declare -p SRUN_PIDS" in ray_sub
    assert 'kill -TERM "$pid"' in ray_sub
    assert 'kill -KILL "$pid"' in ray_sub
    assert 'wait "$pid"' in ray_sub
    assert "trap '_nrl_log_exit 143' TERM" in ray_sub
    assert "trap '_nrl_log_exit 129' HUP" in ray_sub
    assert "trap '_nrl_log_exit 130' INT" in ray_sub
    assert '[[ "${RAY_AUTH_MODE:-}" == "token" ]]' in cleanup
    assert "Failed to remove the Ray auth token file" in cleanup
    assert '[[ -e "$RAY_AUTH_TOKEN_PATH" ]]' in cleanup


def test_ray_sub_term_trap_cannot_report_success(tmp_path: Path) -> None:
    ray_sub = RAY_SUB.read_text()
    cleanup_start = ray_sub.index("_nrl_log_exit()")
    trap_end = ray_sub.index("\n\n", ray_sub.index("trap _nrl_log_exit EXIT"))
    cleanup_and_traps = ray_sub[cleanup_start:trap_end]
    script = tmp_path / "term-cleanup.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "RAY_AUTH_MODE=disabled\n"
        "log_phase() { :; }\n"
        "_nrl_stop_background_sruns() { :; }\n"
        "_nrl_scan_ray_auth_logs() { :; }\n"
        + cleanup_and_traps
        + "\nkill -TERM $$\nexit 0\n"
    )
    result = subprocess.run(["bash", str(script)], check=False)
    assert result.returncode == 143


def test_trusted_head_scans_node_local_ray_logs_after_ray_stop() -> None:
    ray_sub = RAY_SUB.read_text()
    head_cleanup_start = ray_sub.index("_nrl_head_exit()")
    head_cleanup_end = ray_sub.index("trap _nrl_head_exit", head_cleanup_start)
    head_cleanup = ray_sub[head_cleanup_start:head_cleanup_end]
    assert head_cleanup.index('kill -TERM "\\$pid"') < head_cleanup.index(
        "ray stop --force"
    )
    assert head_cleanup.index('wait "\\$pid"') < head_cleanup.index("ray stop --force")
    assert head_cleanup.index("ray stop --force") < head_cleanup.index(
        '_nrl_scan_ray_auth_logs "\\${RAY_AUTH_TOKEN_PATH:-}" /tmp/ray'
    )
    assert "$(declare -f _nrl_scan_ray_auth_logs)" in ray_sub
    assert ray_sub.count("HEAD_SIDECAR_PIDS+=(\\$!)") == 3
    assert "trap '_nrl_head_exit 143' TERM" in ray_sub
    assert "trap '_nrl_head_exit 129' HUP" in ray_sub
    assert "trap '_nrl_head_exit 130' INT" in ray_sub


def test_ray_sub_head_term_trap_cannot_report_success(tmp_path: Path) -> None:
    ray_sub = RAY_SUB.read_text()
    cleanup_start = ray_sub.index("_nrl_head_exit()")
    trap_end = ray_sub.index("\n\n", ray_sub.index("trap _nrl_head_exit EXIT"))
    cleanup_and_traps = ray_sub[cleanup_start:trap_end].replace("\\$", "$")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    script = tmp_path / "head-term-cleanup.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        f"LOG_DIR={log_dir}\n"
        "RAY_AUTH_MODE=disabled\n"
        "HEAD_SIDECAR_PIDS=()\n"
        "_nrl_scan_ray_auth_logs() { :; }\n"
        + cleanup_and_traps
        + "\nkill -TERM $$\nexit 0\n"
    )
    result = subprocess.run(["bash", str(script)], check=False)
    assert result.returncode == 143


def test_ray_auth_is_not_forwarded_to_the_model_sandbox() -> None:
    ray_sub = RAY_SUB.read_text()
    assert "RAY_AUTH_MODE|RAY_AUTH_TOKEN|RAY_AUTH_TOKEN_PATH" in ray_sub
    assert "SANDBOX_EXTRA_MOUNTS must not expose the Ray auth token path" in ray_sub
    assert "unset RAY_AUTH_MODE RAY_AUTH_TOKEN RAY_AUTH_TOKEN_PATH" in ray_sub


def test_exact_image_preflight_uses_ray_local_only_mode() -> None:
    preflight = PREFLIGHT.read_text()
    assert "#SBATCH --partition=cpu" in preflight
    assert "#SBATCH --time=00:20:00" in preflight
    assert "export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0" in preflight
    assert 'export RAY_ADDRESS="ray://192.0.2.10:10001"' in preflight
    assert 'export RAY_ADDRESS="127.0.0.1:1200"' in preflight
    assert '[[ "${RAY_ADDRESS}" == "127.0.0.1:1200" ]]' in preflight
    assert "--node-ip-address=127.0.0.1" in preflight
    assert "python examples/validate_sciprobe_ray_loopback.py" in preflight
    assert "--require-worker" in preflight
    assert "export RAY_AUTH_MODE=token" in preflight
    assert 'export RAY_AUTH_TOKEN_PATH="${ray_auth_token_path}"' in preflight
    assert "--require-token-auth" in preflight
    assert "--ray-log-dir" in preflight
    assert "RAY_TOKEN_FINAL_LOG_SCAN" in preflight
    assert "--num-cpus=8" in preflight


def test_live_validator_covers_fixed_ray_metric_ports() -> None:
    validator = _load_validator()
    assert 44217 in validator.RAY_CONTROL_PORTS
    assert 44227 in validator.RAY_CONTROL_PORTS


def test_live_validator_pins_driver_to_loopback() -> None:
    source = VALIDATOR.read_text()
    assert "_node_ip_address=RAY_LOOPBACK_HOST" in source


def test_unauthenticated_timeout_is_bounded_reaped_and_cross_checked() -> None:
    source = VALIDATOR.read_text()
    assert "start_new_session=True" in source
    assert "process.wait(timeout=20)" in source
    assert "os.killpg(process.pid, signal.SIGKILL)" in source
    assert "process.wait(timeout=5)" in source
    assert 'subprocess_outcome = "timeout"' in source
    assert "dashboard_status != 401" in source
    assert 'ping.remote("after-unauthenticated-probe")' in source


def test_live_validator_rejects_any_node_ip_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_validator()
    monkeypatch.setattr(
        validator,
        "_non_loopback_ipv4_addresses",
        lambda: ["10.2.3.4"],
    )
    monkeypatch.setattr(
        validator,
        "RAY_CONTROL_PORTS",
        (1200, 8265),
    )
    monkeypatch.setattr(
        validator,
        "_connects",
        lambda host, port, timeout=0.01: host == "10.2.3.4" and port == 8265,
    )
    monkeypatch.setattr(
        validator,
        "_ray_owned_tcp_listeners",
        lambda *, extra_pids=None, exclude_inodes=None: (
            3,
            [{"address": "127.0.0.1", "port": 1200, "owners": []}],
        ),
    )
    monkeypatch.setattr(validator, "_non_loopback_control_listeners", lambda: [])

    with pytest.raises(RuntimeError, match="10.2.3.4:8265"):
        validator.main()


def test_proc_listener_gate_finds_all_ray_owned_sockets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_validator()
    proc = tmp_path / "proc"
    process = proc / "101"
    (process / "fd").mkdir(parents=True)
    (process / "cmdline").write_bytes(
        b"/opt/nemo_rl_venv/lib/python3.13/site-packages/ray/dashboard/dashboard.py\0"
    )
    (process / "fd" / "3").symlink_to("socket:[1]")
    (process / "fd" / "4").symlink_to("socket:[2]")
    tcp = tmp_path / "tcp"
    tcp6 = tmp_path / "tcp6"
    header = "  sl  local_address rem_address   st\n"
    tcp.write_text(
        header
        + "   0: 00000000:2049 00000000:0000 0A 00000000:00000000 00:00000000 00000000 0 0 1\n"
    )
    tcp6.write_text(
        header + "   0: 0000000000000000FFFF00000100007F:04B0 "
        "00000000000000000000000000000000:0000 0A 00000000:00000000 "
        "00:00000000 00000000 0 0 2\n"
    )
    monkeypatch.setattr(validator, "PROC_ROOT", proc)
    monkeypatch.setattr(validator, "PROC_TCP_PATHS", (tcp, tcp6))

    process_count, listeners = validator._ray_owned_tcp_listeners()
    assert process_count == 1
    assert [(item["address"], item["port"]) for item in listeners] == [
        ("::ffff:127.0.0.1", 1200),
        ("0.0.0.0", 8265),
    ]
    assert validator._is_expected_ray_loopback("::ffff:127.0.0.1") is True
    assert validator._is_expected_ray_loopback("127.0.0.2") is False
    assert validator._is_expected_ray_loopback("::1") is False


def test_setproctitle_ray_processes_are_included() -> None:
    validator = _load_validator()
    assert validator._is_ray_process([b"ray::IDLE"]) is True
    assert validator._is_ray_process([b"ray::DashboardAgent"]) is True


def test_missing_explicit_ray_pid_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_validator()
    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(validator, "PROC_ROOT", proc)

    with pytest.raises(RuntimeError, match="explicit Ray PIDs were not inspected: 999"):
        validator._ray_owned_tcp_listeners(extra_pids={999})


def test_inherited_listener_inode_is_excluded_from_ray_owned_sockets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_validator()
    proc = tmp_path / "proc"
    process = proc / "101"
    (process / "fd").mkdir(parents=True)
    (process / "cmdline").write_bytes(b"ray::IDLE\0")
    (process / "fd" / "3").symlink_to("socket:[17]")
    (process / "fd" / "4").symlink_to("socket:[22]")
    tcp = tmp_path / "tcp"
    tcp6 = tmp_path / "tcp6"
    header = "  sl  local_address rem_address   st\n"
    tcp.write_text(
        header + "   0: 00000000:57BB 00000000:0000 0A 00000000:00000000 "
        "00:00000000 00000000 0 0 17\n"
        + "   1: 0100007F:07D0 00000000:0000 0A 00000000:00000000 "
        "00:00000000 00000000 0 0 22\n"
    )
    tcp6.write_text(header)
    monkeypatch.setattr(validator, "PROC_ROOT", proc)
    monkeypatch.setattr(validator, "PROC_TCP_PATHS", (tcp, tcp6))

    _, listeners = validator._ray_owned_tcp_listeners(
        extra_pids={101},
        exclude_inodes={"17"},
    )

    assert [(item["address"], item["port"]) for item in listeners] == [
        ("127.0.0.1", 2000)
    ]


def test_advertised_driver_core_worker_requires_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import grpc
    from ray.core.generated import common_pb2, core_worker_pb2, core_worker_pb2_grpc

    validator = _load_validator()
    calls: list[dict[str, object]] = []
    closed = False

    class FakeRpcError(grpc.RpcError):
        def code(self) -> grpc.StatusCode:
            return grpc.StatusCode.UNAUTHENTICATED

    class FakeCall:
        def __call__(
            self,
            request: object,
            *,
            timeout: int,
            metadata: tuple[tuple[str, str], ...] | None = None,
        ) -> object:
            calls.append({"request": request, "timeout": timeout, "metadata": metadata})
            if metadata is None:
                raise FakeRpcError()
            return core_worker_pb2.GetCoreWorkerStatsReply(
                core_worker_stats=common_pb2.CoreWorkerStats(
                    pid=os.getpid(),
                    worker_id=b"worker-id",
                )
            )

    class FakeStub:
        def __init__(self, channel: object) -> None:
            self.GetCoreWorkerStats = FakeCall()

    class FakeChannel:
        def close(self) -> None:
            nonlocal closed
            closed = True

    class FakeReady:
        def result(self, *, timeout: int) -> None:
            assert timeout == 5

    channel = FakeChannel()
    monkeypatch.setattr(grpc, "insecure_channel", lambda *args, **kwargs: channel)
    monkeypatch.setattr(grpc, "channel_ready_future", lambda value: FakeReady())
    monkeypatch.setattr(core_worker_pb2_grpc, "CoreWorkerServiceStub", FakeStub)

    authenticated = validator._probe_driver_core_worker_auth(
        target="127.0.0.1:2000",
        intended_worker_id=b"worker-id",
        token=b"a" * 64,
    )

    assert authenticated is True
    assert closed is True
    assert len(calls) == 2
    assert calls[0]["metadata"] == (("authorization", "Bearer " + "a" * 64),)
    assert calls[1]["metadata"] is None


def test_token_scanner_detects_secret_across_chunk_boundary(tmp_path: Path) -> None:
    validator = _load_validator()
    token_path = tmp_path / "auth" / "token"
    token_path.parent.mkdir(mode=0o700)
    token = b"a" * 64
    token_path.write_bytes(token)
    os.chmod(token_path, 0o600)
    log_root = tmp_path / "logs"
    log_root.mkdir()
    safe_log = log_root / "safe.log"
    safe_log.write_bytes(b"safe")
    result = validator._scan_token_in_tree(token_path, token, log_root)
    assert result["token_log_leaks"] == 0

    leaking_log = log_root / "leak.log"
    prefix_size = validator.SCAN_CHUNK_BYTES - 31
    leaking_log.write_bytes(b"x" * prefix_size + token + b"tail")
    with pytest.raises(RuntimeError, match="leak.log") as error:
        validator._scan_token_in_tree(token_path, token, log_root)
    assert token.decode("ascii") not in str(error.value)
