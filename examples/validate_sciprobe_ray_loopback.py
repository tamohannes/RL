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

"""Fail unless this canary's Ray control plane is unreachable via node IP."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

RAY_LOOPBACK_HOST = "127.0.0.1"
RAY_CONTROL_PORTS = tuple(range(1200, 1202)) + tuple(range(1301, 1313))
RAY_CONTROL_PORTS += tuple(range(2000, 3000)) + (8265, 44217, 44227)
PROC_TCP_PATHS = (Path("/proc/net/tcp"), Path("/proc/net/tcp6"))
PROC_ROOT = Path("/proc")
TOKEN_BYTES = 64
SCAN_CHUNK_BYTES = 1024 * 1024


def _non_loopback_ipv4_addresses() -> list[str]:
    addresses: set[str] = set()
    for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
        address = info[4][0]
        if not address.startswith("127."):
            addresses.add(address)

    if not addresses:
        # No packet is sent by UDP connect; the kernel only selects the source
        # interface that would route this documentation-only destination.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            address = probe.getsockname()[0]
            if not address.startswith("127."):
                addresses.add(address)

    if not addresses:
        raise RuntimeError(
            "could not resolve the compute node's non-loopback IPv4 address"
        )
    return sorted(addresses)


def _connects(host: str, port: int, timeout: float = 0.01) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout)
        return probe.connect_ex((host, port)) == 0


def _decode_proc_address(encoded: str, *, ipv6: bool) -> str:
    raw = bytes.fromhex(encoded)
    if ipv6:
        if len(raw) != 16:
            raise RuntimeError(f"invalid /proc IPv6 address: {encoded!r}")
        # Linux prints each IPv6 u32 in host byte order.
        raw = b"".join(raw[offset : offset + 4][::-1] for offset in range(0, 16, 4))
        return socket.inet_ntop(socket.AF_INET6, raw)
    if len(raw) != 4:
        raise RuntimeError(f"invalid /proc IPv4 address: {encoded!r}")
    return socket.inet_ntoa(raw[::-1])


def _is_expected_ray_loopback(address: str) -> bool:
    parsed = ipaddress.ip_address(address)
    expected = ipaddress.IPv4Address(RAY_LOOPBACK_HOST)
    if parsed == expected:
        return True
    return isinstance(parsed, ipaddress.IPv6Address) and (
        parsed.ipv4_mapped is not None and parsed.ipv4_mapped == expected
    )


def _tcp_listener_table() -> dict[str, dict[str, object]]:
    listeners: dict[str, dict[str, object]] = {}
    for path in PROC_TCP_PATHS:
        ipv6 = path.name == "tcp6"
        with path.open(encoding="ascii") as stream:
            next(stream, None)
            for line in stream:
                fields = line.split()
                if len(fields) < 4 or fields[3] != "0A":
                    continue
                encoded_address, encoded_port = fields[1].rsplit(":", 1)
                port = int(encoded_port, 16)
                address = _decode_proc_address(encoded_address, ipv6=ipv6)
                inode = fields[9]
                listeners[inode] = {
                    "inode": inode,
                    "source": str(path),
                    "address": address,
                    "port": port,
                }
    return listeners


def _is_ray_process(args: list[bytes]) -> bool:
    exact_names = {b"gcs_server", b"raylet", b"plasma_store_server"}
    for arg in args:
        name = arg.rsplit(b"/", 1)[-1]
        if (
            arg.startswith(b"ray::")
            or name in exact_names
            or b"/site-packages/ray/" in arg
            or b"/ray/core/" in arg
        ):
            return True
    return False


def _ray_owned_tcp_listeners(
    *,
    extra_pids: set[int] | None = None,
    exclude_inodes: set[str] | None = None,
) -> tuple[int, list[dict[str, object]]]:
    extra_pids = extra_pids or set()
    exclude_inodes = exclude_inodes or set()
    processes: dict[int, tuple[str, list[str]]] = {}
    socket_owners: dict[str, list[tuple[int, str]]] = {}
    inspected_extra_pids: set[int] = set()
    for process_dir in PROC_ROOT.iterdir():
        if not process_dir.name.isdigit():
            continue
        pid = int(process_dir.name)
        try:
            args = [
                part
                for part in (process_dir / "cmdline").read_bytes().split(b"\0")
                if part
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError) as error:
            if pid in extra_pids:
                raise RuntimeError(
                    f"explicit Ray PID {pid} cmdline is not inspectable"
                ) from error
            continue
        if pid not in extra_pids and not _is_ray_process(args):
            continue
        label = (
            args[0].rsplit(b"/", 1)[-1].decode("utf-8", errors="replace")
            if args
            else "<empty-cmdline>"
        )
        display_args = [
            arg.decode("utf-8", errors="replace")[:256] for arg in args[:12]
        ]
        processes[pid] = (label, display_args)
        try:
            fd_entries = list((process_dir / "fd").iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError) as error:
            if pid in extra_pids:
                raise RuntimeError(
                    f"explicit Ray PID {pid} fd inventory is not inspectable"
                ) from error
            continue
        if pid in extra_pids:
            inspected_extra_pids.add(pid)
        for fd_entry in fd_entries:
            try:
                target = str(fd_entry.readlink())
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if target.startswith("socket:[") and target.endswith("]"):
                inode = target[len("socket:[") : -1]
                socket_owners.setdefault(inode, []).append((pid, label))

    missing_extra_pids = extra_pids - inspected_extra_pids
    if missing_extra_pids:
        raise RuntimeError(
            "explicit Ray PIDs were not inspected: "
            + ", ".join(str(pid) for pid in sorted(missing_extra_pids))
        )
    if not processes:
        raise RuntimeError("no Ray processes found while checking listener ownership")

    listeners_by_inode = _tcp_listener_table()
    owned: list[dict[str, object]] = []
    for inode, owners in socket_owners.items():
        if inode in exclude_inodes:
            continue
        listener = listeners_by_inode.get(inode)
        if listener is None:
            continue
        owned.append(
            {
                **listener,
                "owners": [
                    {
                        "pid": pid,
                        "process": label,
                        "args": processes[pid][1],
                    }
                    for pid, label in sorted(owners)
                ],
            }
        )
    if not owned:
        raise RuntimeError(
            "Ray processes found but no owned TCP listeners were inspectable"
        )
    return len(processes), sorted(
        owned, key=lambda item: (int(item["port"]), str(item["inode"]))
    )


def _current_process_listener_inodes() -> set[str]:
    """Snapshot listening sockets inherited before this process starts Ray."""
    process_fd_dir = PROC_ROOT / str(os.getpid()) / "fd"
    try:
        fd_entries = list(process_fd_dir.iterdir())
    except (FileNotFoundError, PermissionError, ProcessLookupError) as error:
        raise RuntimeError("validator fd inventory is not inspectable") from error

    current_inodes: set[str] = set()
    for fd_entry in fd_entries:
        try:
            target = str(fd_entry.readlink())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            current_inodes.add(target[len("socket:[") : -1])
    return current_inodes.intersection(_tcp_listener_table())


def _non_loopback_control_listeners() -> list[dict[str, object]]:
    control_ports = set(RAY_CONTROL_PORTS)
    return [
        listener
        for listener in _tcp_listener_table().values()
        if int(listener["port"]) in control_ports
        and not _is_expected_ray_loopback(str(listener["address"]))
    ]


def _start_probe_actor() -> tuple[Any, Any, int]:
    import ray

    ray.init(
        address=f"{RAY_LOOPBACK_HOST}:1200",
        _node_ip_address=RAY_LOOPBACK_HOST,
    )

    @ray.remote(num_cpus=0)
    class _ListenerProbeActor:
        def ping(self, value: str) -> tuple[int, str]:
            return os.getpid(), value

    actor = _ListenerProbeActor.remote()
    actor_pid, value = ray.get(actor.ping.remote("before-unauthenticated-probe"))
    if value != "before-unauthenticated-probe":
        raise RuntimeError("authenticated Ray actor returned an invalid probe value")
    time.sleep(0.5)
    return ray, actor, int(actor_pid)


def _validate_auth_token_file() -> tuple[Path, bytes, str]:
    from ray._raylet import AuthenticationMode, get_authentication_mode

    if get_authentication_mode() != AuthenticationMode.TOKEN:
        raise RuntimeError("Ray token authentication is not active in the validator")
    if "RAY_AUTH_TOKEN" in os.environ:
        raise RuntimeError("raw RAY_AUTH_TOKEN must not be present")

    raw_path = os.environ.get("RAY_AUTH_TOKEN_PATH", "")
    if not raw_path:
        raise RuntimeError("RAY_AUTH_TOKEN_PATH is required")
    token_path = Path(raw_path)
    token_stat = token_path.lstat()
    if stat.S_ISLNK(token_stat.st_mode) or not stat.S_ISREG(token_stat.st_mode):
        raise RuntimeError("Ray auth token path must be a non-symlink regular file")
    if token_stat.st_uid != os.getuid():
        raise RuntimeError("Ray auth token file is not owned by the validator user")
    mode = stat.S_IMODE(token_stat.st_mode)
    if mode != 0o600:
        raise RuntimeError(f"Ray auth token file mode is {mode:04o}, expected 0600")

    token = token_path.read_bytes()
    lowercase_hex = b"0123456789abcdef"
    if len(token) != TOKEN_BYTES or any(byte not in lowercase_hex for byte in token):
        raise RuntimeError("Ray auth token must contain exactly 64 lowercase hex bytes")
    return token_path.resolve(strict=True), token, f"{mode:04o}"


def _scan_token_in_tree(
    token_path: Path,
    token: bytes,
    scan_root: Path,
) -> dict[str, int]:
    """Scan regular files without ever returning or printing the secret bytes."""
    if not scan_root.is_dir():
        raise RuntimeError(f"Ray log scan root is not a directory: {scan_root}")

    resolved_token = token_path.resolve(strict=True)
    files_scanned = 0
    bytes_scanned = 0
    leaking_files: list[str] = []
    overlap_size = max(0, len(token) - 1)
    for candidate in sorted(scan_root.rglob("*")):
        try:
            candidate_stat = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISREG(
            candidate_stat.st_mode
        ):
            continue
        try:
            if candidate.resolve(strict=True) == resolved_token:
                continue
        except FileNotFoundError:
            continue
        files_scanned += 1
        previous = b""
        found = False
        try:
            with candidate.open("rb") as stream:
                while True:
                    chunk = stream.read(SCAN_CHUNK_BYTES)
                    if not chunk:
                        break
                    bytes_scanned += len(chunk)
                    combined = previous + chunk
                    if token in combined:
                        found = True
                        break
                    previous = combined[-overlap_size:] if overlap_size else b""
        except (FileNotFoundError, PermissionError):
            continue
        if found:
            leaking_files.append(str(candidate))

    if leaking_files:
        raise RuntimeError(
            "Ray auth token leaked into log files: " + ", ".join(leaking_files)
        )
    return {
        "files_scanned": files_scanned,
        "bytes_scanned": bytes_scanned,
        "token_log_leaks": 0,
    }


def _run_unauthenticated_probe() -> tuple[bool, int, str]:
    child_code = """
import ray
ray.init(address="127.0.0.1:1200", _node_ip_address="127.0.0.1")
ray.get(ray.put("unauthenticated-probe"))
"""
    child_env = os.environ.copy()
    child_env["RAY_AUTH_MODE"] = "disabled"
    child_env.pop("RAY_AUTH_TOKEN", None)
    child_env.pop("RAY_AUTH_TOKEN_PATH", None)
    with tempfile.TemporaryDirectory(prefix="sciprobe-ray-unauth-home-") as home:
        child_env["HOME"] = home
        process = subprocess.Popen(
            [sys.executable, "-c", child_code],
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            child_returncode = process.wait(timeout=20)
            subprocess_outcome = "nonzero"
        except subprocess.TimeoutExpired:
            # Ray retries an unauthenticated GCS connection instead of returning
            # the gRPC denial promptly. Bound the attempt, kill its entire fresh
            # process group, and reap the child before checking HTTP auth and the
            # still-authenticated actor. A timeout alone is never sufficient.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child_returncode = process.wait(timeout=5)
            subprocess_outcome = "timeout"
    if subprocess_outcome == "nonzero" and child_returncode == 0:
        raise RuntimeError("unauthenticated Ray subprocess connected successfully")
    if child_returncode == 0:
        raise RuntimeError("unauthenticated Ray subprocess was not denied")

    request = urllib.request.Request("http://127.0.0.1:8265/api/version")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=5) as response:
            dashboard_status = int(response.status)
    except urllib.error.HTTPError as error:
        dashboard_status = int(error.code)
    except urllib.error.URLError as error:
        raise RuntimeError(
            "unauthenticated dashboard probe could not connect"
        ) from error
    if dashboard_status != 401:
        raise RuntimeError(
            "unauthenticated dashboard request returned "
            f"HTTP {dashboard_status}, expected 401"
        )
    return True, dashboard_status, subprocess_outcome


def _probe_driver_core_worker_auth(
    *,
    target: str,
    intended_worker_id: bytes,
    token: bytes,
) -> bool:
    """Prove the exact advertised driver CoreWorker RPC rejects a missing token."""
    import grpc
    from ray._private.authentication.authentication_constants import (
        AUTHORIZATION_BEARER_PREFIX,
        AUTHORIZATION_HEADER_NAME,
    )
    from ray.core.generated import core_worker_pb2, core_worker_pb2_grpc

    channel = grpc.insecure_channel(
        target,
        options=(("grpc.enable_http_proxy", 0),),
    )
    request = core_worker_pb2.GetCoreWorkerStatsRequest(
        intended_worker_id=intended_worker_id,
        include_memory_info=False,
        include_task_info=False,
    )
    stub = core_worker_pb2_grpc.CoreWorkerServiceStub(channel)
    try:
        grpc.channel_ready_future(channel).result(timeout=5)

        # First prove that the discovered port is this driver's live CoreWorker
        # endpoint. Keep the token only in call metadata; never print it.
        try:
            authenticated_reply = stub.GetCoreWorkerStats(
                request,
                timeout=5,
                metadata=(
                    (
                        AUTHORIZATION_HEADER_NAME,
                        AUTHORIZATION_BEARER_PREFIX + token.decode("ascii"),
                    ),
                ),
            )
        except grpc.RpcError as error:
            raise RuntimeError(
                "authenticated driver CoreWorker gRPC probe failed with "
                f"{error.code().name}"
            ) from error
        authenticated_stats = authenticated_reply.core_worker_stats
        if (
            int(authenticated_stats.pid) != os.getpid()
            or bytes(authenticated_stats.worker_id) != intended_worker_id
        ):
            raise RuntimeError(
                "authenticated CoreWorker response did not identify this driver"
            )

        # Use the same generated stub and request without metadata. Only the
        # explicit gRPC authentication status is acceptable; a timeout or a
        # handler-level error is not evidence that token auth protected it.
        try:
            stub.GetCoreWorkerStats(request, timeout=5)
        except grpc.RpcError as error:
            status = error.code()
        else:
            raise RuntimeError(
                "unauthenticated driver CoreWorker gRPC request succeeded"
            )
        if status != grpc.StatusCode.UNAUTHENTICATED:
            raise RuntimeError(
                "unauthenticated driver CoreWorker gRPC request returned "
                f"{status.name}, expected UNAUTHENTICATED"
            )
    finally:
        channel.close()
    return True


def _ray_runtime_diagnostics(
    ray_module: Any, probe_actor_pid: int
) -> dict[str, object]:
    """Return the exact addresses Ray assigned to this driver and its probe actor."""
    import ray._private.state as state_module
    from ray._common.network_utils import get_localhost_ip
    from ray.core.generated import common_pb2, gcs_pb2

    target_pids = {os.getpid(), probe_actor_pid}
    worker_rows: list[dict[str, object]] = []
    accessor = state_module.state._connect_and_get_accessor()
    for serialized in accessor.get_worker_table():
        row = gcs_pb2.WorkerTableData.FromString(serialized)
        if int(row.pid) not in target_pids:
            continue
        worker_rows.append(
            {
                "pid": int(row.pid),
                "worker_type": int(row.worker_type),
                "is_alive": bool(row.is_alive),
                "advertised_ip": row.worker_address.ip_address,
                "advertised_port": int(row.worker_address.port),
                "debugger_port": int(row.debugger_port)
                if row.HasField("debugger_port")
                else None,
            }
        )

    raylet_cmdlines: list[list[str]] = []
    for process_dir in PROC_ROOT.iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            args = [
                part
                for part in (process_dir / "cmdline").read_bytes().split(b"\0")
                if part
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not any(arg.rsplit(b"/", 1)[-1] == b"raylet" for arg in args):
            continue
        raylet_cmdlines.append(
            [arg.decode("utf-8", errors="replace")[:512] for arg in args]
        )

    localhost_addresses = sorted(
        {info[4][0] for info in socket.getaddrinfo("localhost", None, socket.AF_UNSPEC)}
    )
    owned_ref = ray_module.put("sciprobe-ray-loopback-diagnostic")
    owner_address = common_pb2.Address.FromString(
        ray_module._private.worker.global_worker.core_worker.get_owner_address(
            owned_ref
        )
    )
    global_node = ray_module._private.worker._global_node
    return {
        "ray_version": ray_module.__version__,
        "ray_commit": ray_module.__commit__,
        "enable_ray_cluster": bool(
            ray_module._private.ray_constants.ENABLE_RAY_CLUSTER
        ),
        "requested_node_ip": RAY_LOOPBACK_HOST,
        "global_node_ip": global_node.node_ip_address,
        "global_worker_node_ip": ray_module._private.worker.global_worker.node_ip_address,
        "python_localhost_ip": get_localhost_ip(),
        "localhost_addresses": localhost_addresses,
        "driver_owner_address": {
            "ip": owner_address.ip_address,
            "port": int(owner_address.port),
        },
        "worker_rows": sorted(worker_rows, key=lambda row: int(row["pid"])),
        "raylet_cmdlines": raylet_cmdlines,
    }


def main(
    *,
    require_worker: bool = False,
    require_token_auth: bool = False,
    ray_log_dir: Path | None = None,
) -> None:
    ray_module = None
    probe_actor = None
    probe_actor_pid = None
    ray_diagnostics: dict[str, object] | None = None
    authenticated_actor_roundtrip = False
    unauthenticated_connection_rejected = False
    unauthenticated_dashboard_status: int | None = None
    unauthenticated_subprocess_outcome: str | None = None
    token_file_mode: str | None = None
    token_log_scan = {"files_scanned": 0, "bytes_scanned": 0, "token_log_leaks": 0}
    token_path: Path | None = None
    token: bytes | None = None
    inherited_listener_inodes = _current_process_listener_inodes()
    driver_core_worker_rpc_target: str | None = None
    authenticated_driver_rpc_roundtrip = False
    unauthenticated_driver_rpc_status: str | None = None
    try:
        if require_token_auth:
            if not require_worker:
                raise RuntimeError(
                    "--require-token-auth also requires --require-worker"
                )
            if ray_log_dir is None:
                raise RuntimeError(
                    "--ray-log-dir is required with --require-token-auth"
                )
            token_path, token, token_file_mode = _validate_auth_token_file()

        if require_worker:
            ray_module, probe_actor, probe_actor_pid = _start_probe_actor()
            authenticated_actor_roundtrip = True
            ray_diagnostics = _ray_runtime_diagnostics(ray_module, probe_actor_pid)
            print(
                "RAY_LOOPBACK_DIAGNOSTIC "
                + json.dumps(ray_diagnostics, sort_keys=True),
                flush=True,
            )
            if require_token_auth:
                assert token is not None
                driver_address = ray_diagnostics["driver_owner_address"]
                if not isinstance(driver_address, dict):
                    raise RuntimeError("driver CoreWorker address is unavailable")
                driver_ip = str(driver_address["ip"])
                driver_port = int(driver_address["port"])
                if not _is_expected_ray_loopback(driver_ip):
                    raise RuntimeError(
                        "driver CoreWorker advertised a non-loopback address: "
                        f"{driver_ip}:{driver_port}"
                    )
                driver_core_worker_rpc_target = f"{driver_ip}:{driver_port}"
                driver_worker_id = ray_module._private.worker.global_worker.worker_id
                if not isinstance(driver_worker_id, bytes):
                    driver_worker_id = driver_worker_id.binary()
                authenticated_driver_rpc_roundtrip = _probe_driver_core_worker_auth(
                    target=driver_core_worker_rpc_target,
                    intended_worker_id=driver_worker_id,
                    token=token,
                )
                unauthenticated_driver_rpc_status = "UNAUTHENTICATED"

        node_addresses = _non_loopback_ipv4_addresses()
        explicit_ray_pids = {os.getpid()}
        if probe_actor_pid is not None:
            explicit_ray_pids.add(probe_actor_pid)
        ray_process_count, ray_listeners = _ray_owned_tcp_listeners(
            extra_pids=explicit_ray_pids,
            exclude_inodes=inherited_listener_inodes,
        )
        for listener in ray_listeners:
            for owner in listener["owners"]:
                if owner["pid"] == os.getpid():
                    owner["role"] = "validator-driver"
                elif owner["pid"] == probe_actor_pid:
                    owner["role"] = "probe-actor"
        proc_exposed = [
            listener
            for listener in ray_listeners
            if not _is_expected_ray_loopback(str(listener["address"]))
        ]
        if proc_exposed:
            raise RuntimeError(
                "Ray control-plane listener is not loopback-only: "
                + json.dumps(
                    {
                        "exposed": proc_exposed,
                        "ray_runtime": ray_diagnostics,
                    },
                    sort_keys=True,
                )
            )

        known_port_exposed = _non_loopback_control_listeners()
        if known_port_exposed:
            raise RuntimeError(
                "known Ray control-plane listener is not loopback-only: "
                + json.dumps(known_port_exposed, sort_keys=True)
            )

        exposed = [
            f"{address}:{port}"
            for address in node_addresses
            for port in RAY_CONTROL_PORTS
            if _connects(address, port)
        ]
        if exposed:
            raise RuntimeError(
                "Ray control-plane port reachable through node network: "
                + ", ".join(exposed)
            )

        required_loopback = [1200, 8265]
        missing_loopback = [
            port
            for port in required_loopback
            if not _connects(RAY_LOOPBACK_HOST, port, 0.25)
        ]
        if missing_loopback:
            raise RuntimeError(
                f"expected Ray loopback listeners are unavailable: {missing_loopback}"
            )

        if require_token_auth:
            (
                unauthenticated_connection_rejected,
                unauthenticated_dashboard_status,
                unauthenticated_subprocess_outcome,
            ) = _run_unauthenticated_probe()
            assert ray_module is not None and probe_actor is not None
            _, post_value = ray_module.get(
                probe_actor.ping.remote("after-unauthenticated-probe")
            )
            if post_value != "after-unauthenticated-probe":
                raise RuntimeError(
                    "authenticated Ray actor failed after negative probe"
                )
            assert (
                token_path is not None and token is not None and ray_log_dir is not None
            )
            token_log_scan = _scan_token_in_tree(token_path, token, ray_log_dir)

        print(
            json.dumps(
                {
                    "status": "ok",
                    "node_addresses": node_addresses,
                    "ports_checked": len(RAY_CONTROL_PORTS),
                    "node_ip_exposures": 0,
                    "proc_non_loopback_listeners": len(proc_exposed),
                    "known_port_non_loopback_listeners": 0,
                    "probe_actor_pid": probe_actor_pid,
                    "ray_process_count": ray_process_count,
                    "ray_tcp_listener_count": len(ray_listeners),
                    "required_loopback_ports": required_loopback,
                    "ray_loopback_host": RAY_LOOPBACK_HOST,
                    "ray_auth_mode": "token" if require_token_auth else "disabled",
                    "token_file_mode": token_file_mode,
                    "token_log_leaks": token_log_scan["token_log_leaks"],
                    "token_log_files_scanned": token_log_scan["files_scanned"],
                    "token_log_bytes_scanned": token_log_scan["bytes_scanned"],
                    "unauthenticated_connection_rejected": (
                        unauthenticated_connection_rejected
                    ),
                    "unauthenticated_dashboard_status": (
                        unauthenticated_dashboard_status
                    ),
                    "unauthenticated_subprocess_outcome": (
                        unauthenticated_subprocess_outcome
                    ),
                    "authenticated_actor_roundtrip": authenticated_actor_roundtrip,
                    "inherited_listener_inodes_excluded": len(
                        inherited_listener_inodes
                    ),
                    "driver_core_worker_rpc_target": driver_core_worker_rpc_target,
                    "authenticated_driver_rpc_roundtrip": (
                        authenticated_driver_rpc_roundtrip
                    ),
                    "unauthenticated_driver_rpc_status": (
                        unauthenticated_driver_rpc_status
                    ),
                },
                sort_keys=True,
            )
        )
    finally:
        if ray_module is not None:
            if probe_actor is not None:
                ray_module.kill(probe_actor, no_restart=True)
            ray_module.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-worker", action="store_true")
    parser.add_argument("--require-token-auth", action="store_true")
    parser.add_argument("--ray-log-dir", type=Path)
    args = parser.parse_args()
    main(
        require_worker=args.require_worker,
        require_token_auth=args.require_token_auth,
        ray_log_dir=args.ray_log_dir,
    )
