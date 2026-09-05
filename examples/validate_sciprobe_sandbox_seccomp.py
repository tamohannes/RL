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

"""Validate SciProbe's shell-child seccomp gate in an isolated interpreter."""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import io
import json
import multiprocessing
import os
import pickle
import platform
import resource
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

_EXPECTED_ERRNO = errno.ENETUNREACH
_REQUIRED_ENV = "SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK"
_HOOK_LOADED_ENV = "SCIPROBE_SECCOMP_HOOK_LOADED"
_FILTER_ACTIVE_ENV = "SCIPROBE_SECCOMP_FILTER_ACTIVE"
_PROCESS_FILTER_ACTIVE_ENV = "SCIPROBE_SECCOMP_PROCESS_FILTER_ACTIVE"
_INHERITED_FDS_CLEAN_ENV = "SCIPROBE_SECCOMP_INHERITED_FDS_CLEAN"
_RESTRICTED_UNPICKLER_ENV = "SCIPROBE_RESTRICTED_MP_UNPICKLER_ACTIVE"
_LANDLOCK_ACTIVE_ENV = "SCIPROBE_LANDLOCK_FILTER_ACTIVE"
_LANDLOCK_EFFECTIVE_ABI_ENV = "SCIPROBE_LANDLOCK_EFFECTIVE_ABI"
_LANDLOCK_MISSING_CONTROLS_ENV = "SCIPROBE_LANDLOCK_MISSING_CONTROLS"
_SESSION_STATE_ENV = "SCIPROBE_SESSION_STATE_DIR"
_SESSION_VAR_TMP_ENV = "SCIPROBE_SESSION_VAR_TMP_DIR"
_SESSION_SHM_ENV = "SCIPROBE_SESSION_SHM_DIR"
_READONLY_PATHS_ENV = "SCIPROBE_SANDBOX_READONLY_PATHS"
_MAX_CONTROL_MESSAGE_BYTES = 16 * 1024 * 1024
_SOL_SOCKET = 1
_SO_DOMAIN = 39
_IPC_PRIVATE = 0
_IPC_CREAT = 0o1000
_IPC_RMID = 0
_SHM_RDONLY = 0o10000
_IOPRIO_WHO_PROCESS = 1

_X86_64_SYSCALLS = {
    "shmget": 29,
    "shmat": 30,
    "shmctl": 31,
    "socket": 41,
    "connect": 42,
    "accept": 43,
    "sendto": 44,
    "sendmsg": 46,
    "bind": 49,
    "listen": 50,
    "socketpair": 53,
    "accept4": 288,
    "sendmmsg": 307,
    "io_uring_setup": 425,
    "clone": 56,
    "fork": 57,
    "vfork": 58,
    "execve": 59,
    "kill": 62,
    "semget": 64,
    "semop": 65,
    "semctl": 66,
    "shmdt": 67,
    "msgget": 68,
    "msgsnd": 69,
    "msgrcv": 70,
    "msgctl": 71,
    "ptrace": 101,
    "rt_sigqueueinfo": 129,
    "getpriority": 140,
    "setpriority": 141,
    "sched_setparam": 142,
    "sched_getparam": 143,
    "sched_setscheduler": 144,
    "sched_getscheduler": 145,
    "sched_rr_get_interval": 148,
    "tkill": 200,
    "sched_setaffinity": 203,
    "sched_getaffinity": 204,
    "semtimedop": 220,
    "tgkill": 234,
    "ioprio_set": 251,
    "ioprio_get": 252,
    "rt_tgsigqueueinfo": 297,
    "prlimit64": 302,
    "process_vm_readv": 310,
    "process_vm_writev": 311,
    "execveat": 322,
    "sched_setattr": 314,
    "sched_getattr": 315,
    "pidfd_send_signal": 424,
    "pidfd_open": 434,
    "clone3": 435,
    "pidfd_getfd": 438,
    "process_madvise": 440,
    "process_mrelease": 448,
}
_AARCH64_SYSCALLS = {
    "ioprio_set": 30,
    "ioprio_get": 31,
    "socket": 198,
    "socketpair": 199,
    "bind": 200,
    "listen": 201,
    "accept": 202,
    "connect": 203,
    "sendto": 206,
    "sendmsg": 211,
    "accept4": 242,
    "sendmmsg": 269,
    "io_uring_setup": 425,
    "clone": 220,
    "execve": 221,
    "ptrace": 117,
    "sched_setparam": 118,
    "sched_setscheduler": 119,
    "sched_getscheduler": 120,
    "sched_getparam": 121,
    "sched_setaffinity": 122,
    "sched_getaffinity": 123,
    "sched_rr_get_interval": 127,
    "kill": 129,
    "tkill": 130,
    "tgkill": 131,
    "rt_sigqueueinfo": 138,
    "setpriority": 140,
    "getpriority": 141,
    "msgget": 186,
    "msgctl": 187,
    "msgrcv": 188,
    "msgsnd": 189,
    "semget": 190,
    "semctl": 191,
    "semtimedop": 192,
    "semop": 193,
    "shmget": 194,
    "shmctl": 195,
    "shmat": 196,
    "shmdt": 197,
    "rt_tgsigqueueinfo": 240,
    "prlimit64": 261,
    "process_vm_readv": 270,
    "process_vm_writev": 271,
    "sched_setattr": 274,
    "sched_getattr": 275,
    "execveat": 281,
    "pidfd_send_signal": 424,
    "pidfd_open": 434,
    "clone3": 435,
    "pidfd_getfd": 438,
    "process_madvise": 440,
    "process_mrelease": 448,
}
_SYSCALLS = {
    "x86_64": _X86_64_SYSCALLS,
    "amd64": _X86_64_SYSCALLS,
    "aarch64": _AARCH64_SYSCALLS,
    "arm64": _AARCH64_SYSCALLS,
}


class _SockaddrUn(ctypes.Structure):
    _fields_ = [("sun_family", ctypes.c_ushort), ("sun_path", ctypes.c_char * 108)]


def _write_exploit_marker(path: str) -> str:
    Path(path).write_text("executed", encoding="utf-8")
    return "executed"


class _ReducerPayload:
    def __init__(self, marker_path: str) -> None:
        self.marker_path = marker_path

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return _write_exploit_marker, (self.marker_path,)


def _control_payload_rejected(payload: object, *, raw: bool = False) -> bool:
    receiver, sender = multiprocessing.Pipe(duplex=False)
    try:
        if raw:
            sender.send_bytes(payload)
        else:
            sender.send(payload)
        try:
            receiver.recv()
        except pickle.UnpicklingError:
            return receiver.closed
        return False
    finally:
        receiver.close()
        sender.close()


def _oversize_control_payload_rejected() -> bool:
    receiver, sender = multiprocessing.Pipe(duplex=True)
    try:
        header = struct.pack("!i", _MAX_CONTROL_MESSAGE_BYTES + 1)
        if os.write(sender.fileno(), header) != len(header):
            return False
        try:
            receiver.recv()
        except OSError:
            return receiver.closed
        return False
    finally:
        receiver.close()
        sender.close()


def _expect_network_denied(call: Callable[[], Any]) -> bool:
    try:
        value = call()
    except OSError as error:
        return error.errno == _EXPECTED_ERRNO
    if hasattr(value, "close"):
        value.close()
    return False


def _direct_syscall_denied(syscall_number: int, *arguments: object) -> bool:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    ctypes.set_errno(0)
    result = libc.syscall(ctypes.c_long(syscall_number), *arguments)
    error = ctypes.get_errno()
    return result == -1 and error == _EXPECTED_ERRNO


def _direct_syscall_has_errno(
    syscall_number: int, expected_errno: int, *arguments: object
) -> bool:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    ctypes.set_errno(0)
    result = libc.syscall(ctypes.c_long(syscall_number), *arguments)
    return result == -1 and ctypes.get_errno() == expected_errno


def _direct_syscall_value(syscall_number: int, *arguments: object) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    ctypes.set_errno(0)
    result = int(libc.syscall(ctypes.c_long(syscall_number), *arguments))
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def _subprocess_creation_denied(arguments: list[str]) -> bool:
    try:
        subprocess.run(
            arguments,
            env={},
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError as error:
        return error.errno == errno.EPERM
    return False


def _unix_sockaddr(address: str | bytes) -> tuple[_SockaddrUn, int]:
    encoded = os.fsencode(address) if isinstance(address, str) else address
    if len(encoded) >= 108:
        raise ValueError("AF_UNIX address is too long")
    sockaddr = _SockaddrUn(sun_family=socket.AF_UNIX)
    sockaddr.sun_path = encoded
    terminator = 0 if encoded.startswith(b"\0") else 1
    return sockaddr, 2 + len(encoded) + terminator


def _socket_domain(libc: ctypes.CDLL, descriptor: int) -> int | None:
    domain = ctypes.c_int()
    size = ctypes.c_uint(ctypes.sizeof(domain))
    ctypes.set_errno(0)
    result = libc.getsockopt(
        descriptor,
        _SOL_SOCKET,
        _SO_DOMAIN,
        ctypes.byref(domain),
        ctypes.byref(size),
    )
    if result == 0:
        return int(domain.value)
    if ctypes.get_errno() in {errno.EBADF, errno.ENOTSOCK}:
        return None
    raise OSError(ctypes.get_errno(), "getsockopt(SO_DOMAIN) failed")


def _socket_descriptors() -> list[dict[str, int]]:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.getsockopt.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
    ]
    libc.getsockopt.restype = ctypes.c_int
    found: list[dict[str, int]] = []
    for entry in os.listdir("/proc/self/fd"):
        try:
            descriptor = int(entry)
        except ValueError:
            continue
        if descriptor <= 2:
            continue
        domain = _socket_domain(libc, descriptor)
        if domain is not None:
            found.append({"fd": descriptor, "domain": domain})
    return found


def _seccomp_mode() -> int | None:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition(":")
        if separator and name == "Seccomp":
            return int(value.strip())
    return None


def _nested_ipc_worker(connection: multiprocessing.connection.Connection) -> None:
    connection.send({"answer": 6 * 7})
    connection.close()


def _filesystem_call_denied(call: Callable[[], Any]) -> bool:
    try:
        value = call()
    except OSError as error:
        return error.errno in {errno.EACCES, errno.EPERM, errno.EXDEV}
    if hasattr(value, "close"):
        value.close()
    return False


def _signal_zero_denied(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except OSError as error:
        return error.errno == errno.EPERM
    return False


def _prlimit_denied(
    process_id: int,
    limits: tuple[int, int] | None = None,
) -> bool:
    try:
        if limits is None:
            resource.prlimit(process_id, resource.RLIMIT_NOFILE)
        else:
            resource.prlimit(process_id, resource.RLIMIT_NOFILE, limits)
    except OSError as error:
        return error.errno == errno.EPERM
    return False


def _create_sysv_shm_secret(secret: bytes) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.shmget.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_int]
    libc.shmget.restype = ctypes.c_int
    libc.shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    libc.shmat.restype = ctypes.c_void_p
    libc.shmdt.argtypes = [ctypes.c_void_p]
    libc.shmdt.restype = ctypes.c_int
    identifier = libc.shmget(
        _IPC_PRIVATE,
        max(len(secret), 1),
        _IPC_CREAT | 0o600,
    )
    if identifier < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    address = libc.shmat(identifier, None, 0)
    if address == ctypes.c_void_p(-1).value:
        error = ctypes.get_errno()
        _remove_sysv_shm(identifier)
        raise OSError(error, os.strerror(error))
    try:
        ctypes.memmove(address, secret, len(secret))
    finally:
        if libc.shmdt(ctypes.c_void_p(address)) != 0:
            error = ctypes.get_errno()
            _remove_sysv_shm(identifier)
            raise OSError(error, os.strerror(error))
    return identifier


def _remove_sysv_shm(identifier: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.shmctl.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    libc.shmctl.restype = ctypes.c_int
    if libc.shmctl(identifier, _IPC_RMID, None) != 0:
        error = ctypes.get_errno()
        if error not in {errno.EIDRM, errno.EINVAL}:
            raise OSError(error, os.strerror(error))


def _sysv_shm_attach_read_denied(identifier: int, secret_size: int) -> bool:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    libc.shmat.restype = ctypes.c_void_p
    libc.shmdt.argtypes = [ctypes.c_void_p]
    libc.shmdt.restype = ctypes.c_int
    ctypes.set_errno(0)
    address = libc.shmat(identifier, None, _SHM_RDONLY)
    if address == ctypes.c_void_p(-1).value:
        return ctypes.get_errno() == errno.EPERM
    try:
        ctypes.string_at(address, secret_size)
    finally:
        libc.shmdt(ctypes.c_void_p(address))
    return False


def _session_writer(connection: multiprocessing.connection.Connection) -> None:
    state_directory = os.environ[_SESSION_STATE_ENV]
    secret_path = str(Path(state_directory) / "sibling-secret.txt")
    Path(secret_path).write_text("writer-secret", encoding="utf-8")
    connection.send(
        {
            "landlock_active": os.environ.get(_LANDLOCK_ACTIVE_ENV) == "1",
            "state_directory": state_directory,
            "var_tmp_directory": os.environ[_SESSION_VAR_TMP_ENV],
            "shm_directory": os.environ[_SESSION_SHM_ENV],
            "secret_path": secret_path,
        }
    )
    command = connection.recv()
    connection.send(
        {
            "stop_received": command == "stop",
            "secret_intact": (
                Path(secret_path).read_text(encoding="utf-8") == "writer-secret"
            ),
        }
    )
    connection.close()


def _session_reader(
    connection: multiprocessing.connection.Connection,
    writer_info: dict[str, str | bool],
    readonly_probe_path: str,
    global_sentinels: dict[str, str],
    global_create_paths: dict[str, str],
    inherited_global_fd: int,
    inherited_secret_pipe_fd: int,
    parent_sysv_shm_id: int,
    parent_sysv_secret_size: int,
    writer_pid: int,
    writer_nofile_limits: tuple[int, int],
    writer_affinity: tuple[int, ...],
    writer_nice: int,
    writer_ioprio: int,
    trusted_parent_pid: int,
) -> None:
    result: dict[str, Any] = {}
    try:
        state_directory = os.environ[_SESSION_STATE_ENV]
        var_tmp_directory = os.environ[_SESSION_VAR_TMP_ENV]
        shm_directory = os.environ[_SESSION_SHM_ENV]
        result.update(
            {
                "landlock_active": (os.environ.get(_LANDLOCK_ACTIVE_ENV) == "1"),
                "state_directory": state_directory,
                "var_tmp_directory": var_tmp_directory,
                "shm_directory": shm_directory,
                "home_is_private": os.environ.get("HOME")
                == str(Path(state_directory) / "home"),
                "tmpdir_is_private": os.environ.get("TMPDIR")
                == str(Path(state_directory) / "tmp"),
            }
        )

        writer_state = str(writer_info["state_directory"])
        writer_secret = str(writer_info["secret_path"])
        result["sibling_read_denied"] = _filesystem_call_denied(
            lambda: Path(writer_secret).read_text(encoding="utf-8")
        )
        result["sibling_write_denied"] = _filesystem_call_denied(
            lambda: Path(writer_secret).write_text("stolen", encoding="utf-8")
        )
        result["sibling_list_denied"] = _filesystem_call_denied(
            lambda: os.listdir(writer_state)
        )
        result["sibling_delete_denied"] = _filesystem_call_denied(
            lambda: os.unlink(writer_secret)
        )
        result["sibling_signal_zero_denied"] = _signal_zero_denied(writer_pid)
        result["all_processes_signal_zero_denied"] = _signal_zero_denied(-1)
        result["sibling_prlimit_query_denied"] = _prlimit_denied(writer_pid)
        result["sibling_prlimit_mutation_denied"] = _prlimit_denied(
            writer_pid, writer_nofile_limits
        )
        try:
            os.sched_getaffinity(writer_pid)
        except OSError as error:
            result["sibling_affinity_query_denied"] = error.errno == errno.EPERM
        else:
            result["sibling_affinity_query_denied"] = False
        try:
            os.sched_setaffinity(writer_pid, set(writer_affinity))
        except OSError as error:
            result["sibling_affinity_mutation_denied"] = error.errno == errno.EPERM
        else:
            result["sibling_affinity_mutation_denied"] = False
        try:
            os.getpriority(os.PRIO_PROCESS, writer_pid)
        except OSError as error:
            result["sibling_priority_query_denied"] = error.errno == errno.EPERM
        else:
            result["sibling_priority_query_denied"] = False
        try:
            os.setpriority(os.PRIO_PROCESS, writer_pid, writer_nice)
        except OSError as error:
            result["sibling_priority_mutation_denied"] = error.errno == errno.EPERM
        else:
            result["sibling_priority_mutation_denied"] = False
        machine = platform.machine().lower()
        syscalls = _SYSCALLS[machine]
        result["sibling_ioprio_query_denied"] = _direct_syscall_has_errno(
            syscalls["ioprio_get"],
            errno.EPERM,
            ctypes.c_long(_IOPRIO_WHO_PROCESS),
            ctypes.c_long(writer_pid),
        )
        result["sibling_ioprio_mutation_denied"] = _direct_syscall_has_errno(
            syscalls["ioprio_set"],
            errno.EPERM,
            ctypes.c_long(_IOPRIO_WHO_PROCESS),
            ctypes.c_long(writer_pid),
            ctypes.c_long(writer_ioprio),
        )
        try:
            os.read(inherited_secret_pipe_fd, 64)
        except OSError as error:
            result["parent_secret_pipe_fd_closed"] = error.errno == errno.EBADF
        else:
            result["parent_secret_pipe_fd_closed"] = False
        result["parent_sysv_shm_attach_read_denied"] = _sysv_shm_attach_read_denied(
            parent_sysv_shm_id,
            parent_sysv_secret_size,
        )

        for label, sentinel_path in global_sentinels.items():
            result[f"global_{label}_read_denied"] = _filesystem_call_denied(
                lambda path=sentinel_path: Path(path).read_text(encoding="utf-8")
            )
            result[f"global_{label}_write_denied"] = _filesystem_call_denied(
                lambda path=sentinel_path: Path(path).write_text(
                    "stolen", encoding="utf-8"
                )
            )
        for label, create_path in global_create_paths.items():
            result[f"global_{label}_create_denied"] = _filesystem_call_denied(
                lambda path=create_path: Path(path).write_text(
                    "forbidden", encoding="utf-8"
                )
            )
        for label, root in (
            ("tmp", "/tmp"),
            ("var_tmp", "/var/tmp"),
            ("shm", "/dev/shm"),
        ):
            result[f"global_{label}_list_denied"] = _filesystem_call_denied(
                lambda path=root: os.listdir(path)
            )
        try:
            inherited_target = os.readlink(f"/proc/self/fd/{inherited_global_fd}")
        except OSError as error:
            result["inherited_global_file_fd_closed"] = error.errno == errno.ENOENT
        else:
            result["inherited_global_file_fd_closed"] = (
                inherited_target != global_sentinels["tmp"]
            )

        parent_status = f"/proc/{trusted_parent_pid}/status"
        parent_fds = f"/proc/{trusted_parent_pid}/fd"
        result["trusted_parent_proc_read_denied"] = _filesystem_call_denied(
            lambda: Path(parent_status).read_text(encoding="utf-8")
        )
        result["trusted_parent_proc_list_denied"] = _filesystem_call_denied(
            lambda: os.listdir(parent_fds)
        )
        result["self_proc_read_allowed"] = "Pid:" in Path(
            "/proc/self/status"
        ).read_text(encoding="utf-8")

        result["readonly_probe_read_allowed"] = (
            Path(readonly_probe_path).read_text(encoding="utf-8") == '{"probe": "ok"}\n'
        )
        result["readonly_probe_write_denied"] = _filesystem_call_denied(
            lambda: Path(readonly_probe_path).write_text(
                '{"probe": "changed"}\n', encoding="utf-8"
            )
        )
        result["readonly_probe_truncate_denied"] = _filesystem_call_denied(
            lambda: os.truncate(readonly_probe_path, 0)
        )
        result["readonly_probe_delete_denied"] = _filesystem_call_denied(
            lambda: os.unlink(readonly_probe_path)
        )
        result["readonly_probe_hardlink_denied"] = _filesystem_call_denied(
            lambda: os.link(
                readonly_probe_path,
                str(Path(state_directory) / "probe-hardlink"),
            )
        )

        import fractions

        result["python_imports_allowed"] = fractions.Fraction(84, 2) == 42
        own_state_checks = []
        for directory, filename in (
            (state_directory, "state.txt"),
            (var_tmp_directory, "var-state.txt"),
            (shm_directory, "shm-state.txt"),
        ):
            own_path = Path(directory) / filename
            own_path.write_text("first", encoding="utf-8")
            with own_path.open("a", encoding="utf-8") as output:
                output.write("-second")
            own_state_checks.append(
                own_path.read_text(encoding="utf-8") == "first-second"
            )
        result["own_state_writes_persist"] = all(own_state_checks)
    except BaseException as error:
        result = {
            "reader_error": f"{type(error).__name__}: {error}",
            **result,
        }
    connection.send(result)
    connection.close()


def _worker_probe(
    connection: multiprocessing.connection.Connection,
    inherited_socket_target: str,
    listener_port: int,
    unix_stream_path: str,
    unix_datagram_path: str,
    abstract_stream_address: bytes,
    abstract_datagram_address: bytes,
    inherited_unix_socket_targets: list[str],
    inherited_unix_listener_fds: list[int],
) -> None:
    result: dict[str, Any] = {}
    try:
        machine = platform.machine().lower()
        syscalls = _SYSCALLS[machine]
        result["hook_loaded"] = os.environ.get(_HOOK_LOADED_ENV) == "1"
        result["filter_active"] = os.environ.get(_FILTER_ACTIVE_ENV) == "1"
        result["process_filter_active"] = (
            os.environ.get(_PROCESS_FILTER_ACTIVE_ENV) == "1"
        )
        result["inherited_fds_clean_marker"] = (
            os.environ.get(_INHERITED_FDS_CLEAN_ENV) == "1"
        )
        result["restricted_unpickler_active"] = (
            os.environ.get(_RESTRICTED_UNPICKLER_ENV) == "1"
        )
        result["landlock_active"] = os.environ.get(_LANDLOCK_ACTIVE_ENV) == "1"
        result["landlock_effective_abi"] = os.environ.get(
            _LANDLOCK_EFFECTIVE_ABI_ENV, ""
        )
        result["landlock_missing_controls"] = [
            control
            for control in os.environ.get(_LANDLOCK_MISSING_CONTROLS_ENV, "").split(",")
            if control
        ]
        result["session_state_directory"] = os.environ[_SESSION_STATE_ENV]
        result["session_var_tmp_directory"] = os.environ[_SESSION_VAR_TMP_ENV]
        result["session_shm_directory"] = os.environ[_SESSION_SHM_ENV]
        result["seccomp_mode_filter"] = _seccomp_mode() == 2

        import _socket

        result["python_ipv4_denied"] = _expect_network_denied(
            lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        )
        result["python_ipv6_denied"] = _expect_network_denied(
            lambda: socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        )
        result["native_socket_type_denied"] = _expect_network_denied(
            lambda: _socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        )

        class DerivedSocket(socket.socket):
            pass

        native_base = DerivedSocket.__mro__[1].__mro__[1]
        result["mro_base_denied"] = _expect_network_denied(
            lambda: native_base(socket.AF_INET, socket.SOCK_STREAM)
        )
        result["direct_syscall_denied"] = _direct_syscall_denied(
            syscalls["socket"],
            ctypes.c_long(socket.AF_INET),
            ctypes.c_long(socket.SOCK_STREAM),
            ctypes.c_long(0),
        )
        result["raw_socket_denied"] = _expect_network_denied(
            lambda: socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3))
        )
        socket_pair = (ctypes.c_int * 2)()
        result["non_unix_socketpair_denied"] = _direct_syscall_denied(
            syscalls["socketpair"],
            ctypes.c_long(socket.AF_INET),
            ctypes.c_long(socket.SOCK_STREAM),
            ctypes.c_long(0),
            ctypes.byref(socket_pair),
        )
        io_uring_parameters = ctypes.create_string_buffer(256)
        result["io_uring_setup_denied"] = _direct_syscall_denied(
            syscalls["io_uring_setup"],
            ctypes.c_long(1),
            ctypes.byref(io_uring_parameters),
        )

        try:
            forked_pid = os.fork()
        except OSError as error:
            result["fork_denied"] = error.errno == errno.EPERM
        else:
            if forked_pid == 0:
                os._exit(91)
            os.waitpid(forked_pid, 0)
            result["fork_denied"] = False

        result["clone3_denied"] = _direct_syscall_has_errno(
            syscalls["clone3"],
            errno.ENOSYS,
            ctypes.c_void_p(),
            ctypes.c_size_t(0),
        )
        for syscall_name in (
            "ptrace",
            "process_vm_readv",
            "process_vm_writev",
            "pidfd_send_signal",
            "pidfd_open",
            "pidfd_getfd",
        ):
            result[f"{syscall_name}_denied"] = _direct_syscall_has_errno(
                syscalls[syscall_name],
                errno.EPERM,
                ctypes.c_long(0),
                ctypes.c_long(0),
                ctypes.c_long(0),
                ctypes.c_long(0),
                ctypes.c_long(0),
                ctypes.c_long(0),
            )
        for syscall_name in (
            "kill",
            "tkill",
            "tgkill",
            "rt_sigqueueinfo",
            "rt_tgsigqueueinfo",
            "process_madvise",
            "process_mrelease",
        ):
            result[f"{syscall_name}_syscall_denied"] = _direct_syscall_has_errno(
                syscalls[syscall_name],
                errno.EPERM,
                ctypes.c_long(0),
                ctypes.c_long(0),
                ctypes.c_long(0),
                ctypes.c_long(0),
                ctypes.c_long(0),
                ctypes.c_long(0),
            )
        result["prlimit64_syscall_denied"] = _direct_syscall_has_errno(
            syscalls["prlimit64"],
            errno.EPERM,
            ctypes.c_long(os.getppid()),
            ctypes.c_long(resource.RLIMIT_NOFILE),
            ctypes.c_void_p(),
            ctypes.c_void_p(),
        )
        result["sysv_ipc_syscalls_denied"] = all(
            _direct_syscall_has_errno(
                syscalls[syscall_name],
                errno.EPERM,
                ctypes.c_long(0),
                ctypes.c_long(0),
                ctypes.c_long(0),
                ctypes.c_long(0),
                ctypes.c_long(0),
                ctypes.c_long(0),
            )
            for syscall_name in (
                "shmget",
                "shmat",
                "shmdt",
                "shmctl",
                "msgget",
                "msgsnd",
                "msgrcv",
                "msgctl",
                "semget",
                "semop",
                "semtimedop",
                "semctl",
            )
        )
        parent_pid = os.getppid()
        pid_targeted_scheduler_denied = all(
            _direct_syscall_has_errno(
                syscalls[syscall_name],
                errno.EPERM,
                ctypes.c_long(parent_pid),
                ctypes.c_long(0),
                ctypes.c_long(0),
                ctypes.c_long(0),
                ctypes.c_long(0),
                ctypes.c_long(0),
            )
            for syscall_name in (
                "sched_setaffinity",
                "sched_getaffinity",
                "sched_setparam",
                "sched_getparam",
                "sched_setscheduler",
                "sched_getscheduler",
                "sched_setattr",
                "sched_getattr",
                "sched_rr_get_interval",
            )
        )
        priority_syscalls_denied = all(
            _direct_syscall_has_errno(
                syscalls[syscall_name],
                errno.EPERM,
                ctypes.c_long(os.PRIO_PROCESS),
                ctypes.c_long(parent_pid),
                ctypes.c_long(0),
            )
            for syscall_name in (
                "setpriority",
                "getpriority",
            )
        )
        ioprio_syscalls_denied = all(
            _direct_syscall_has_errno(
                syscalls[syscall_name],
                errno.EPERM,
                ctypes.c_long(_IOPRIO_WHO_PROCESS),
                ctypes.c_long(parent_pid),
                ctypes.c_long(0),
            )
            for syscall_name in (
                "ioprio_set",
                "ioprio_get",
            )
        )
        result["scheduler_priority_syscalls_denied"] = (
            pid_targeted_scheduler_denied
            and priority_syscalls_denied
            and ioprio_syscalls_denied
        )

        self_affinity = os.sched_getaffinity(0)
        self_limits = resource.prlimit(0, resource.RLIMIT_NOFILE)
        self_priority = os.getpriority(os.PRIO_PROCESS, 0)
        self_scheduler = os.sched_getscheduler(0)
        self_parameters = os.sched_getparam(0)
        self_ioprio = _direct_syscall_value(
            syscalls["ioprio_get"],
            ctypes.c_long(_IOPRIO_WHO_PROCESS),
            ctypes.c_long(0),
        )
        result["self_process_queries_allowed"] = (
            bool(self_affinity)
            and len(self_limits) == 2
            and isinstance(self_priority, int)
            and isinstance(self_scheduler, int)
            and self_parameters is not None
            and self_ioprio >= 0
        )

        result["direct_execve_denied"] = _direct_syscall_has_errno(
            syscalls["execve"],
            errno.EPERM,
            ctypes.c_void_p(),
            ctypes.c_void_p(),
            ctypes.c_void_p(),
        )
        result["direct_execveat_denied"] = _direct_syscall_has_errno(
            syscalls["execveat"],
            errno.EPERM,
            ctypes.c_long(-1),
            ctypes.c_void_p(),
            ctypes.c_void_p(),
            ctypes.c_void_p(),
            ctypes.c_long(0),
        )

        thread_result: list[int] = []
        thread = threading.Thread(target=lambda: thread_result.append(42))
        try:
            thread.start()
            thread.join(timeout=5)
            result["thread_clone_allowed"] = (
                not thread.is_alive() and thread_result == [42]
            )
        except (OSError, RuntimeError):
            result["thread_clone_allowed"] = False

        result["python_unix_socket_denied"] = _expect_network_denied(
            lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        )
        result["direct_unix_socket_denied"] = _direct_syscall_denied(
            syscalls["socket"],
            ctypes.c_long(socket.AF_UNIX),
            ctypes.c_long(socket.SOCK_STREAM),
            ctypes.c_long(0),
        )

        def control_socket_operation_denied(
            operation: Callable[[socket.socket], object],
        ) -> bool:
            duplicated = os.dup(connection.fileno())
            unix_socket = socket.socket(fileno=duplicated)
            try:
                return _expect_network_denied(lambda: operation(unix_socket))
            finally:
                unix_socket.close()

        result["unix_filesystem_connect_denied"] = control_socket_operation_denied(
            lambda unix_socket: unix_socket.connect(unix_stream_path)
        )
        result["unix_abstract_connect_denied"] = control_socket_operation_denied(
            lambda unix_socket: unix_socket.connect(abstract_stream_address)
        )
        child_bind_path = f"{unix_stream_path}.child-bind"
        result["unix_filesystem_bind_denied"] = control_socket_operation_denied(
            lambda unix_socket: unix_socket.bind(child_bind_path)
        )
        child_abstract_address = abstract_stream_address + b"-child-bind"
        result["unix_abstract_bind_denied"] = control_socket_operation_denied(
            lambda unix_socket: unix_socket.bind(child_abstract_address)
        )
        result["unix_listen_denied"] = control_socket_operation_denied(
            lambda unix_socket: unix_socket.listen(1)
        )

        datagram_left, datagram_right = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_DGRAM
        )
        try:
            result["unix_filesystem_sendto_denied"] = _expect_network_denied(
                lambda: datagram_left.sendto(b"blocked", unix_datagram_path)
            )
            result["unix_abstract_sendto_denied"] = _expect_network_denied(
                lambda: datagram_left.sendto(b"blocked", abstract_datagram_address)
            )
            result["unix_named_sendmsg_denied"] = _expect_network_denied(
                lambda: datagram_left.sendmsg([b"blocked"], [], 0, unix_datagram_path)
            )
        finally:
            datagram_left.close()
            datagram_right.close()

        direct_socket_fd = os.dup(connection.fileno())
        try:
            sockaddr, sockaddr_size = _unix_sockaddr(unix_stream_path)
            result["direct_unix_connect_denied"] = _direct_syscall_denied(
                syscalls["connect"],
                ctypes.c_long(direct_socket_fd),
                ctypes.byref(sockaddr),
                ctypes.c_long(sockaddr_size),
            )
        finally:
            os.close(direct_socket_fd)
        result["inherited_unix_accept_denied"] = all(
            _direct_syscall_denied(
                syscalls["accept"],
                ctypes.c_long(descriptor),
                ctypes.c_void_p(),
                ctypes.c_void_p(),
            )
            for descriptor in inherited_unix_listener_fds
        )
        result["inherited_unix_accept4_denied"] = all(
            _direct_syscall_denied(
                syscalls["accept4"],
                ctypes.c_long(descriptor),
                ctypes.c_void_p(),
                ctypes.c_void_p(),
                ctypes.c_long(0),
            )
            for descriptor in inherited_unix_listener_fds
        )

        exec_probe = (
            "import errno,socket,sys; "
            "\ntry: socket.socket(socket.AF_INET,socket.SOCK_STREAM)"
            "\nexcept OSError as e: sys.exit(0 if e.errno==errno.ENETUNREACH else 2)"
            "\nelse: sys.exit(3)"
        )
        result["env_clearing_exec_denied"] = _subprocess_creation_denied(
            [sys.executable, "-I", "-c", exec_probe]
        )
        unix_exec_probe = (
            "import errno,socket,sys; "
            "\ntry: socket.socket(socket.AF_UNIX,socket.SOCK_STREAM).connect(sys.argv[1])"
            "\nexcept OSError as e: sys.exit(0 if e.errno==errno.ENETUNREACH else 2)"
            "\nelse: sys.exit(3)"
        )
        result["env_clearing_exec_unix_denied"] = _subprocess_creation_denied(
            [sys.executable, "-I", "-c", unix_exec_probe, unix_stream_path]
        )

        curl = shutil.which("curl")
        if curl is None:
            raise RuntimeError("curl is required for the seccomp bypass preflight")
        result["curl_denied"] = _subprocess_creation_denied(
            [
                curl,
                "--silent",
                "--show-error",
                "--max-time",
                "1",
                f"http://127.0.0.1:{listener_port}/",
            ]
        )
        result["curl_unix_socket_denied"] = _subprocess_creation_denied(
            [
                curl,
                "--silent",
                "--show-error",
                "--max-time",
                "1",
                "--unix-socket",
                unix_stream_path,
                "http://localhost/",
            ]
        )

        left, right = socket.socketpair()
        try:
            left.sendall(b"unix-ok")
            result["unix_socketpair_allowed"] = right.recv(7) == b"unix-ok"
        finally:
            left.close()
            right.close()

        nested_read, nested_write = multiprocessing.Pipe(duplex=False)
        try:
            nested_process = multiprocessing.get_context("fork").Process(
                target=_nested_ipc_worker,
                args=(nested_write,),
            )
            nested_process.start()
        except OSError as error:
            result["multiprocessing_process_denied"] = error.errno == errno.EPERM
        else:
            nested_process.kill()
            nested_process.join(timeout=5)
            result["multiprocessing_process_denied"] = False
        finally:
            nested_read.close()
            nested_write.close()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.txt"
            path.write_text("file-ok", encoding="utf-8")
            result["file_io_allowed"] = path.read_text(encoding="utf-8") == "file-ok"

        namespace: dict[str, object] = {}
        exec("values = [6, 7]", namespace)
        exec("answer = values[0] * values[1]", namespace)
        result["stateful_math_allowed"] = namespace["answer"] == 42

        inherited_targets = []
        for entry in os.listdir("/proc/self/fd"):
            path = Path("/proc/self/fd") / entry
            try:
                target = os.readlink(path)
            except OSError:
                continue
            inherited_targets.append(target)
        result["inherited_inet_listener_closed"] = (
            inherited_socket_target not in inherited_targets
        )
        result["inherited_unix_listeners_closed"] = all(
            target not in inherited_targets for target in inherited_unix_socket_targets
        )
        remaining = _socket_descriptors()
        untrusted = [item for item in remaining if item["fd"] != connection.fileno()]
        result["control_socket_preserved"] = any(
            item["fd"] == connection.fileno() and item["domain"] == socket.AF_UNIX
            for item in remaining
        )
        result["untrusted_inherited_socket_count"] = len(untrusted)
        result["untrusted_inherited_sockets"] = untrusted
        result["non_unix_inherited_socket_count"] = sum(
            item["domain"] != socket.AF_UNIX for item in remaining
        )

        try:
            import numpy as np
            import pandas as pd

            csv_buffer = io.StringIO()
            frame = pd.DataFrame({"value": np.array([6, 7], dtype=np.int64)})
            frame.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            restored = pd.read_csv(csv_buffer)
            result["numpy_pandas_csv_allowed"] = (
                int(restored["value"].sum()) == 13
                and int(np.dot(np.array([6]), np.array([7]))) == 42
            )
        except BaseException as error:
            result["numpy_pandas_csv_allowed"] = False
            result["numpy_pandas_error"] = f"{type(error).__name__}: {error}"

        try:
            from IPython.terminal.interactiveshell import TerminalInteractiveShell
            from traitlets.config import Config

            shell_config = Config()
            shell_config.HistoryManager.hist_file = ":memory:"
            shell = TerminalInteractiveShell(config=shell_config)
            first_cell = shell.run_cell("sciprobe_state = 40", store_history=False)
            second_cell = shell.run_cell("sciprobe_state += 2", store_history=False)
            result["ipython_stateful_cells_allowed"] = (
                first_cell.error_before_exec is None
                and first_cell.error_in_exec is None
                and second_cell.error_before_exec is None
                and second_cell.error_in_exec is None
                and shell.user_ns.get("sciprobe_state") == 42
            )
        except BaseException as error:
            result["ipython_stateful_cells_allowed"] = False
            result["ipython_error"] = f"{type(error).__name__}: {error}"
    except BaseException as error:
        result = {
            "worker_error": f"{type(error).__name__}: {error}",
            **result,
        }
    connection.send(result)
    connection.close()


def _driver() -> dict[str, Any]:
    from multiprocessing.process import BaseProcess

    parent_result = {
        "hook_loaded_in_sandbox_parent": os.environ.get(_HOOK_LOADED_ENV) == "1",
        "filter_absent_in_sandbox_parent": (os.environ.get(_FILTER_ACTIVE_ENV) is None),
        "bootstrap_wrapped_in_sandbox_parent": bool(
            getattr(BaseProcess._bootstrap, "_sciprobe_seccomp_hook", False)
        ),
    }
    with tempfile.TemporaryDirectory() as directory, ExitStack() as sockets:
        pipe_read, pipe_write = os.pipe()
        inherited_secret_pipe_fd = fcntl.fcntl(
            pipe_read,
            fcntl.F_DUPFD_CLOEXEC,
            512,
        )
        os.close(pipe_read)
        try:
            os.write(pipe_write, b"parent-secret-pipe")
        finally:
            os.close(pipe_write)
        sockets.callback(os.close, inherited_secret_pipe_fd)

        sysv_secret = b"parent-secret-sysv-shm"
        parent_sysv_shm_id = _create_sysv_shm_secret(sysv_secret)
        sockets.callback(_remove_sysv_shm, parent_sysv_shm_id)

        readonly_probe_path = str(Path(directory) / "readonly-probe.json")
        Path(readonly_probe_path).write_text('{"probe": "ok"}\n', encoding="utf-8")
        os.environ[_READONLY_PATHS_ENV] = readonly_probe_path

        global_sentinels: dict[str, str] = {}
        global_create_paths: dict[str, str] = {}
        inherited_global_fd = -1
        filesystem_token = os.urandom(12).hex()
        for label, root in (
            ("tmp", "/tmp"),
            ("var_tmp", "/var/tmp"),
            ("shm", "/dev/shm"),
        ):
            sentinel = str(Path(root) / f".sciprobe-global-{filesystem_token}")
            create_path = str(
                Path(root) / f".sciprobe-forbidden-create-{filesystem_token}"
            )
            descriptor = os.open(
                sentinel,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600,
            )
            try:
                os.write(descriptor, b"global-secret")
            finally:
                if label == "tmp":
                    inherited_global_fd = descriptor
                    sockets.callback(os.close, descriptor)
                else:
                    os.close(descriptor)
            global_sentinels[label] = sentinel
            global_create_paths[label] = create_path
            sockets.callback(Path(sentinel).unlink, missing_ok=True)
            sockets.callback(Path(create_path).unlink, missing_ok=True)

        listener = sockets.enter_context(
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        )
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(2)
        listener_port = int(listener.getsockname()[1])
        listener_target = os.readlink(f"/proc/self/fd/{listener.fileno()}")
        parent_result["sandbox_parent_inet_listener_allowed"] = True

        address_token = os.urandom(8).hex()
        unix_stream_path = str(Path(directory) / "stream.sock")
        unix_datagram_path = str(Path(directory) / "datagram.sock")
        abstract_stream_address = b"\0sciprobe-seccomp-stream-" + address_token.encode(
            "ascii"
        )
        abstract_datagram_address = (
            b"\0sciprobe-seccomp-datagram-" + address_token.encode("ascii")
        )
        unix_stream_listener = sockets.enter_context(
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        )
        unix_stream_listener.bind(unix_stream_path)
        unix_stream_listener.listen(1)
        unix_stream_listener.settimeout(2)
        abstract_stream_listener = sockets.enter_context(
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        )
        abstract_stream_listener.bind(abstract_stream_address)
        abstract_stream_listener.listen(1)
        abstract_stream_listener.settimeout(2)
        unix_datagram_listener = sockets.enter_context(
            socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        )
        unix_datagram_listener.bind(unix_datagram_path)
        unix_datagram_listener.settimeout(2)
        abstract_datagram_listener = sockets.enter_context(
            socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        )
        abstract_datagram_listener.bind(abstract_datagram_address)
        abstract_datagram_listener.settimeout(2)
        unix_listeners = (
            unix_stream_listener,
            abstract_stream_listener,
            unix_datagram_listener,
            abstract_datagram_listener,
        )
        inherited_unix_socket_targets = [
            os.readlink(f"/proc/self/fd/{unix_socket.fileno()}")
            for unix_socket in unix_listeners
        ]
        inherited_unix_listener_fds = [
            unix_stream_listener.fileno(),
            abstract_stream_listener.fileno(),
        ]
        parent_result["sandbox_parent_unix_listeners_allowed"] = True

        read_connection, write_connection = multiprocessing.Pipe(duplex=True)
        context = multiprocessing.get_context("fork")
        process = context.Process(
            target=_worker_probe,
            args=(
                write_connection,
                listener_target,
                listener_port,
                unix_stream_path,
                unix_datagram_path,
                abstract_stream_address,
                abstract_datagram_address,
                inherited_unix_socket_targets,
                inherited_unix_listener_fds,
            ),
        )
        process.start()
        write_connection.close()
        if not read_connection.poll(15):
            process.kill()
            process.join(timeout=5)
            raise RuntimeError("seccomp worker produced no result")
        worker_result = read_connection.recv()
        parent_result["primitive_control_messages_allowed"] = isinstance(
            worker_result, dict
        )
        read_connection.close()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
            raise RuntimeError("seccomp worker did not exit")
        if process.exitcode != 0:
            raise RuntimeError(f"seccomp worker exited {process.exitcode}")
        for key in (
            "session_state_directory",
            "session_var_tmp_directory",
            "session_shm_directory",
        ):
            session_path = worker_result.get(key)
            if isinstance(session_path, str):
                sockets.callback(shutil.rmtree, session_path, ignore_errors=True)

        writer_parent, writer_child = multiprocessing.Pipe(duplex=True)
        writer_process = context.Process(
            target=_session_writer,
            args=(writer_child,),
        )
        writer_process.start()
        writer_child.close()
        reader_process: multiprocessing.Process | None = None
        reader_parent: multiprocessing.connection.Connection | None = None
        try:
            if not writer_parent.poll(10):
                raise RuntimeError("filesystem writer produced no session details")
            writer_info = writer_parent.recv()
            if not isinstance(writer_info, dict):
                raise RuntimeError("filesystem writer returned invalid details")
            for key in (
                "state_directory",
                "var_tmp_directory",
                "shm_directory",
            ):
                session_path = writer_info.get(key)
                if isinstance(session_path, str):
                    sockets.callback(shutil.rmtree, session_path, ignore_errors=True)

            reader_parent, reader_child = multiprocessing.Pipe(duplex=False)
            writer_pid = writer_process.pid
            if writer_pid is None:
                raise RuntimeError("filesystem writer has no process id")
            writer_nofile_limits = resource.prlimit(
                writer_pid,
                resource.RLIMIT_NOFILE,
            )
            writer_affinity = tuple(sorted(os.sched_getaffinity(writer_pid)))
            writer_nice = os.getpriority(os.PRIO_PROCESS, writer_pid)
            machine = platform.machine().lower()
            writer_ioprio = _direct_syscall_value(
                _SYSCALLS[machine]["ioprio_get"],
                ctypes.c_long(_IOPRIO_WHO_PROCESS),
                ctypes.c_long(writer_pid),
            )
            reader_process = context.Process(
                target=_session_reader,
                args=(
                    reader_child,
                    writer_info,
                    readonly_probe_path,
                    global_sentinels,
                    global_create_paths,
                    inherited_global_fd,
                    inherited_secret_pipe_fd,
                    parent_sysv_shm_id,
                    len(sysv_secret),
                    writer_pid,
                    writer_nofile_limits,
                    writer_affinity,
                    writer_nice,
                    writer_ioprio,
                    os.getpid(),
                ),
            )
            reader_process.start()
            reader_child.close()
            if not reader_parent.poll(10):
                raise RuntimeError("filesystem reader produced no result")
            reader_result = reader_parent.recv()
            if not isinstance(reader_result, dict):
                raise RuntimeError("filesystem reader returned invalid result")
            reader_parent.close()
            reader_parent = None
            reader_process.join(timeout=5)
            if reader_process.is_alive():
                raise RuntimeError("filesystem reader did not exit")
            if reader_process.exitcode != 0:
                raise RuntimeError(
                    f"filesystem reader exited {reader_process.exitcode}"
                )
            for key in (
                "state_directory",
                "var_tmp_directory",
                "shm_directory",
            ):
                session_path = reader_result.get(key)
                if isinstance(session_path, str):
                    sockets.callback(shutil.rmtree, session_path, ignore_errors=True)

            parent_result.update(reader_result)
            parent_result["session_directories_distinct"] = all(
                writer_info[key] != reader_result[key]
                for key in (
                    "state_directory",
                    "var_tmp_directory",
                    "shm_directory",
                )
            )
            parent_result["writer_landlock_active"] = (
                writer_info.get("landlock_active") is True
            )
            writer_parent.send("stop")
            if not writer_parent.poll(5):
                raise RuntimeError("filesystem writer did not acknowledge stop")
            writer_exit = writer_parent.recv()
            parent_result["writer_stop_received"] = (
                isinstance(writer_exit, dict)
                and writer_exit.get("stop_received") is True
            )
            parent_result["writer_secret_intact"] = (
                isinstance(writer_exit, dict)
                and writer_exit.get("secret_intact") is True
            )
            writer_process.join(timeout=5)
            if writer_process.is_alive():
                raise RuntimeError("filesystem writer did not exit")
            if writer_process.exitcode != 0:
                raise RuntimeError(
                    f"filesystem writer exited {writer_process.exitcode}"
                )
        finally:
            if reader_parent is not None:
                reader_parent.close()
            if reader_process is not None and reader_process.is_alive():
                reader_process.kill()
                reader_process.join(timeout=5)
            writer_parent.close()
            if writer_process.is_alive():
                writer_process.kill()
                writer_process.join(timeout=5)

        exploit_marker = Path(directory) / "pickle-exploit-executed"
        parent_result["reducer_pickle_rejected"] = _control_payload_rejected(
            _ReducerPayload(str(exploit_marker))
        )
        parent_result["reducer_never_executed"] = not exploit_marker.exists()
        parent_result["global_pickle_rejected"] = _control_payload_rejected(os.system)
        parent_result["persistent_pickle_rejected"] = _control_payload_rejected(
            b"Pforbidden-reference\n.", raw=True
        )
        parent_result["oversize_control_payload_rejected"] = (
            _oversize_control_payload_rejected()
        )

        with socket.create_connection(
            ("127.0.0.1", listener_port), timeout=2
        ) as client:
            accepted, _ = listener.accept()
            sockets.enter_context(accepted)
            client.sendall(b"parent-ok")
            parent_result["sandbox_parent_listener_still_usable"] = (
                accepted.recv(9) == b"parent-ok"
            )

        unix_stream_checks = []
        for unix_listener, address in (
            (unix_stream_listener, unix_stream_path),
            (abstract_stream_listener, abstract_stream_address),
        ):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(2)
                client.connect(address)
                accepted, _ = unix_listener.accept()
                with accepted:
                    client.sendall(b"unix-parent-ok")
                    unix_stream_checks.append(accepted.recv(14) == b"unix-parent-ok")
        datagram_checks = []
        for unix_listener, address in (
            (unix_datagram_listener, unix_datagram_path),
            (abstract_datagram_listener, abstract_datagram_address),
        ):
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
                client.sendto(b"unix-datagram-ok", address)
                datagram_checks.append(unix_listener.recv(16) == b"unix-datagram-ok")
        parent_result["sandbox_parent_unix_listeners_still_usable"] = all(
            unix_stream_checks + datagram_checks
        )

    return {**parent_result, **worker_result}


def _run_probe(hook_dir: Path) -> dict[str, Any]:
    hook = hook_dir.resolve() / "sitecustomize.py"
    if not hook.is_file():
        raise RuntimeError(f"required seccomp hook is absent: {hook}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(hook.parent)
    env[_REQUIRED_ENV] = "1"
    for marker in (
        _HOOK_LOADED_ENV,
        _FILTER_ACTIVE_ENV,
        _PROCESS_FILTER_ACTIVE_ENV,
        _INHERITED_FDS_CLEAN_ENV,
        _RESTRICTED_UNPICKLER_ENV,
        _LANDLOCK_ACTIVE_ENV,
        _SESSION_STATE_ENV,
        _SESSION_VAR_TMP_ENV,
        _SESSION_SHM_ENV,
        _LANDLOCK_EFFECTIVE_ABI_ENV,
        _LANDLOCK_MISSING_CONTROLS_ENV,
    ):
        env.pop(marker, None)
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--driver"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "seccomp validation driver failed "
            f"({completed.returncode}): {completed.stderr.strip()}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("seccomp validation driver returned invalid output")
    result = json.loads(lines[0])
    if not isinstance(result, dict):
        raise RuntimeError("seccomp validation driver returned a non-object")

    boolean_checks = (
        "hook_loaded_in_sandbox_parent",
        "filter_absent_in_sandbox_parent",
        "bootstrap_wrapped_in_sandbox_parent",
        "sandbox_parent_inet_listener_allowed",
        "sandbox_parent_listener_still_usable",
        "sandbox_parent_unix_listeners_allowed",
        "sandbox_parent_unix_listeners_still_usable",
        "primitive_control_messages_allowed",
        "reducer_pickle_rejected",
        "reducer_never_executed",
        "global_pickle_rejected",
        "persistent_pickle_rejected",
        "oversize_control_payload_rejected",
        "hook_loaded",
        "filter_active",
        "process_filter_active",
        "inherited_fds_clean_marker",
        "restricted_unpickler_active",
        "landlock_active",
        "writer_landlock_active",
        "home_is_private",
        "tmpdir_is_private",
        "session_directories_distinct",
        "sibling_read_denied",
        "sibling_write_denied",
        "sibling_list_denied",
        "sibling_delete_denied",
        "sibling_signal_zero_denied",
        "all_processes_signal_zero_denied",
        "sibling_prlimit_query_denied",
        "sibling_prlimit_mutation_denied",
        "sibling_affinity_query_denied",
        "sibling_affinity_mutation_denied",
        "sibling_priority_query_denied",
        "sibling_priority_mutation_denied",
        "sibling_ioprio_query_denied",
        "sibling_ioprio_mutation_denied",
        "parent_secret_pipe_fd_closed",
        "parent_sysv_shm_attach_read_denied",
        "global_tmp_read_denied",
        "global_tmp_write_denied",
        "global_tmp_create_denied",
        "global_tmp_list_denied",
        "global_var_tmp_read_denied",
        "global_var_tmp_write_denied",
        "global_var_tmp_create_denied",
        "global_var_tmp_list_denied",
        "global_shm_read_denied",
        "global_shm_write_denied",
        "global_shm_create_denied",
        "global_shm_list_denied",
        "inherited_global_file_fd_closed",
        "trusted_parent_proc_read_denied",
        "trusted_parent_proc_list_denied",
        "self_proc_read_allowed",
        "readonly_probe_read_allowed",
        "readonly_probe_write_denied",
        "readonly_probe_delete_denied",
        "readonly_probe_hardlink_denied",
        "python_imports_allowed",
        "own_state_writes_persist",
        "writer_stop_received",
        "writer_secret_intact",
        "seccomp_mode_filter",
        "python_ipv4_denied",
        "python_ipv6_denied",
        "native_socket_type_denied",
        "mro_base_denied",
        "direct_syscall_denied",
        "raw_socket_denied",
        "non_unix_socketpair_denied",
        "io_uring_setup_denied",
        "fork_denied",
        "clone3_denied",
        "ptrace_denied",
        "process_vm_readv_denied",
        "process_vm_writev_denied",
        "pidfd_send_signal_denied",
        "pidfd_open_denied",
        "pidfd_getfd_denied",
        "kill_syscall_denied",
        "tkill_syscall_denied",
        "tgkill_syscall_denied",
        "rt_sigqueueinfo_syscall_denied",
        "rt_tgsigqueueinfo_syscall_denied",
        "process_madvise_syscall_denied",
        "process_mrelease_syscall_denied",
        "prlimit64_syscall_denied",
        "sysv_ipc_syscalls_denied",
        "scheduler_priority_syscalls_denied",
        "self_process_queries_allowed",
        "numpy_pandas_csv_allowed",
        "direct_execve_denied",
        "direct_execveat_denied",
        "thread_clone_allowed",
        "python_unix_socket_denied",
        "direct_unix_socket_denied",
        "unix_filesystem_connect_denied",
        "unix_abstract_connect_denied",
        "unix_filesystem_bind_denied",
        "unix_abstract_bind_denied",
        "unix_listen_denied",
        "unix_filesystem_sendto_denied",
        "unix_abstract_sendto_denied",
        "unix_named_sendmsg_denied",
        "direct_unix_connect_denied",
        "inherited_unix_accept_denied",
        "inherited_unix_accept4_denied",
        "env_clearing_exec_denied",
        "env_clearing_exec_unix_denied",
        "curl_denied",
        "curl_unix_socket_denied",
        "unix_socketpair_allowed",
        "multiprocessing_process_denied",
        "file_io_allowed",
        "stateful_math_allowed",
        "ipython_stateful_cells_allowed",
        "inherited_inet_listener_closed",
        "inherited_unix_listeners_closed",
        "control_socket_preserved",
    )
    failed = [name for name in boolean_checks if result.get(name) is not True]
    # Truncation confinement is Landlock ABI 3. On an older kernel the sandbox
    # cannot install it, so asserting it would fail for a reason no
    # configuration can fix. Require it whenever the kernel offers it, and
    # otherwise record it as unenforced so the run states what confined it
    # rather than implying a guarantee it never had.
    unenforced = []
    if "truncate" in result.get("landlock_missing_controls", []):
        unenforced.append("readonly_probe_truncate_denied")
    elif result.get("readonly_probe_truncate_denied") is not True:
        failed.append("readonly_probe_truncate_denied")
    result["unenforced_controls"] = unenforced
    if result.get("non_unix_inherited_socket_count") != 0:
        failed.append("non_unix_inherited_socket_count")
    if result.get("untrusted_inherited_socket_count") != 0:
        failed.append("untrusted_inherited_socket_count")
    if failed:
        raise RuntimeError(
            f"seccomp validation failed checks: {failed}; result={result}"
        )
    result["status"] = "ok"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", action="store_true")
    parser.add_argument(
        "--hook-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "sandbox_seccomp_hook",
    )
    arguments = parser.parse_args()
    result = _driver() if arguments.driver else _run_probe(arguments.hook_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
