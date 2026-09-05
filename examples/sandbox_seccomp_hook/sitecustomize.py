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

"""Install SciProbe's kernel isolation gate in multiprocessing shell children.

The sandbox service must keep its HTTP listener, so the filter cannot be put on
the parent process. NeMo-Skills runs each stateful Python shell in a
``multiprocessing.Process``; wrapping ``BaseProcess._bootstrap`` installs the
filter in that child before its target, and therefore before model code, runs.
"""

from __future__ import annotations

import ctypes
import errno
import io
import os
import pickle
import platform
import stat
import sys
from multiprocessing.connection import Connection, _ConnectionBase
from multiprocessing.process import BaseProcess
from typing import Final

_REQUIRED_ENV: Final = "SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK"
_HOOK_LOADED_ENV: Final = "SCIPROBE_SECCOMP_HOOK_LOADED"
_FILTER_ACTIVE_ENV: Final = "SCIPROBE_SECCOMP_FILTER_ACTIVE"
_PROCESS_FILTER_ACTIVE_ENV: Final = "SCIPROBE_SECCOMP_PROCESS_FILTER_ACTIVE"
_INHERITED_FDS_CLEAN_ENV: Final = "SCIPROBE_SECCOMP_INHERITED_FDS_CLEAN"
_RESTRICTED_UNPICKLER_ENV: Final = "SCIPROBE_RESTRICTED_MP_UNPICKLER_ACTIVE"
_LANDLOCK_ACTIVE_ENV: Final = "SCIPROBE_LANDLOCK_FILTER_ACTIVE"
_SESSION_STATE_ENV: Final = "SCIPROBE_SESSION_STATE_DIR"
_SESSION_VAR_TMP_ENV: Final = "SCIPROBE_SESSION_VAR_TMP_DIR"
_SESSION_SHM_ENV: Final = "SCIPROBE_SESSION_SHM_DIR"
_READONLY_PATHS_ENV: Final = "SCIPROBE_SANDBOX_READONLY_PATHS"
_MAX_CONTROL_MESSAGE_BYTES: Final = 16 * 1024 * 1024

_PR_SET_SECCOMP: Final = 22
_PR_SET_NO_NEW_PRIVS: Final = 38
_SECCOMP_MODE_FILTER: Final = 2

_SECCOMP_RET_KILL_PROCESS: Final = 0x80000000
_SECCOMP_RET_ERRNO: Final = 0x00050000
_SECCOMP_RET_ALLOW: Final = 0x7FFF0000

_BPF_LD_W_ABS: Final = 0x20
_BPF_JMP_JEQ_K: Final = 0x15
_BPF_JMP_JSET_K: Final = 0x45
_BPF_RET_K: Final = 0x06

_SECCOMP_DATA_NR_OFFSET: Final = 0
_SECCOMP_DATA_ARCH_OFFSET: Final = 4
_SECCOMP_DATA_ARG0_OFFSET: Final = 16
_SECCOMP_DATA_ARG0_HIGH_OFFSET: Final = 20
_SECCOMP_DATA_ARG1_OFFSET: Final = 24
_SECCOMP_DATA_ARG1_HIGH_OFFSET: Final = 28
_SECCOMP_DATA_ARG4_LOW_OFFSET: Final = 48
_SECCOMP_DATA_ARG4_HIGH_OFFSET: Final = 52
_AF_UNIX: Final = 1
_X32_SYSCALL_BIT: Final = 0x40000000
_SOL_SOCKET: Final = 1
_SO_DOMAIN: Final = 39
_CLONE_THREAD: Final = 0x00010000
_IOPRIO_WHO_PROCESS: Final = 1

_LANDLOCK_CREATE_RULESET_VERSION: Final = 1
_LANDLOCK_RULE_PATH_BENEATH: Final = 1
_DEFAULT_MIN_LANDLOCK_ABI: Final = 3
_MIN_LANDLOCK_ABI_ENV: Final = "SCIPROBE_LANDLOCK_MIN_ABI"
_LANDLOCK_EFFECTIVE_ABI_ENV: Final = "SCIPROBE_LANDLOCK_EFFECTIVE_ABI"
_LANDLOCK_MISSING_CONTROLS_ENV: Final = "SCIPROBE_LANDLOCK_MISSING_CONTROLS"
_LANDLOCK_ACCESS_FS_EXECUTE: Final = 1 << 0
_LANDLOCK_ACCESS_FS_WRITE_FILE: Final = 1 << 1
_LANDLOCK_ACCESS_FS_READ_FILE: Final = 1 << 2
_LANDLOCK_ACCESS_FS_READ_DIR: Final = 1 << 3
_LANDLOCK_ACCESS_FS_REMOVE_DIR: Final = 1 << 4
_LANDLOCK_ACCESS_FS_REMOVE_FILE: Final = 1 << 5
_LANDLOCK_ACCESS_FS_MAKE_CHAR: Final = 1 << 6
_LANDLOCK_ACCESS_FS_MAKE_DIR: Final = 1 << 7
_LANDLOCK_ACCESS_FS_MAKE_REG: Final = 1 << 8
_LANDLOCK_ACCESS_FS_MAKE_SOCK: Final = 1 << 9
_LANDLOCK_ACCESS_FS_MAKE_FIFO: Final = 1 << 10
_LANDLOCK_ACCESS_FS_MAKE_BLOCK: Final = 1 << 11
_LANDLOCK_ACCESS_FS_MAKE_SYM: Final = 1 << 12
_LANDLOCK_ACCESS_FS_REFER: Final = 1 << 13
_LANDLOCK_ACCESS_FS_TRUNCATE: Final = 1 << 14
_LANDLOCK_ACCESS_FS_IOCTL_DEV: Final = 1 << 15

_ARCH_CONFIG: Final = {
    "x86_64": {
        "audit_arch": 0xC000003E,
        "socket": 41,
        "socketpair": 53,
        "connect": 42,
        "accept": 43,
        "sendto": 44,
        "sendmsg": 46,
        "bind": 49,
        "listen": 50,
        "accept4": 288,
        "sendmmsg": 307,
        "io_uring_setup": 425,
        "clone": 56,
        "clone3": 435,
        "landlock_create_ruleset": 444,
        "landlock_add_rule": 445,
        "landlock_restrict_self": 446,
        "process_denied": (
            29,  # shmget
            30,  # shmat
            31,  # shmctl
            57,  # fork
            58,  # vfork
            59,  # execve
            62,  # kill
            64,  # semget
            65,  # semop
            66,  # semctl
            67,  # shmdt
            68,  # msgget
            69,  # msgsnd
            70,  # msgrcv
            71,  # msgctl
            101,  # ptrace
            129,  # rt_sigqueueinfo
            200,  # tkill
            220,  # semtimedop
            234,  # tgkill
            297,  # rt_tgsigqueueinfo
            310,  # process_vm_readv
            311,  # process_vm_writev
            322,  # execveat
            424,  # pidfd_send_signal
            434,  # pidfd_open
            438,  # pidfd_getfd
            440,  # process_madvise
            448,  # process_mrelease
        ),
        "pid_zero_allowed": (
            142,  # sched_setparam
            143,  # sched_getparam
            144,  # sched_setscheduler
            145,  # sched_getscheduler
            148,  # sched_rr_get_interval
            203,  # sched_setaffinity
            204,  # sched_getaffinity
            302,  # prlimit64
            314,  # sched_setattr
            315,  # sched_getattr
        ),
        "priority_self_only": (140, 141),  # getpriority, setpriority
        "ioprio_self_only": (251, 252),  # ioprio_set, ioprio_get
        "deny_x32": True,
    },
    "aarch64": {
        "audit_arch": 0xC00000B7,
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
        "clone3": 435,
        "landlock_create_ruleset": 444,
        "landlock_add_rule": 445,
        "landlock_restrict_self": 446,
        "process_denied": (
            117,  # ptrace
            129,  # kill
            130,  # tkill
            131,  # tgkill
            138,  # rt_sigqueueinfo
            186,  # msgget
            187,  # msgctl
            188,  # msgrcv
            189,  # msgsnd
            190,  # semget
            191,  # semctl
            192,  # semtimedop
            193,  # semop
            194,  # shmget
            195,  # shmctl
            196,  # shmat
            197,  # shmdt
            221,  # execve
            240,  # rt_tgsigqueueinfo
            270,  # process_vm_readv
            271,  # process_vm_writev
            281,  # execveat
            424,  # pidfd_send_signal
            434,  # pidfd_open
            438,  # pidfd_getfd
            440,  # process_madvise
            448,  # process_mrelease
        ),
        "pid_zero_allowed": (
            118,  # sched_setparam
            119,  # sched_setscheduler
            120,  # sched_getscheduler
            121,  # sched_getparam
            122,  # sched_setaffinity
            123,  # sched_getaffinity
            127,  # sched_rr_get_interval
            261,  # prlimit64
            274,  # sched_setattr
            275,  # sched_getattr
        ),
        "priority_self_only": (140, 141),  # setpriority, getpriority
        "ioprio_self_only": (30, 31),  # ioprio_set, ioprio_get
        "deny_x32": False,
    },
}


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [
        ("len", ctypes.c_ushort),
        ("filter", ctypes.POINTER(_SockFilter)),
    ]


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickle primitive control messages without importing or calling code."""

    def find_class(self, module: str, name: str) -> object:
        raise pickle.UnpicklingError(
            f"global references are forbidden: {module}.{name}"
        )

    def persistent_load(self, persistent_id: object) -> object:
        raise pickle.UnpicklingError("persistent references are forbidden")


def _restricted_loads(payload: bytes | bytearray | memoryview) -> object:
    stream = io.BytesIO(payload)
    value = _RestrictedUnpickler(stream).load()
    if stream.read(1):
        raise pickle.UnpicklingError("trailing pickle data is forbidden")
    return value


def _install_restricted_connection_unpickler() -> None:
    if getattr(_ConnectionBase.recv, "_sciprobe_restricted_unpickler", False):
        os.environ[_RESTRICTED_UNPICKLER_ENV] = "1"
        return

    def restricted_recv(self: _ConnectionBase) -> object:
        try:
            payload = self.recv_bytes(maxlength=_MAX_CONTROL_MESSAGE_BYTES)
            return _restricted_loads(payload)
        except Exception:
            self.close()
            raise

    restricted_recv._sciprobe_restricted_unpickler = True
    _ConnectionBase.recv = restricted_recv
    os.environ[_RESTRICTED_UNPICKLER_ENV] = "1"


def _fatal(message: str) -> None:
    try:
        os.write(2, f"SciProbe seccomp gate: {message}\n".encode("utf-8"))
    finally:
        os._exit(78)


def _architecture() -> tuple[str, dict[str, object]]:
    if sys.platform != "linux" or sys.byteorder != "little":
        raise RuntimeError("requires little-endian Linux")
    machine = platform.machine().lower()
    aliases = {
        "amd64": "x86_64",
        "arm64": "aarch64",
    }
    machine = aliases.get(machine, machine)
    try:
        return machine, _ARCH_CONFIG[machine]
    except KeyError as error:
        raise RuntimeError(f"unsupported architecture: {machine}") from error


def _assemble_filter(config: dict[str, object]) -> ctypes.Array[_SockFilter]:
    """Assemble a forward-only classic BPF program with named jump labels."""
    specs: list[tuple[object, ...]] = [
        ("stmt", _BPF_LD_W_ABS, _SECCOMP_DATA_ARCH_OFFSET),
        (
            "jump",
            _BPF_JMP_JEQ_K,
            int(config["audit_arch"]),
            "load_nr",
            "kill",
        ),
        ("label", "kill"),
        ("stmt", _BPF_RET_K, _SECCOMP_RET_KILL_PROCESS),
        ("label", "load_nr"),
        ("stmt", _BPF_LD_W_ABS, _SECCOMP_DATA_NR_OFFSET),
    ]
    if bool(config["deny_x32"]):
        specs.append(
            (
                "jump",
                _BPF_JMP_JSET_K,
                _X32_SYSCALL_BIT,
                "deny_process",
                "check_socket",
            )
        )
    specs.extend(
        [
            ("label", "check_socket"),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                int(config["socket"]),
                "deny_network",
                "check_socketpair",
            ),
            ("label", "check_socketpair"),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                int(config["socketpair"]),
                "load_domain",
                "check_connect",
            ),
            ("label", "check_connect"),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                int(config["connect"]),
                "deny_network",
                "check_bind",
            ),
            ("label", "check_bind"),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                int(config["bind"]),
                "deny_network",
                "check_listen",
            ),
            ("label", "check_listen"),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                int(config["listen"]),
                "deny_network",
                "check_accept",
            ),
            ("label", "check_accept"),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                int(config["accept"]),
                "deny_network",
                "check_accept4",
            ),
            ("label", "check_accept4"),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                int(config["accept4"]),
                "deny_network",
                "check_sendto",
            ),
            ("label", "check_sendto"),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                int(config["sendto"]),
                "load_sendto_dest_low",
                "check_sendmsg",
            ),
            ("label", "check_sendmsg"),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                int(config["sendmsg"]),
                "deny_network",
                "check_sendmmsg",
            ),
            ("label", "check_sendmmsg"),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                int(config["sendmmsg"]),
                "deny_network",
                "check_io_uring",
            ),
            ("label", "check_io_uring"),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                int(config["io_uring_setup"]),
                "deny_network",
                "check_clone",
            ),
            ("label", "check_clone"),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                int(config["clone"]),
                "load_clone_flags",
                "check_clone3",
            ),
            ("label", "check_clone3"),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                int(config["clone3"]),
                "deny_not_supported",
                "check_self_pid_0",
            ),
        ]
    )
    pid_zero_allowed = tuple(int(value) for value in config["pid_zero_allowed"])
    priority_self_only = tuple(int(value) for value in config["priority_self_only"])
    ioprio_self_only = tuple(int(value) for value in config["ioprio_self_only"])
    if not pid_zero_allowed or not priority_self_only or not ioprio_self_only:
        raise RuntimeError("self-only process syscall lists cannot be empty")
    for index, syscall_number in enumerate(pid_zero_allowed):
        next_label = (
            f"check_self_pid_{index + 1}"
            if index + 1 < len(pid_zero_allowed)
            else "check_priority_0"
        )
        specs.extend(
            [
                ("label", f"check_self_pid_{index}"),
                (
                    "jump",
                    _BPF_JMP_JEQ_K,
                    syscall_number,
                    "load_self_pid_low",
                    next_label,
                ),
            ]
        )
    for index, syscall_number in enumerate(priority_self_only):
        next_label = (
            f"check_priority_{index + 1}"
            if index + 1 < len(priority_self_only)
            else "check_ioprio_0"
        )
        specs.extend(
            [
                ("label", f"check_priority_{index}"),
                (
                    "jump",
                    _BPF_JMP_JEQ_K,
                    syscall_number,
                    "load_priority_selector",
                    next_label,
                ),
            ]
        )
    for index, syscall_number in enumerate(ioprio_self_only):
        next_label = (
            f"check_ioprio_{index + 1}"
            if index + 1 < len(ioprio_self_only)
            else "check_process_0"
        )
        specs.extend(
            [
                ("label", f"check_ioprio_{index}"),
                (
                    "jump",
                    _BPF_JMP_JEQ_K,
                    syscall_number,
                    "load_ioprio_selector",
                    next_label,
                ),
            ]
        )
    process_denied = tuple(int(value) for value in config["process_denied"])
    if not process_denied:
        raise RuntimeError("process syscall deny list cannot be empty")
    for index, syscall_number in enumerate(process_denied):
        next_label = (
            f"check_process_{index + 1}" if index + 1 < len(process_denied) else "allow"
        )
        specs.extend(
            [
                ("label", f"check_process_{index}"),
                (
                    "jump",
                    _BPF_JMP_JEQ_K,
                    syscall_number,
                    "deny_process",
                    next_label,
                ),
            ]
        )
    specs.extend(
        [
            ("label", "load_domain"),
            ("stmt", _BPF_LD_W_ABS, _SECCOMP_DATA_ARG0_OFFSET),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                _AF_UNIX,
                "allow",
                "deny_network",
            ),
            ("label", "load_sendto_dest_low"),
            ("stmt", _BPF_LD_W_ABS, _SECCOMP_DATA_ARG4_LOW_OFFSET),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                0,
                "load_sendto_dest_high",
                "deny_network",
            ),
            ("label", "load_sendto_dest_high"),
            ("stmt", _BPF_LD_W_ABS, _SECCOMP_DATA_ARG4_HIGH_OFFSET),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                0,
                "allow",
                "deny_network",
            ),
            ("label", "load_clone_flags"),
            ("stmt", _BPF_LD_W_ABS, _SECCOMP_DATA_ARG0_OFFSET),
            (
                "jump",
                _BPF_JMP_JSET_K,
                _CLONE_THREAD,
                "allow",
                "deny_process",
            ),
            ("label", "load_self_pid_low"),
            ("stmt", _BPF_LD_W_ABS, _SECCOMP_DATA_ARG0_OFFSET),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                0,
                "load_self_pid_high",
                "deny_process",
            ),
            ("label", "load_self_pid_high"),
            ("stmt", _BPF_LD_W_ABS, _SECCOMP_DATA_ARG0_HIGH_OFFSET),
            ("jump", _BPF_JMP_JEQ_K, 0, "allow", "deny_process"),
            ("label", "load_priority_selector"),
            ("stmt", _BPF_LD_W_ABS, _SECCOMP_DATA_ARG0_OFFSET),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                0,
                "load_priority_selector_high",
                "deny_process",
            ),
            ("label", "load_priority_selector_high"),
            ("stmt", _BPF_LD_W_ABS, _SECCOMP_DATA_ARG0_HIGH_OFFSET),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                0,
                "load_priority_who_low",
                "deny_process",
            ),
            ("label", "load_priority_who_low"),
            ("stmt", _BPF_LD_W_ABS, _SECCOMP_DATA_ARG1_OFFSET),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                0,
                "load_priority_who_high",
                "deny_process",
            ),
            ("label", "load_priority_who_high"),
            ("stmt", _BPF_LD_W_ABS, _SECCOMP_DATA_ARG1_HIGH_OFFSET),
            ("jump", _BPF_JMP_JEQ_K, 0, "allow", "deny_process"),
            ("label", "load_ioprio_selector"),
            ("stmt", _BPF_LD_W_ABS, _SECCOMP_DATA_ARG0_OFFSET),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                _IOPRIO_WHO_PROCESS,
                "load_ioprio_selector_high",
                "deny_process",
            ),
            ("label", "load_ioprio_selector_high"),
            ("stmt", _BPF_LD_W_ABS, _SECCOMP_DATA_ARG0_HIGH_OFFSET),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                0,
                "load_ioprio_who_low",
                "deny_process",
            ),
            ("label", "load_ioprio_who_low"),
            ("stmt", _BPF_LD_W_ABS, _SECCOMP_DATA_ARG1_OFFSET),
            (
                "jump",
                _BPF_JMP_JEQ_K,
                0,
                "load_ioprio_who_high",
                "deny_process",
            ),
            ("label", "load_ioprio_who_high"),
            ("stmt", _BPF_LD_W_ABS, _SECCOMP_DATA_ARG1_HIGH_OFFSET),
            ("jump", _BPF_JMP_JEQ_K, 0, "allow", "deny_process"),
            ("label", "deny_network"),
            (
                "stmt",
                _BPF_RET_K,
                _SECCOMP_RET_ERRNO | errno.ENETUNREACH,
            ),
            ("label", "deny_process"),
            (
                "stmt",
                _BPF_RET_K,
                _SECCOMP_RET_ERRNO | errno.EPERM,
            ),
            # Returning ENOSYS for clone3 makes glibc fall back to clone.  The
            # clone rule then permits CLONE_THREAD only, keeping normal Python
            # threads usable while denying process creation.
            ("label", "deny_not_supported"),
            (
                "stmt",
                _BPF_RET_K,
                _SECCOMP_RET_ERRNO | errno.ENOSYS,
            ),
            ("label", "allow"),
            ("stmt", _BPF_RET_K, _SECCOMP_RET_ALLOW),
        ]
    )

    labels: dict[str, int] = {}
    instruction_index = 0
    for spec in specs:
        if spec[0] == "label":
            labels[str(spec[1])] = instruction_index
        else:
            instruction_index += 1

    instructions: list[_SockFilter] = []
    for spec in specs:
        kind = spec[0]
        if kind == "label":
            continue
        if kind == "stmt":
            instructions.append(
                _SockFilter(code=int(spec[1]), jt=0, jf=0, k=int(spec[2]))
            )
            continue
        current = len(instructions)
        true_offset = labels[str(spec[3])] - current - 1
        false_offset = labels[str(spec[4])] - current - 1
        if not 0 <= true_offset <= 255 or not 0 <= false_offset <= 255:
            raise RuntimeError("invalid seccomp BPF jump")
        instructions.append(
            _SockFilter(
                code=int(spec[1]),
                jt=true_offset,
                jf=false_offset,
                k=int(spec[2]),
            )
        )

    program_type = _SockFilter * len(instructions)
    return program_type(*instructions)


def _install_seccomp_filter() -> None:
    _, config = _architecture()
    filters = _assemble_filter(config)
    program = _SockFprog(
        len=len(filters),
        filter=ctypes.cast(filters, ctypes.POINTER(_SockFilter)),
    )
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    libc.prctl.restype = ctypes.c_int

    ctypes.set_errno(0)
    result = libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))

    ctypes.set_errno(0)
    result = libc.prctl(
        _PR_SET_SECCOMP,
        _SECCOMP_MODE_FILTER,
        ctypes.addressof(program),
        0,
        0,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    os.environ[_FILTER_ACTIVE_ENV] = "1"
    os.environ[_PROCESS_FILTER_ACTIVE_ENV] = "1"


def _minimum_landlock_abi() -> int:
    """Lowest Landlock ABI this sandbox will accept.

    Defaults to the strict floor. A cluster whose kernel predates an ABI can
    lower it, but only by saying so explicitly, because doing so gives up real
    controls and the run should not decide that quietly on its own.
    """
    configured = os.environ.get(_MIN_LANDLOCK_ABI_ENV, "").strip()
    if not configured:
        return _DEFAULT_MIN_LANDLOCK_ABI
    try:
        minimum = int(configured)
    except ValueError:
        raise RuntimeError(
            f"{_MIN_LANDLOCK_ABI_ENV} must be an integer, got {configured!r}"
        ) from None
    if not 1 <= minimum <= _DEFAULT_MIN_LANDLOCK_ABI:
        raise RuntimeError(
            f"{_MIN_LANDLOCK_ABI_ENV} must be between 1 and "
            f"{_DEFAULT_MIN_LANDLOCK_ABI}, got {minimum}"
        )
    return minimum


def _landlock_handled_access(abi_version: int) -> int:
    minimum = _minimum_landlock_abi()
    if abi_version < minimum:
        raise RuntimeError(
            "Landlock ABI does not support safe truncation isolation: "
            f"required >= {minimum}, found {abi_version}"
        )
    # Record what the kernel could actually enforce. Read confinement and the
    # write, create and delete controls are all ABI 1, so the properties this
    # sandbox depends on to keep a grader unreadable survive an older kernel.
    # What is lost below ABI 3 is rename and truncate confinement, which matter
    # for paths the model can already open for writing.
    missing = []
    if abi_version < 2:
        missing.append("refer")
    if abi_version < 3:
        missing.append("truncate")
    if abi_version < 5:
        missing.append("ioctl_dev")
    os.environ[_LANDLOCK_EFFECTIVE_ABI_ENV] = str(abi_version)
    os.environ[_LANDLOCK_MISSING_CONTROLS_ENV] = ",".join(missing)
    access = (
        _LANDLOCK_ACCESS_FS_EXECUTE
        | _LANDLOCK_ACCESS_FS_WRITE_FILE
        | _LANDLOCK_ACCESS_FS_READ_FILE
        | _LANDLOCK_ACCESS_FS_READ_DIR
        | _LANDLOCK_ACCESS_FS_REMOVE_DIR
        | _LANDLOCK_ACCESS_FS_REMOVE_FILE
        | _LANDLOCK_ACCESS_FS_MAKE_CHAR
        | _LANDLOCK_ACCESS_FS_MAKE_DIR
        | _LANDLOCK_ACCESS_FS_MAKE_REG
        | _LANDLOCK_ACCESS_FS_MAKE_SOCK
        | _LANDLOCK_ACCESS_FS_MAKE_FIFO
        | _LANDLOCK_ACCESS_FS_MAKE_BLOCK
        | _LANDLOCK_ACCESS_FS_MAKE_SYM
    )
    if abi_version >= 2:
        access |= _LANDLOCK_ACCESS_FS_REFER
    if abi_version >= 3:
        access |= _LANDLOCK_ACCESS_FS_TRUNCATE
    if abi_version >= 5:
        access |= _LANDLOCK_ACCESS_FS_IOCTL_DEV
    return access


def _landlock_syscall(
    libc: ctypes.CDLL, syscall_number: int, *arguments: object
) -> int:
    ctypes.set_errno(0)
    result = int(libc.syscall(ctypes.c_long(syscall_number), *arguments))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def _private_session_directory(root: str, label: str) -> str:
    """Create an unpredictable, process-private directory under a global tmpfs."""
    root_path = os.path.realpath(root)
    root_stat = os.stat(root_path, follow_symlinks=False)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise RuntimeError(f"session root is not a directory: {root_path}")

    base_path = os.path.join(root_path, ".sciprobe-shells")
    try:
        os.mkdir(base_path, 0o700)
    except FileExistsError:
        pass
    base_stat = os.stat(base_path, follow_symlinks=False)
    if (
        not stat.S_ISDIR(base_stat.st_mode)
        or base_stat.st_uid != os.geteuid()
        or stat.S_IMODE(base_stat.st_mode) != 0o700
    ):
        raise RuntimeError(f"unsafe shared session directory: {base_path}")

    base_fd = os.open(
        base_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        for _ in range(16):
            token = os.urandom(16).hex()
            name = f"{os.getpid()}-{label}-{token}"
            try:
                os.mkdir(name, 0o700, dir_fd=base_fd)
            except FileExistsError:
                continue
            return os.path.join(base_path, name)
    finally:
        os.close(base_fd)
    raise RuntimeError(f"could not allocate private {label} directory")


def _configured_readonly_paths() -> list[str]:
    candidates = [
        "/usr",
        "/lib",
        "/lib64",
        "/bin",
        "/sbin",
        "/etc",
        "/opt",
        "/app",
        # The task contract starts each notebook in this directory and then
        # reads its sole ``data`` child.  The production isolation preflight
        # fails unless that is the complete directory listing.
        "/workspace/sciprobe-probe",
        "/workspace/sciprobe-seccomp-hook",
        "/proc/self",
        sys.prefix,
        sys.base_prefix,
        os.path.dirname(sys.executable),
    ]
    protected_roots = tuple(
        os.path.realpath(path) for path in ("/tmp", "/var/tmp", "/dev/shm")
    )
    for path in sys.path:
        if not path or not os.path.exists(path):
            continue
        canonical = os.path.realpath(path)
        overlaps_private_state = any(
            canonical == root
            or canonical.startswith(root + os.sep)
            or root.startswith(canonical + os.sep)
            for root in protected_roots
        )
        if canonical != os.sep and not overlaps_private_state:
            candidates.append(canonical)

    configured = os.environ.get(_READONLY_PATHS_ENV, "")
    for path in configured.split(os.pathsep):
        if not path:
            continue
        if not os.path.isabs(path):
            raise RuntimeError(f"Landlock read-only path is not absolute: {path}")
        if not os.path.exists(path):
            raise RuntimeError(f"Landlock read-only path does not exist: {path}")
        canonical = os.path.realpath(path)
        exposes_private_state = canonical == os.sep or any(
            canonical == root or root.startswith(canonical + os.sep)
            for root in protected_roots
        )
        if exposes_private_state:
            raise RuntimeError(
                f"Landlock read-only path exposes private session roots: {path}"
            )
        candidates.append(canonical)

    resolved: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        canonical = os.path.realpath(path)
        if canonical in seen:
            continue
        seen.add(canonical)
        resolved.append(canonical)
    return resolved


def _add_landlock_path_rule(
    *,
    libc: ctypes.CDLL,
    config: dict[str, object],
    ruleset_fd: int,
    path: str,
    allowed_access: int,
) -> None:
    path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        path_stat = os.fstat(path_fd)
        if stat.S_ISDIR(path_stat.st_mode):
            compatible_access = allowed_access
        else:
            compatible_access = allowed_access & (
                _LANDLOCK_ACCESS_FS_EXECUTE
                | _LANDLOCK_ACCESS_FS_WRITE_FILE
                | _LANDLOCK_ACCESS_FS_READ_FILE
                | _LANDLOCK_ACCESS_FS_TRUNCATE
                | _LANDLOCK_ACCESS_FS_IOCTL_DEV
            )
        attribute = _LandlockPathBeneathAttr(
            allowed_access=compatible_access,
            parent_fd=path_fd,
            reserved=0,
        )
        _landlock_syscall(
            libc,
            int(config["landlock_add_rule"]),
            ctypes.c_int(ruleset_fd),
            ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
            ctypes.byref(attribute),
            ctypes.c_uint(0),
        )
    finally:
        os.close(path_fd)


def _install_landlock_filter() -> None:
    """Give each shell read-only runtime access and private writable state."""
    _, config = _architecture()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    abi_version = _landlock_syscall(
        libc,
        int(config["landlock_create_ruleset"]),
        ctypes.c_void_p(),
        ctypes.c_size_t(0),
        ctypes.c_uint(_LANDLOCK_CREATE_RULESET_VERSION),
    )
    handled_access = _landlock_handled_access(abi_version)

    previous_umask = os.umask(0o077)
    try:
        state_directory = _private_session_directory("/tmp", "state")
        var_tmp_directory = _private_session_directory("/var/tmp", "state")
        shm_directory = _private_session_directory("/dev/shm", "state")
        home_directory = os.path.join(state_directory, "home")
        temp_directory = os.path.join(state_directory, "tmp")
        os.mkdir(home_directory, 0o700)
        os.mkdir(temp_directory, 0o700)
    finally:
        os.umask(previous_umask)

    ruleset_attribute = _LandlockRulesetAttr(handled_access_fs=handled_access)
    ruleset_fd = _landlock_syscall(
        libc,
        int(config["landlock_create_ruleset"]),
        ctypes.byref(ruleset_attribute),
        ctypes.sizeof(ruleset_attribute),
        ctypes.c_uint(0),
    )
    read_access = (
        _LANDLOCK_ACCESS_FS_EXECUTE
        | _LANDLOCK_ACCESS_FS_READ_FILE
        | _LANDLOCK_ACCESS_FS_READ_DIR
    ) & handled_access
    write_access = (
        read_access
        | _LANDLOCK_ACCESS_FS_WRITE_FILE
        | _LANDLOCK_ACCESS_FS_REMOVE_DIR
        | _LANDLOCK_ACCESS_FS_REMOVE_FILE
        | _LANDLOCK_ACCESS_FS_MAKE_DIR
        | _LANDLOCK_ACCESS_FS_MAKE_REG
        | _LANDLOCK_ACCESS_FS_MAKE_SOCK
        | _LANDLOCK_ACCESS_FS_MAKE_FIFO
        | _LANDLOCK_ACCESS_FS_MAKE_SYM
        | _LANDLOCK_ACCESS_FS_REFER
        | _LANDLOCK_ACCESS_FS_TRUNCATE
    ) & handled_access
    device_access = (
        _LANDLOCK_ACCESS_FS_READ_FILE
        | _LANDLOCK_ACCESS_FS_WRITE_FILE
        | _LANDLOCK_ACCESS_FS_IOCTL_DEV
    ) & handled_access
    try:
        for path in _configured_readonly_paths():
            _add_landlock_path_rule(
                libc=libc,
                config=config,
                ruleset_fd=ruleset_fd,
                path=path,
                allowed_access=read_access,
            )
        for path in (
            state_directory,
            var_tmp_directory,
            shm_directory,
        ):
            _add_landlock_path_rule(
                libc=libc,
                config=config,
                ruleset_fd=ruleset_fd,
                path=path,
                allowed_access=write_access,
            )
        for path in (
            "/dev/null",
            "/dev/zero",
            "/dev/full",
            "/dev/random",
            "/dev/urandom",
        ):
            if os.path.exists(path):
                _add_landlock_path_rule(
                    libc=libc,
                    config=config,
                    ruleset_fd=ruleset_fd,
                    path=path,
                    allowed_access=device_access,
                )
        _landlock_syscall(
            libc,
            int(config["landlock_restrict_self"]),
            ctypes.c_int(ruleset_fd),
            ctypes.c_uint(0),
        )
    finally:
        os.close(ruleset_fd)

    os.environ[_SESSION_STATE_ENV] = state_directory
    os.environ[_SESSION_VAR_TMP_ENV] = var_tmp_directory
    os.environ[_SESSION_SHM_ENV] = shm_directory
    os.environ["HOME"] = home_directory
    os.environ["TMPDIR"] = temp_directory
    os.environ["TMP"] = temp_directory
    os.environ["TEMP"] = temp_directory
    os.environ["XDG_CACHE_HOME"] = os.path.join(home_directory, ".cache")
    os.environ["MPLCONFIGDIR"] = os.path.join(home_directory, ".matplotlib")
    tempfile_module = sys.modules.get("tempfile")
    if tempfile_module is not None:
        tempfile_module.tempdir = temp_directory
    os.chdir(state_directory)
    os.umask(0o077)
    os.environ[_LANDLOCK_ACTIVE_ENV] = "1"


def _socket_domain(libc: ctypes.CDLL, fd: int) -> int | None:
    """Return a socket's address family, or ``None`` for a non-socket FD."""
    domain = ctypes.c_int()
    domain_size = ctypes.c_uint(ctypes.sizeof(domain))
    ctypes.set_errno(0)
    result = libc.getsockopt(
        fd,
        _SOL_SOCKET,
        _SO_DOMAIN,
        ctypes.byref(domain),
        ctypes.byref(domain_size),
    )
    if result == 0:
        if domain_size.value != ctypes.sizeof(domain):
            raise RuntimeError(f"unexpected SO_DOMAIN size for fd {fd}")
        return int(domain.value)
    error = ctypes.get_errno()
    if error in {errno.EBADF, errno.ENOTSOCK}:
        return None
    raise OSError(error, f"getsockopt(SO_DOMAIN) failed for fd {fd}")


def _open_file_descriptors() -> list[int]:
    try:
        entries = os.listdir("/proc/self/fd")
    except OSError as error:
        raise RuntimeError("cannot inspect inherited file descriptors") from error
    descriptors: list[int] = []
    for entry in entries:
        try:
            descriptor = int(entry)
        except ValueError:
            continue
        if descriptor > 2:
            descriptors.append(descriptor)
    return descriptors


def _connection_file_descriptors(value: object) -> set[int]:
    """Find trusted multiprocessing control connections in process arguments."""
    descriptors: set[int] = set()
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(current, Connection):
            descriptors.add(current.fileno())
        elif isinstance(current, dict):
            pending.extend(current.keys())
            pending.extend(current.values())
        elif isinstance(current, (list, tuple, set, frozenset)):
            pending.extend(current)
    return descriptors


def _close_inherited_non_control_descriptors(
    control_descriptors: set[int],
) -> None:
    """Keep only stdio and the current shell's trusted control connections."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.getsockopt.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
    ]
    libc.getsockopt.restype = ctypes.c_int

    for descriptor in _open_file_descriptors():
        if descriptor in control_descriptors:
            domain = _socket_domain(libc, descriptor)
            if domain is not None and domain != _AF_UNIX:
                raise RuntimeError("multiprocessing control socket is not AF_UNIX")
            continue
        try:
            os.close(descriptor)
        except OSError as error:
            if error.errno != errno.EBADF:
                raise

    remaining = []
    for descriptor in _open_file_descriptors():
        try:
            os.fstat(descriptor)
        except OSError as error:
            if error.errno == errno.EBADF:
                continue
            raise
        if descriptor not in control_descriptors:
            remaining.append(descriptor)
    if remaining:
        raise RuntimeError(f"untrusted inherited descriptors remain: {remaining}")
    os.environ[_INHERITED_FDS_CLEAN_ENV] = "1"


def _install_bootstrap_hook() -> None:
    if getattr(BaseProcess._bootstrap, "_sciprobe_seccomp_hook", False):
        return
    original_bootstrap = BaseProcess._bootstrap

    def bootstrap_with_seccomp(self, *args, **kwargs):
        try:
            control_descriptors = _connection_file_descriptors(
                (getattr(self, "_args", ()), getattr(self, "_kwargs", {}))
            )
            _install_seccomp_filter()
            _close_inherited_non_control_descriptors(control_descriptors)
            _install_landlock_filter()
        except BaseException as error:
            _fatal(f"filter installation failed: {type(error).__name__}: {error}")
        return original_bootstrap(self, *args, **kwargs)

    bootstrap_with_seccomp._sciprobe_seccomp_hook = True
    BaseProcess._bootstrap = bootstrap_with_seccomp


if os.environ.get(_REQUIRED_ENV) == "1":
    try:
        _architecture()
        _install_restricted_connection_unpickler()
        _install_bootstrap_hook()
        os.environ[_HOOK_LOADED_ENV] = "1"
    except BaseException as error:
        _fatal(f"hook installation failed: {type(error).__name__}: {error}")
