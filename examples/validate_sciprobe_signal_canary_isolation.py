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

"""Prove the live SciProbe sandbox exposes data but no trusted state or network."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
import uuid
from typing import Any

EXPECTED_DATA_SHA256 = (
    "16713f67f959a4c276baea508c1fb64fa54bf622f4e14b0b4def77d6c152a590"
)
VISIBLE_ROOT = "/workspace/sciprobe-probe"
HIDDEN_NAMES = [
    "gold.json",
    "checks.py",
    "reference.py",
    "wrong_reference.py",
    "meta.json",
]
SENSITIVE_ENV_NAMES = [
    "COMMAND",
    "MOUNTS",
    "SCIPROBE_PRIVATE_PROBE_ROOT",
    "SCIPROBE_HOST_PRIVATE_PROBE_ROOT",
    "SCIPROBE_RUNTIME_DATASET_PATH",
    "SCIPROBE_CAPABILITY_STORE_PATH",
    "SCIPROBE_VERIFIER_TOKEN",
    "SCIPROBE_VERIFIER_CAPABILITY",
    "SCIPROBE_CAPABILITY_SIGNING_KEY",
    "SCIPROBE_TRUSTED_INGRESS_TOKEN",
    "SCIPROBE_POLICY_GENERATION_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "WANDB_API_KEY",
    "NVIDIA_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "RAY_AUTH_MODE",
    "RAY_AUTH_TOKEN",
    "RAY_AUTH_TOKEN_PATH",
]
SECCOMP_MARKERS = [
    "SCIPROBE_SECCOMP_HOOK_LOADED",
    "SCIPROBE_SECCOMP_FILTER_ACTIVE",
    "SCIPROBE_SECCOMP_PROCESS_FILTER_ACTIVE",
    "SCIPROBE_SECCOMP_INHERITED_FDS_CLEAN",
    "SCIPROBE_RESTRICTED_MP_UNPICKLER_ACTIVE",
    "SCIPROBE_LANDLOCK_FILTER_ACTIVE",
]
TRUSTED_SECRET_NAMES = [
    "SCIPROBE_VERIFIER_TOKEN",
    "SCIPROBE_CAPABILITY_SIGNING_KEY",
    "SCIPROBE_TRUSTED_INGRESS_TOKEN",
    "SCIPROBE_POLICY_GENERATION_TOKEN",
]


FIRST_SANDBOX_CODE = r"""import _socket
import ctypes
import errno
import hashlib
import json
import os
import platform
import resource
import shutil
import socket
import stat
import subprocess
import sys
import threading
from pathlib import Path

expected_errno = errno.ENETUNREACH
visible_root = Path("/workspace/sciprobe-probe")
data_root = visible_root / "data"
top_entries = sorted(path.name for path in visible_root.iterdir())
digest = hashlib.sha256()
files = 0
total_bytes = 0
symlinks = []
for path in sorted(data_root.rglob("*")):
    relative = path.relative_to(data_root).as_posix()
    if path.is_symlink():
        symlinks.append(relative)
        continue
    if path.is_dir():
        continue
    if not path.is_file():
        raise RuntimeError("non-regular data entry: " + relative)
    contents = path.read_bytes()
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(len(contents)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contents)
    digest.update(b"\0")
    files += 1
    total_bytes += len(contents)

hidden = {}
for name in ["gold.json", "checks.py", "reference.py", "wrong_reference.py", "meta.json"]:
    path = visible_root / name
    readable = True
    try:
        path.read_bytes()
    except OSError:
        readable = False
    hidden[name] = {"exists": path.exists(), "readable": readable}

sensitive_env_names = [
    "COMMAND",
    "MOUNTS",
    "SCIPROBE_PRIVATE_PROBE_ROOT",
    "SCIPROBE_HOST_PRIVATE_PROBE_ROOT",
    "SCIPROBE_RUNTIME_DATASET_PATH",
    "SCIPROBE_CAPABILITY_STORE_PATH",
    "SCIPROBE_VERIFIER_TOKEN",
    "SCIPROBE_VERIFIER_CAPABILITY",
    "SCIPROBE_CAPABILITY_SIGNING_KEY",
    "SCIPROBE_TRUSTED_INGRESS_TOKEN",
    "SCIPROBE_POLICY_GENERATION_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "WANDB_API_KEY",
    "NVIDIA_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "RAY_AUTH_MODE",
    "RAY_AUTH_TOKEN",
    "RAY_AUTH_TOKEN_PATH",
]
sensitive_env_present = sorted(
    name for name in sensitive_env_names if name in os.environ
)

trusted_driver_pid = __TRUSTED_DRIVER_PID__
target_proc = Path("/proc") / str(trusted_driver_pid)
try:
    target_environ = (target_proc / "environ").read_bytes()
    target_environ_readable = True
except OSError:
    target_environ = b""
    target_environ_readable = False
target_root_checkout_visible = (target_proc / "root/workspace/RL").exists()
sibling_sensitive_env_names = []
try:
    proc_entries = list(Path("/proc").iterdir())
except OSError as error:
    trusted_parent_proc_list_denied = error.errno in {
        errno.EACCES, errno.EPERM, errno.EXDEV
    }
    proc_entries = []
else:
    trusted_parent_proc_list_denied = False
for proc_entry in proc_entries:
    if not proc_entry.name.isdigit():
        continue
    try:
        contents = (proc_entry / "environ").read_bytes()
    except OSError:
        continue
    for name in sensitive_env_names:
        if (name + "=").encode("utf-8") in contents:
            sibling_sensitive_env_names.append(name)
sibling_sensitive_env_names = sorted(set(sibling_sensitive_env_names))

def denied(call):
    try:
        value = call()
    except OSError as error:
        return error.errno == expected_errno
    if hasattr(value, "close"):
        value.close()
    return False

machine = platform.machine().lower()
if machine in {"x86_64", "amd64"}:
    socket_syscall = 41
    io_uring_setup_syscall = 425
    process_syscalls = {
        "execve": 59, "ptrace": 101, "process_vm_readv": 310,
        "process_vm_writev": 311, "execveat": 322,
        "pidfd_send_signal": 424, "pidfd_open": 434,
        "clone3": 435, "pidfd_getfd": 438,
    }
elif machine in {"aarch64", "arm64"}:
    socket_syscall = 198
    io_uring_setup_syscall = 425
    process_syscalls = {
        "execve": 221, "ptrace": 117, "process_vm_readv": 270,
        "process_vm_writev": 271, "execveat": 281,
        "pidfd_send_signal": 424, "pidfd_open": 434,
        "clone3": 435, "pidfd_getfd": 438,
    }
else:
    raise RuntimeError("unsupported architecture: " + machine)

libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long

def direct_syscall_denied(number, *arguments):
    ctypes.set_errno(0)
    result = libc.syscall(ctypes.c_long(number), *arguments)
    return result == -1 and ctypes.get_errno() == expected_errno

class DerivedSocket(socket.socket):
    pass

native_socket_base = DerivedSocket.__mro__[1].__mro__[1]
python_ipv4_denied = denied(
    lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM)
)
python_ipv6_denied = denied(
    lambda: socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
)
native_socket_denied = denied(
    lambda: _socket.socket(socket.AF_INET, socket.SOCK_STREAM)
)
mro_base_denied = denied(
    lambda: native_socket_base(socket.AF_INET, socket.SOCK_STREAM)
)
raw_socket_denied = denied(
    lambda: socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
)
direct_socket_syscall_denied = direct_syscall_denied(
    socket_syscall,
    ctypes.c_long(socket.AF_INET),
    ctypes.c_long(socket.SOCK_STREAM),
    ctypes.c_long(0),
)
io_uring_setup_denied = direct_syscall_denied(
    io_uring_setup_syscall,
    ctypes.c_long(1),
    ctypes.c_void_p(),
)

def process_denied(number, *arguments):
    ctypes.set_errno(0)
    result = libc.syscall(ctypes.c_long(number), *arguments)
    return result == -1 and ctypes.get_errno() == errno.EPERM

try:
    subprocess.run(
        [sys.executable, "-I", "-c", "print(42)"],
        env={}, capture_output=True, text=True, timeout=5, check=False,
    )
except OSError as error:
    subprocess_creation_denied = error.errno == errno.EPERM
else:
    subprocess_creation_denied = False

direct_execve_denied = process_denied(
    process_syscalls["execve"], ctypes.c_char_p(b"/sciprobe-does-not-exist"),
    ctypes.c_void_p(), ctypes.c_void_p(),
)
direct_execveat_denied = process_denied(
    process_syscalls["execveat"], ctypes.c_long(-1),
    ctypes.c_char_p(b"sciprobe-does-not-exist"), ctypes.c_void_p(),
    ctypes.c_void_p(), ctypes.c_long(0),
)
ctypes.set_errno(0)
clone3_result = libc.syscall(
    ctypes.c_long(process_syscalls["clone3"]), ctypes.c_void_p(), ctypes.c_long(0)
)
clone3_denied = clone3_result == -1 and ctypes.get_errno() == errno.ENOSYS
ptrace_denied = process_denied(
    process_syscalls["ptrace"], ctypes.c_long(-1), ctypes.c_long(0),
    ctypes.c_void_p(), ctypes.c_void_p(),
)
process_vm_readv_denied = process_denied(
    process_syscalls["process_vm_readv"], ctypes.c_long(os.getpid()),
    ctypes.c_void_p(), ctypes.c_long(0), ctypes.c_void_p(),
    ctypes.c_long(0), ctypes.c_long(0),
)
process_vm_writev_denied = process_denied(
    process_syscalls["process_vm_writev"], ctypes.c_long(os.getpid()),
    ctypes.c_void_p(), ctypes.c_long(0), ctypes.c_void_p(),
    ctypes.c_long(0), ctypes.c_long(0),
)
pidfd_open_denied = process_denied(
    process_syscalls["pidfd_open"], ctypes.c_long(os.getpid()), ctypes.c_long(0)
)
pidfd_getfd_denied = process_denied(
    process_syscalls["pidfd_getfd"], ctypes.c_long(-1), ctypes.c_long(-1),
    ctypes.c_long(0),
)
pidfd_send_signal_denied = process_denied(
    process_syscalls["pidfd_send_signal"], ctypes.c_long(-1), ctypes.c_long(0),
    ctypes.c_void_p(), ctypes.c_long(0),
)
thread_result = []
thread = threading.Thread(target=lambda: thread_result.append(42))
thread.start()
thread.join(timeout=5)
thread_clone_allowed = not thread.is_alive() and thread_result == [42]

left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    left.sendall(b"unix-ok")
    unix_socketpair_allowed = right.recv(7) == b"unix-ok"
finally:
    left.close()
    right.close()

SOL_SOCKET = 1
SO_DOMAIN = 39
libc.getsockopt.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint),
]
libc.getsockopt.restype = ctypes.c_int

def socket_domain(descriptor):
    domain = ctypes.c_int()
    size = ctypes.c_uint(ctypes.sizeof(domain))
    ctypes.set_errno(0)
    result = libc.getsockopt(
        descriptor,
        SOL_SOCKET,
        SO_DOMAIN,
        ctypes.byref(domain),
        ctypes.byref(size),
    )
    if result == 0:
        return int(domain.value)
    if ctypes.get_errno() in {errno.EBADF, errno.ENOTSOCK}:
        return None
    raise OSError(ctypes.get_errno(), "getsockopt(SO_DOMAIN) failed")

socket_fds = []
open_fds = []
for entry in os.listdir("/proc/self/fd"):
    try:
        descriptor = int(entry)
    except ValueError:
        continue
    if descriptor <= 2:
        continue
    try:
        metadata = os.fstat(descriptor)
    except OSError as error:
        if error.errno == errno.EBADF:
            continue
        raise
    domain = socket_domain(descriptor)
    open_fds.append({
        "fd": descriptor,
        "kind": "socket" if domain is not None else stat.S_IFMT(metadata.st_mode),
        "domain": domain,
    })
    if domain is not None:
        socket_fds.append({"fd": descriptor, "domain": domain})
non_unix_socket_fds = [
    item for item in socket_fds if item["domain"] != socket.AF_UNIX
]

seccomp_mode = None
for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
    name, separator, value = line.partition(":")
    if separator and name == "Seccomp":
        seccomp_mode = int(value.strip())
        break

write_blocked = False
try:
    with (data_root / "crispresso_output" / "README_pipeline.txt").open("ab") as handle:
        handle.write(b"sandbox-write-probe")
except OSError:
    write_blocked = True

_sciprobe_isolation_state = {"calls": 1, "value": 40}
_sciprobe_state_file = (
    Path(os.environ["SCIPROBE_SESSION_STATE_DIR"])
    / "production-isolation-state.txt"
)
_sciprobe_state_file.write_text("state-1764", encoding="utf-8")

print(json.dumps({
    "top_entries": top_entries,
    "data_sha256": digest.hexdigest(),
    "data_files": files,
    "data_bytes": total_bytes,
    "symlinks": symlinks,
    "hidden": hidden,
    "private_root_exists": Path("/workspace/sciprobe-private").exists(),
    "training_checkout_exists": Path("/workspace/RL").exists(),
    "sensitive_env_present": sensitive_env_present,
    "trusted_driver_proc_visible": target_proc.exists(),
    "trusted_driver_environ_readable": target_environ_readable,
    "trusted_driver_root_checkout_visible": target_root_checkout_visible,
    "trusted_parent_proc_list_denied": trusted_parent_proc_list_denied,
    "sibling_sensitive_env_names": sibling_sensitive_env_names,
    "unshare_available": shutil.which("unshare") is not None,
    "hook_loaded": os.environ.get("SCIPROBE_SECCOMP_HOOK_LOADED") == "1",
    "filter_active": os.environ.get("SCIPROBE_SECCOMP_FILTER_ACTIVE") == "1",
    "process_filter_active": os.environ.get("SCIPROBE_SECCOMP_PROCESS_FILTER_ACTIVE") == "1",
    "inherited_fds_clean": os.environ.get("SCIPROBE_SECCOMP_INHERITED_FDS_CLEAN") == "1",
    "restricted_unpickler_active": os.environ.get("SCIPROBE_RESTRICTED_MP_UNPICKLER_ACTIVE") == "1",
    "landlock_active": os.environ.get("SCIPROBE_LANDLOCK_FILTER_ACTIVE") == "1",
    "state_file_path": str(_sciprobe_state_file),
    "state_file_persisted": _sciprobe_state_file.read_text(
        encoding="utf-8"
    ) == "state-1764",
    "seccomp_mode_filter": seccomp_mode == 2,
    "python_ipv4_denied": python_ipv4_denied,
    "python_ipv6_denied": python_ipv6_denied,
    "native_socket_denied": native_socket_denied,
    "mro_base_denied": mro_base_denied,
    "raw_socket_denied": raw_socket_denied,
    "direct_socket_syscall_denied": direct_socket_syscall_denied,
    "io_uring_setup_denied": io_uring_setup_denied,
    "subprocess_creation_denied": subprocess_creation_denied,
    "direct_execve_denied": direct_execve_denied,
    "direct_execveat_denied": direct_execveat_denied,
    "clone3_denied": clone3_denied,
    "ptrace_denied": ptrace_denied,
    "process_vm_readv_denied": process_vm_readv_denied,
    "process_vm_writev_denied": process_vm_writev_denied,
    "pidfd_open_denied": pidfd_open_denied,
    "pidfd_getfd_denied": pidfd_getfd_denied,
    "pidfd_send_signal_denied": pidfd_send_signal_denied,
    "thread_clone_allowed": thread_clone_allowed,
    "unix_socketpair_allowed": unix_socketpair_allowed,
    "shell_pid": os.getpid(),
    "open_fd_count": len(open_fds),
    "open_fds": open_fds,
    "socket_fd_count": len(socket_fds),
    "non_unix_socket_fd_count": len(non_unix_socket_fds),
    "write_blocked": write_blocked,
    "state_value": _sciprobe_isolation_state["value"],
}, sort_keys=True))
"""


CROSS_SESSION_SANDBOX_CODE = r"""import ctypes
import errno
import json
import os
import platform
import resource
import socket
import stat
from pathlib import Path

writer_file = Path(__WRITER_FILE__)
writer_pid = __WRITER_PID__
writer_directory = writer_file.parent
own_directory = Path(os.environ["SCIPROBE_SESSION_STATE_DIR"])
own_file = own_directory / "reader-state.txt"

def filesystem_denied(call):
    try:
        value = call()
    except OSError as error:
        return error.errno in {errno.EACCES, errno.EPERM, errno.EXDEV}
    if hasattr(value, "close"):
        value.close()
    return False

def signal_denied(process_id):
    try:
        os.kill(process_id, 0)
    except OSError as error:
        return error.errno == errno.EPERM
    return False

machine = platform.machine().lower()
if machine in {"x86_64", "amd64"}:
    syscalls = {
        "shmget": 29, "shmat": 30, "semget": 64, "msgget": 68,
        "rt_sigqueueinfo": 129, "getpriority": 140, "setpriority": 141,
        "sched_setparam": 142, "sched_setscheduler": 144, "tkill": 200,
        "sched_setaffinity": 203, "sched_getaffinity": 204, "tgkill": 234,
        "ioprio_set": 251, "ioprio_get": 252,
        "rt_tgsigqueueinfo": 297, "prlimit64": 302,
        "sched_setattr": 314, "process_madvise": 440,
        "process_mrelease": 448,
    }
elif machine in {"aarch64", "arm64"}:
    syscalls = {
        "ioprio_set": 30, "ioprio_get": 31, "sched_setparam": 118,
        "sched_setscheduler": 119, "sched_setaffinity": 122,
        "sched_getaffinity": 123, "tkill": 130, "tgkill": 131,
        "rt_sigqueueinfo": 138, "setpriority": 140, "getpriority": 141,
        "msgget": 186, "semget": 190, "shmget": 194, "shmat": 196,
        "rt_tgsigqueueinfo": 240, "prlimit64": 261,
        "sched_setattr": 274, "process_madvise": 440,
        "process_mrelease": 448,
    }
else:
    raise RuntimeError("unsupported architecture: " + machine)

libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
libc.getsockopt.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint),
]
libc.getsockopt.restype = ctypes.c_int

def process_denied(name, *arguments):
    ctypes.set_errno(0)
    result = libc.syscall(ctypes.c_long(syscalls[name]), *arguments)
    return result == -1 and ctypes.get_errno() == errno.EPERM

def socket_domain(descriptor):
    domain = ctypes.c_int()
    size = ctypes.c_uint(ctypes.sizeof(domain))
    ctypes.set_errno(0)
    result = libc.getsockopt(
        descriptor, 1, 39, ctypes.byref(domain), ctypes.byref(size)
    )
    if result == 0:
        return int(domain.value)
    if ctypes.get_errno() in {errno.EBADF, errno.ENOTSOCK}:
        return None
    raise OSError(ctypes.get_errno(), "getsockopt(SO_DOMAIN) failed")

open_fds = []
for entry in os.listdir("/proc/self/fd"):
    try:
        descriptor = int(entry)
    except ValueError:
        continue
    if descriptor <= 2:
        continue
    try:
        metadata = os.fstat(descriptor)
    except OSError as error:
        if error.errno == errno.EBADF:
            continue
        raise
    domain = socket_domain(descriptor)
    open_fds.append({
        "fd": descriptor,
        "kind": "socket" if domain is not None else stat.S_IFMT(metadata.st_mode),
        "domain": domain,
    })

socket_fds = [item for item in open_fds if item["domain"] is not None]

own_file.write_text("reader-state", encoding="utf-8")
global_roots = {
    "tmp": Path("/tmp"),
    "var_tmp": Path("/var/tmp"),
    "shm": Path("/dev/shm"),
}
proof = {
    "landlock_active": os.environ.get("SCIPROBE_LANDLOCK_FILTER_ACTIVE") == "1",
    "session_directories_distinct": own_directory != writer_directory,
    "own_state_write_allowed": own_file.read_text(encoding="utf-8") == "reader-state",
    "sibling_read_denied": filesystem_denied(lambda: writer_file.read_text(encoding="utf-8")),
    "sibling_write_denied": filesystem_denied(lambda: writer_file.write_text("stolen", encoding="utf-8")),
    "sibling_list_denied": filesystem_denied(lambda: list(writer_directory.iterdir())),
    "sibling_delete_denied": filesystem_denied(lambda: writer_file.unlink()),
    "sibling_signal_zero_denied": signal_denied(writer_pid),
    "all_processes_signal_zero_denied": signal_denied(-1),
    "sibling_tkill_denied": process_denied("tkill", ctypes.c_long(writer_pid), ctypes.c_long(0)),
    "sibling_tgkill_denied": process_denied("tgkill", ctypes.c_long(writer_pid), ctypes.c_long(writer_pid), ctypes.c_long(0)),
    "sibling_rt_sigqueueinfo_denied": process_denied("rt_sigqueueinfo", ctypes.c_long(writer_pid), ctypes.c_long(0), ctypes.c_void_p()),
    "sibling_rt_tgsigqueueinfo_denied": process_denied("rt_tgsigqueueinfo", ctypes.c_long(writer_pid), ctypes.c_long(writer_pid), ctypes.c_long(0), ctypes.c_void_p()),
    "sibling_prlimit_denied": process_denied("prlimit64", ctypes.c_long(writer_pid), ctypes.c_long(resource.RLIMIT_NOFILE), ctypes.c_void_p(), ctypes.c_void_p()),
    "sibling_sched_getaffinity_denied": process_denied("sched_getaffinity", ctypes.c_long(writer_pid), ctypes.c_long(0), ctypes.c_void_p()),
    "sibling_sched_setaffinity_denied": process_denied("sched_setaffinity", ctypes.c_long(writer_pid), ctypes.c_long(0), ctypes.c_void_p()),
    "sibling_sched_setparam_denied": process_denied("sched_setparam", ctypes.c_long(writer_pid), ctypes.c_void_p()),
    "sibling_sched_setscheduler_denied": process_denied("sched_setscheduler", ctypes.c_long(writer_pid), ctypes.c_long(-1), ctypes.c_void_p()),
    "sibling_sched_setattr_denied": process_denied("sched_setattr", ctypes.c_long(writer_pid), ctypes.c_void_p(), ctypes.c_long(0)),
    "sibling_getpriority_denied": process_denied("getpriority", ctypes.c_long(os.PRIO_PROCESS), ctypes.c_long(writer_pid)),
    "sibling_setpriority_denied": process_denied("setpriority", ctypes.c_long(-1), ctypes.c_long(writer_pid), ctypes.c_long(0)),
    "sibling_ioprio_get_denied": process_denied("ioprio_get", ctypes.c_long(1), ctypes.c_long(writer_pid)),
    "sibling_ioprio_set_denied": process_denied("ioprio_set", ctypes.c_long(-1), ctypes.c_long(writer_pid), ctypes.c_long(0)),
    "sysv_shmget_denied": process_denied("shmget", ctypes.c_long(0), ctypes.c_long(0), ctypes.c_long(0)),
    "sysv_shmat_denied": process_denied("shmat", ctypes.c_long(-1), ctypes.c_void_p(), ctypes.c_long(0)),
    "sysv_semget_denied": process_denied("semget", ctypes.c_long(0), ctypes.c_long(0), ctypes.c_long(0)),
    "sysv_msgget_denied": process_denied("msgget", ctypes.c_long(0), ctypes.c_long(0)),
    "process_madvise_denied": process_denied("process_madvise", ctypes.c_long(-1), ctypes.c_void_p(), ctypes.c_long(0), ctypes.c_long(0), ctypes.c_long(0)),
    "process_mrelease_denied": process_denied("process_mrelease", ctypes.c_long(-1), ctypes.c_long(0)),
    "open_fd_count": len(open_fds),
    "open_fds": open_fds,
    "socket_fd_count": len(socket_fds),
    "non_unix_socket_fd_count": sum(
        item["domain"] != socket.AF_UNIX for item in socket_fds
    ),
    "only_control_unix_fd": (
        len(socket_fds) == 1 and socket_fds[0]["domain"] == socket.AF_UNIX
    ),
}
for label, root in global_roots.items():
    proof[f"global_{label}_list_denied"] = filesystem_denied(lambda path=root: list(path.iterdir()))
    proof[f"global_{label}_create_denied"] = filesystem_denied(
        lambda path=root: (path / __GLOBAL_NAME__).write_text(
            "forbidden", encoding="utf-8"
        )
    )
print(json.dumps(proof, sort_keys=True))
"""


SECOND_SANDBOX_CODE = r"""import ctypes
import errno
import json
import os
import socket
import stat

_sciprobe_isolation_state["calls"] += 1
_sciprobe_isolation_state["value"] += 2

libc = ctypes.CDLL(None, use_errno=True)
SOL_SOCKET = 1
SO_DOMAIN = 39
libc.getsockopt.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint),
]
libc.getsockopt.restype = ctypes.c_int
socket_fds = []
open_fds = []
for entry in os.listdir("/proc/self/fd"):
    try:
        descriptor = int(entry)
    except ValueError:
        continue
    if descriptor <= 2:
        continue
    try:
        metadata = os.fstat(descriptor)
    except OSError as error:
        if error.errno == errno.EBADF:
            continue
        raise
    domain = ctypes.c_int()
    size = ctypes.c_uint(ctypes.sizeof(domain))
    ctypes.set_errno(0)
    result = libc.getsockopt(
        descriptor,
        SOL_SOCKET,
        SO_DOMAIN,
        ctypes.byref(domain),
        ctypes.byref(size),
    )
    if result == 0:
        resolved_domain = int(domain.value)
        socket_fds.append({"fd": descriptor, "domain": resolved_domain})
    elif ctypes.get_errno() not in {errno.EBADF, errno.ENOTSOCK}:
        raise OSError(ctypes.get_errno(), "getsockopt(SO_DOMAIN) failed")
    else:
        resolved_domain = None
    open_fds.append({
        "fd": descriptor,
        "kind": "socket" if resolved_domain is not None else stat.S_IFMT(metadata.st_mode),
        "domain": resolved_domain,
    })

print(json.dumps({
    "state_persisted": _sciprobe_isolation_state == {"calls": 2, "value": 42},
    "state_calls": _sciprobe_isolation_state["calls"],
    "state_value": _sciprobe_isolation_state["value"],
    "hook_loaded": os.environ.get("SCIPROBE_SECCOMP_HOOK_LOADED") == "1",
    "filter_active": os.environ.get("SCIPROBE_SECCOMP_FILTER_ACTIVE") == "1",
    "process_filter_active": os.environ.get("SCIPROBE_SECCOMP_PROCESS_FILTER_ACTIVE") == "1",
    "inherited_fds_clean": os.environ.get("SCIPROBE_SECCOMP_INHERITED_FDS_CLEAN") == "1",
    "restricted_unpickler_active": os.environ.get("SCIPROBE_RESTRICTED_MP_UNPICKLER_ACTIVE") == "1",
    "landlock_active": os.environ.get("SCIPROBE_LANDLOCK_FILTER_ACTIVE") == "1",
    "state_file_path": str(_sciprobe_state_file),
    "state_file_persisted": _sciprobe_state_file.read_text(
        encoding="utf-8"
    ) == "state-1764",
    "open_fd_count": len(open_fds),
    "open_fds": open_fds,
    "socket_fd_count": len(socket_fds),
    "non_unix_socket_fd_count": sum(
        item["domain"] != socket.AF_UNIX for item in socket_fds
    ),
}, sort_keys=True))
"""


def _request(
    url: str,
    *,
    method: str,
    session_id: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30,
) -> tuple[int, str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "X-Session-ID": session_id,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


def _execute(base_url: str, session_id: str, code: str) -> dict[str, Any]:
    status, body = _request(
        f"{base_url}/execute",
        method="POST",
        session_id=session_id,
        payload={
            "generated_code": code,
            "std_input": "",
            "timeout": 20,
            "language": "ipython",
            "max_output_characters": 10000,
            "traceback_verbosity": "Plain",
        },
    )
    assert status == 200, f"sandbox execute returned HTTP {status}: {body}"
    response = json.loads(body)
    assert response["process_status"] == "completed", response
    assert not response.get("stderr", "").strip(), response.get("stderr")
    proof = json.loads(response["stdout"].strip())
    assert isinstance(proof, dict)
    return proof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:6000")
    args = parser.parse_args()

    for name in TRUSTED_SECRET_NAMES:
        assert len(os.environ.get(name, "")) >= 32, f"{name} is not configured"
    for name in (
        "SCIPROBE_RUNTIME_DATASET_PATH",
        "SCIPROBE_CAPABILITY_STORE_PATH",
    ):
        assert os.environ.get(name), f"{name} is not configured"

    base_url = args.base_url.rstrip("/")
    session_id = str(uuid.uuid4())
    reader_session_id = str(uuid.uuid4())
    try:
        first_code = FIRST_SANDBOX_CODE.replace(
            "__TRUSTED_DRIVER_PID__", str(os.getpid())
        )
        first = _execute(base_url, session_id, first_code)
        second = _execute(base_url, session_id, SECOND_SANDBOX_CODE)
        writer_file = first.get("state_file_path")
        assert isinstance(writer_file, str) and writer_file
        writer_pid = first.get("shell_pid")
        assert isinstance(writer_pid, int) and writer_pid > 1
        cross_session_code = (
            CROSS_SESSION_SANDBOX_CODE.replace(
                "__WRITER_FILE__", json.dumps(writer_file)
            )
            .replace("__WRITER_PID__", str(writer_pid))
            .replace(
                "__GLOBAL_NAME__",
                json.dumps(f"sciprobe-denied-{uuid.uuid4().hex}"),
            )
        )
        cross_session = _execute(
            base_url,
            reader_session_id,
            cross_session_code,
        )
        parent_listener_accepted_later_call = True
    finally:
        for cleanup_session_id in (session_id, reader_session_id):
            delete_status, _ = _request(
                f"{base_url}/sessions/{cleanup_session_id}",
                method="DELETE",
                session_id=cleanup_session_id,
                timeout=15,
            )
            assert delete_status in {200, 404}, (
                f"sandbox session cleanup returned HTTP {delete_status}"
            )

    assert first["top_entries"] == ["data"], first["top_entries"]
    assert first["data_sha256"] == EXPECTED_DATA_SHA256
    assert first["data_files"] == 25
    assert first["data_bytes"] == 8137
    assert first["symlinks"] == []
    assert set(first["hidden"]) == set(HIDDEN_NAMES)
    assert all(
        not value["exists"] and not value["readable"]
        for value in first["hidden"].values()
    )
    assert first["private_root_exists"] is False
    assert first["training_checkout_exists"] is False
    assert first["sensitive_env_present"] == []
    assert first["trusted_driver_proc_visible"] is False
    assert first["trusted_driver_environ_readable"] is False
    assert first["trusted_driver_root_checkout_visible"] is False
    assert first["trusted_parent_proc_list_denied"] is True
    assert first["sibling_sensitive_env_names"] == []
    assert first["unshare_available"] is True
    for marker in (
        "hook_loaded",
        "filter_active",
        "process_filter_active",
        "inherited_fds_clean",
        "restricted_unpickler_active",
        "landlock_active",
        "seccomp_mode_filter",
        "python_ipv4_denied",
        "python_ipv6_denied",
        "native_socket_denied",
        "mro_base_denied",
        "raw_socket_denied",
        "direct_socket_syscall_denied",
        "io_uring_setup_denied",
        "subprocess_creation_denied",
        "direct_execve_denied",
        "direct_execveat_denied",
        "clone3_denied",
        "ptrace_denied",
        "process_vm_readv_denied",
        "process_vm_writev_denied",
        "pidfd_open_denied",
        "pidfd_getfd_denied",
        "pidfd_send_signal_denied",
        "thread_clone_allowed",
        "unix_socketpair_allowed",
        "write_blocked",
        "state_file_persisted",
    ):
        assert first[marker] is True, f"{marker} did not pass"
    assert first["state_value"] == 40
    assert first["socket_fd_count"] == 1
    assert first["non_unix_socket_fd_count"] == 0

    assert second["state_persisted"] is True
    assert second["state_calls"] == 2
    assert second["state_value"] == 42
    assert second["hook_loaded"] is True
    assert second["filter_active"] is True
    assert second["process_filter_active"] is True
    assert second["inherited_fds_clean"] is True
    assert second["restricted_unpickler_active"] is True
    assert second["landlock_active"] is True
    assert second["state_file_persisted"] is True
    assert second["state_file_path"] == first["state_file_path"]
    assert second["socket_fd_count"] == 1
    assert second["non_unix_socket_fd_count"] == 0

    for marker in (
        "landlock_active",
        "session_directories_distinct",
        "own_state_write_allowed",
        "sibling_read_denied",
        "sibling_write_denied",
        "sibling_list_denied",
        "sibling_delete_denied",
        "sibling_signal_zero_denied",
        "all_processes_signal_zero_denied",
        "sibling_tkill_denied",
        "sibling_tgkill_denied",
        "sibling_rt_sigqueueinfo_denied",
        "sibling_rt_tgsigqueueinfo_denied",
        "sibling_prlimit_denied",
        "sibling_sched_getaffinity_denied",
        "sibling_sched_setaffinity_denied",
        "sibling_sched_setparam_denied",
        "sibling_sched_setscheduler_denied",
        "sibling_sched_setattr_denied",
        "sibling_getpriority_denied",
        "sibling_setpriority_denied",
        "sibling_ioprio_get_denied",
        "sibling_ioprio_set_denied",
        "sysv_shmget_denied",
        "sysv_shmat_denied",
        "sysv_semget_denied",
        "sysv_msgget_denied",
        "process_madvise_denied",
        "process_mrelease_denied",
        "only_control_unix_fd",
        "global_tmp_list_denied",
        "global_tmp_create_denied",
        "global_var_tmp_list_denied",
        "global_var_tmp_create_denied",
        "global_shm_list_denied",
        "global_shm_create_denied",
    ):
        assert cross_session[marker] is True, f"{marker} did not pass"
    assert cross_session["socket_fd_count"] == 1
    assert cross_session["non_unix_socket_fd_count"] == 0

    serialized = json.dumps(
        {"first": first, "second": second, "cross_session": cross_session},
        sort_keys=True,
    )
    for name in SENSITIVE_ENV_NAMES:
        assert name not in serialized
        value = os.environ.get(name, "")
        if value:
            assert value not in serialized

    print(
        json.dumps(
            {
                "status": "ok",
                "visible_root": VISIBLE_ROOT,
                "sandbox_calls": 3,
                "state_persisted": True,
                "state_value": 42,
                "sandbox_parent_listener_accepted_later_call": (
                    parent_listener_accepted_later_call
                ),
                "seccomp_markers": {name: "1" for name in SECCOMP_MARKERS},
                "socket_fd_count": second["socket_fd_count"],
                "non_unix_socket_fd_count": second["non_unix_socket_fd_count"],
                "direct_bypasses_blocked": True,
                "cross_session_isolation": True,
                "write_blocked": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
