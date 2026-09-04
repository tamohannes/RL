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

"""Exercise sandbox network blocking through the real DirectPythonTool path."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import struct
from pathlib import Path

import httpx
from nemo_skills.mcp.servers.python_tool import DirectPythonTool

STATEFUL_PROOF_CHUNK_ITEMS = 1


INITIALIZE_CODE = r"""import json
import math
import os
from pathlib import Path

sciprobe_network_state = {"calls": 1, "value": 40}
sciprobe_network_file = (
    Path(os.environ["SCIPROBE_SESSION_STATE_DIR"])
    / "sciprobe-network-state.txt"
)
sciprobe_network_file.write_text("state-1764", encoding="utf-8")
print(json.dumps({
    "file_write_ok": sciprobe_network_file.read_text(encoding="utf-8") == "state-1764",
    "landlock_active": os.environ.get("SCIPROBE_LANDLOCK_FILTER_ACTIVE") == "1",
    "math_ok": math.isqrt(1764) == 42,
    "shell_pid": os.getpid(),
    "state_file_path": str(sciprobe_network_file),
}, sort_keys=True))
"""


PROBE_CODE = r"""import _socket
import ctypes
import errno
import json
import multiprocessing
import os
import pickle
import platform
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


EXPECTED_ERRNO = errno.ENETUNREACH
MAX_CONTROL_MESSAGE_BYTES = 16 * 1024 * 1024
SOL_SOCKET = 1
SO_DOMAIN = 39


def network_denied(factory):
    try:
        candidate = factory()
    except OSError as error:
        return error.errno == EXPECTED_ERRNO
    try:
        return False
    finally:
        if hasattr(candidate, "close"):
            candidate.close()


def write_exploit_marker(path):
    Path(path).write_text("executed", encoding="utf-8")
    return "executed"


class ReducerPayload:
    def __init__(self, marker_path):
        self.marker_path = marker_path

    def __reduce__(self):
        return write_exploit_marker, (self.marker_path,)


def control_payload_rejected(payload, raw=False):
    receiver, sender = multiprocessing.Pipe(duplex=False)
    try:
        sender.send_bytes(payload) if raw else sender.send(payload)
        try:
            receiver.recv()
        except pickle.UnpicklingError:
            return receiver.closed
        return False
    finally:
        receiver.close()
        sender.close()


def oversize_control_payload_rejected():
    receiver, sender = multiprocessing.Pipe(duplex=True)
    try:
        header = struct.pack("!i", MAX_CONTROL_MESSAGE_BYTES + 1)
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


def socket_domain(libc, descriptor):
    domain = ctypes.c_int()
    size = ctypes.c_uint(ctypes.sizeof(domain))
    ctypes.set_errno(0)
    result = libc.getsockopt(
        descriptor, SOL_SOCKET, SO_DOMAIN, ctypes.byref(domain), ctypes.byref(size)
    )
    if result == 0:
        return int(domain.value)
    if ctypes.get_errno() in {errno.EBADF, errno.ENOTSOCK}:
        return None
    raise OSError(ctypes.get_errno(), "getsockopt(SO_DOMAIN) failed")


def socket_descriptors(libc):
    found = []
    for entry in os.listdir("/proc/self/fd"):
        try:
            descriptor = int(entry)
        except ValueError:
            continue
        if descriptor <= 2:
            continue
        domain = socket_domain(libc, descriptor)
        if domain is not None:
            found.append({"fd": descriptor, "domain": domain})
    return found


def open_descriptors(libc):
    found = []
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
        domain = socket_domain(libc, descriptor)
        found.append({
            "fd": descriptor,
            "kind": "socket" if domain is not None else stat.S_IFMT(metadata.st_mode),
            "domain": domain,
        })
    return found


class SockaddrUn(ctypes.Structure):
    _fields_ = [("sun_family", ctypes.c_ushort), ("sun_path", ctypes.c_char * 108)]


def unix_sockaddr(address):
    encoded = os.fsencode(address) if isinstance(address, str) else address
    sockaddr = SockaddrUn(sun_family=socket.AF_UNIX)
    sockaddr.sun_path = encoded
    terminator = 0 if encoded.startswith(b"\0") else 1
    return sockaddr, 2 + len(encoded) + terminator


machine = platform.machine().lower()
syscalls_by_arch = {
    "x86_64": {
        "socket": 41,
        "connect": 42,
        "accept": 43,
        "sendto": 44,
        "sendmsg": 46,
        "bind": 49,
        "listen": 50,
        "socketpair": 53,
        "accept4": 288,
        "io_uring_setup": 425,
        "execve": 59,
        "ptrace": 101,
        "process_vm_readv": 310,
        "process_vm_writev": 311,
        "execveat": 322,
        "pidfd_send_signal": 424,
        "pidfd_open": 434,
        "clone3": 435,
        "pidfd_getfd": 438,
    },
    "amd64": {
        "socket": 41,
        "connect": 42,
        "accept": 43,
        "sendto": 44,
        "sendmsg": 46,
        "bind": 49,
        "listen": 50,
        "socketpair": 53,
        "accept4": 288,
        "io_uring_setup": 425,
        "execve": 59,
        "ptrace": 101,
        "process_vm_readv": 310,
        "process_vm_writev": 311,
        "execveat": 322,
        "pidfd_send_signal": 424,
        "pidfd_open": 434,
        "clone3": 435,
        "pidfd_getfd": 438,
    },
    "aarch64": {
        "socket": 198,
        "socketpair": 199,
        "bind": 200,
        "listen": 201,
        "accept": 202,
        "connect": 203,
        "sendto": 206,
        "sendmsg": 211,
        "accept4": 242,
        "io_uring_setup": 425,
        "execve": 221,
        "ptrace": 117,
        "process_vm_readv": 270,
        "process_vm_writev": 271,
        "execveat": 281,
        "pidfd_send_signal": 424,
        "pidfd_open": 434,
        "clone3": 435,
        "pidfd_getfd": 438,
    },
    "arm64": {
        "socket": 198,
        "socketpair": 199,
        "bind": 200,
        "listen": 201,
        "accept": 202,
        "connect": 203,
        "sendto": 206,
        "sendmsg": 211,
        "accept4": 242,
        "io_uring_setup": 425,
        "execve": 221,
        "ptrace": 117,
        "process_vm_readv": 270,
        "process_vm_writev": 271,
        "execveat": 281,
        "pidfd_send_signal": 424,
        "pidfd_open": 434,
        "clone3": 435,
        "pidfd_getfd": 438,
    },
}
syscalls = syscalls_by_arch.get(machine)
direct_syscall_number_known = syscalls is not None
if syscalls is None:
    raise RuntimeError("unsupported architecture: " + machine)

libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
libc.getsockopt.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint),
]
libc.getsockopt.restype = ctypes.c_int


def direct_syscall_denied(number, *arguments):
    ctypes.set_errno(0)
    result = libc.syscall(ctypes.c_long(number), *arguments)
    return result == -1 and ctypes.get_errno() == EXPECTED_ERRNO


python_ipv4_denied = network_denied(
    lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM)
)
python_ipv6_denied = network_denied(
    lambda: socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
)
native_socket_denied = network_denied(
    lambda: _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
)


class DerivedSocket(socket.socket):
    pass


native_socket_base = DerivedSocket.__mro__[1].__mro__[1]
mro_base_denied = network_denied(
    lambda: native_socket_base(socket.AF_INET, socket.SOCK_STREAM)
)
raw_socket_denied = network_denied(
    lambda: socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
)
python_unix_socket_denied = network_denied(
    lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
)
direct_socket_syscall_denied = direct_syscall_denied(
    syscalls["socket"],
    ctypes.c_long(socket.AF_INET),
    ctypes.c_long(socket.SOCK_STREAM),
    ctypes.c_long(0),
)
direct_unix_socket_denied = direct_syscall_denied(
    syscalls["socket"],
    ctypes.c_long(socket.AF_UNIX),
    ctypes.c_long(socket.SOCK_STREAM),
    ctypes.c_long(0),
)
socket_pair = (ctypes.c_int * 2)()
non_unix_socketpair_denied = direct_syscall_denied(
    syscalls["socketpair"],
    ctypes.c_long(socket.AF_INET),
    ctypes.c_long(socket.SOCK_STREAM),
    ctypes.c_long(0),
    ctypes.byref(socket_pair),
)
io_uring_parameters = ctypes.create_string_buffer(256)
io_uring_setup_denied = direct_syscall_denied(
    syscalls["io_uring_setup"],
    ctypes.c_long(1),
    ctypes.byref(io_uring_parameters),
)

initial_socket_fds = socket_descriptors(libc)
control_fd = initial_socket_fds[0]["fd"] if len(initial_socket_fds) == 1 else -1
unix_stream_path = "/tmp/sciprobe-seccomp-nonexistent.sock"
unix_datagram_path = "/tmp/sciprobe-seccomp-nonexistent-dgram.sock"
abstract_stream_address = b"\0sciprobe-seccomp-stream-nonexistent"
abstract_datagram_address = b"\0sciprobe-seccomp-dgram-nonexistent"


def control_socket_operation_denied(operation):
    if control_fd < 0:
        return False
    duplicated = os.dup(control_fd)
    unix_socket = socket.socket(fileno=duplicated)
    try:
        return network_denied(lambda: operation(unix_socket))
    finally:
        unix_socket.close()


unix_filesystem_connect_denied = control_socket_operation_denied(
    lambda unix_socket: unix_socket.connect(unix_stream_path)
)
unix_abstract_connect_denied = control_socket_operation_denied(
    lambda unix_socket: unix_socket.connect(abstract_stream_address)
)
unix_filesystem_bind_denied = control_socket_operation_denied(
    lambda unix_socket: unix_socket.bind(unix_stream_path + ".bind")
)
unix_abstract_bind_denied = control_socket_operation_denied(
    lambda unix_socket: unix_socket.bind(abstract_stream_address + b"-bind")
)
unix_listen_denied = control_socket_operation_denied(
    lambda unix_socket: unix_socket.listen(1)
)

datagram_left, datagram_right = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
try:
    unix_filesystem_sendto_denied = network_denied(
        lambda: datagram_left.sendto(b"blocked", unix_datagram_path)
    )
    unix_abstract_sendto_denied = network_denied(
        lambda: datagram_left.sendto(b"blocked", abstract_datagram_address)
    )
    unix_named_sendmsg_denied = network_denied(
        lambda: datagram_left.sendmsg([b"blocked"], [], 0, unix_datagram_path)
    )
finally:
    datagram_left.close()
    datagram_right.close()

direct_control_fd = os.dup(control_fd) if control_fd >= 0 else -1
try:
    sockaddr, sockaddr_size = unix_sockaddr(unix_stream_path)
    direct_unix_connect_denied = (
        direct_syscall_denied(
            syscalls["connect"],
            ctypes.c_long(direct_control_fd),
            ctypes.byref(sockaddr),
            ctypes.c_long(sockaddr_size),
        )
        if direct_control_fd >= 0
        else False
    )
    inherited_unix_accept_denied = (
        direct_syscall_denied(
            syscalls["accept"],
            ctypes.c_long(direct_control_fd),
            ctypes.c_void_p(),
            ctypes.c_void_p(),
        )
        if direct_control_fd >= 0
        else False
    )
    inherited_unix_accept4_denied = (
        direct_syscall_denied(
            syscalls["accept4"],
            ctypes.c_long(direct_control_fd),
            ctypes.c_void_p(),
            ctypes.c_void_p(),
            ctypes.c_long(0),
        )
        if direct_control_fd >= 0
        else False
    )
finally:
    if direct_control_fd >= 0:
        os.close(direct_control_fd)

PROCESS_ERRNO = errno.EPERM


def direct_process_syscall_denied(number, *arguments):
    ctypes.set_errno(0)
    result = libc.syscall(ctypes.c_long(number), *arguments)
    return result == -1 and ctypes.get_errno() == PROCESS_ERRNO


try:
    subprocess.run(
        [sys.executable, "-I", "-c", "print(42)"],
        env={},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
except OSError as error:
    subprocess_creation_denied = error.errno == PROCESS_ERRNO
else:
    subprocess_creation_denied = False

direct_execve_denied = direct_process_syscall_denied(
    syscalls["execve"],
    ctypes.c_char_p(b"/sciprobe-does-not-exist"),
    ctypes.c_void_p(),
    ctypes.c_void_p(),
)
direct_execveat_denied = direct_process_syscall_denied(
    syscalls["execveat"],
    ctypes.c_long(-1),
    ctypes.c_char_p(b"sciprobe-does-not-exist"),
    ctypes.c_void_p(),
    ctypes.c_void_p(),
    ctypes.c_long(0),
)
ctypes.set_errno(0)
clone3_result = libc.syscall(
    ctypes.c_long(syscalls["clone3"]), ctypes.c_void_p(), ctypes.c_long(0)
)
clone3_denied = clone3_result == -1 and ctypes.get_errno() == errno.ENOSYS
ptrace_denied = direct_process_syscall_denied(
    syscalls["ptrace"],
    ctypes.c_long(-1),
    ctypes.c_long(0),
    ctypes.c_void_p(),
    ctypes.c_void_p(),
)
process_vm_readv_denied = direct_process_syscall_denied(
    syscalls["process_vm_readv"],
    ctypes.c_long(os.getpid()),
    ctypes.c_void_p(),
    ctypes.c_long(0),
    ctypes.c_void_p(),
    ctypes.c_long(0),
    ctypes.c_long(0),
)
process_vm_writev_denied = direct_process_syscall_denied(
    syscalls["process_vm_writev"],
    ctypes.c_long(os.getpid()),
    ctypes.c_void_p(),
    ctypes.c_long(0),
    ctypes.c_void_p(),
    ctypes.c_long(0),
    ctypes.c_long(0),
)
pidfd_open_denied = direct_process_syscall_denied(
    syscalls["pidfd_open"], ctypes.c_long(os.getpid()), ctypes.c_long(0)
)
pidfd_getfd_denied = direct_process_syscall_denied(
    syscalls["pidfd_getfd"], ctypes.c_long(-1), ctypes.c_long(-1), ctypes.c_long(0)
)
pidfd_send_signal_denied = direct_process_syscall_denied(
    syscalls["pidfd_send_signal"],
    ctypes.c_long(-1),
    ctypes.c_long(0),
    ctypes.c_void_p(),
    ctypes.c_long(0),
)

thread_result = []
thread = threading.Thread(target=lambda: thread_result.append(42))
thread.start()
thread.join(timeout=5)
thread_clone_allowed = not thread.is_alive() and thread_result == [42]

unix_left, unix_right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    unix_left.sendall(b"unix-ok")
    unix_socketpair_allowed = unix_right.recv(7) == b"unix-ok"
finally:
    unix_left.close()
    unix_right.close()

with tempfile.TemporaryDirectory() as directory:
    exploit_marker = Path(directory) / "pickle-exploit-executed"
    reducer_pickle_rejected = control_payload_rejected(
        ReducerPayload(str(exploit_marker))
    )
    reducer_never_executed = not exploit_marker.exists()
    global_pickle_rejected = control_payload_rejected(os.system)
    persistent_pickle_rejected = control_payload_rejected(
        b"Pforbidden-reference\n.", raw=True
    )
    oversize_payload_rejected = oversize_control_payload_rejected()

socket_fds = socket_descriptors(libc)
open_fds = open_descriptors(libc)
non_unix_socket_fds = [
    item for item in socket_fds if item["domain"] != socket.AF_UNIX
]
seccomp_mode = None
for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
    name, separator, value = line.partition(":")
    if separator and name == "Seccomp":
        seccomp_mode = int(value.strip())
        break

trusted_driver_pid = __TRUSTED_DRIVER_PID__
trusted_driver_proc_hidden = not Path("/proc", str(trusted_driver_pid)).exists()

proof = {
    "hook_loaded": os.environ.get("SCIPROBE_SECCOMP_HOOK_LOADED") == "1",
    "filter_active": os.environ.get("SCIPROBE_SECCOMP_FILTER_ACTIVE") == "1",
    "inherited_fds_clean": os.environ.get("SCIPROBE_SECCOMP_INHERITED_FDS_CLEAN") == "1",
    "restricted_unpickler_active": os.environ.get("SCIPROBE_RESTRICTED_MP_UNPICKLER_ACTIVE") == "1",
    "seccomp_mode_filter": seccomp_mode == 2,
    "python_ipv4_denied": python_ipv4_denied,
    "python_ipv6_denied": python_ipv6_denied,
    "native_socket_denied": native_socket_denied,
    "mro_base_denied": mro_base_denied,
    "raw_socket_denied": raw_socket_denied,
    "python_unix_socket_denied": python_unix_socket_denied,
    "direct_syscall_number_known": direct_syscall_number_known,
    "direct_socket_syscall_denied": direct_socket_syscall_denied,
    "direct_unix_socket_denied": direct_unix_socket_denied,
    "non_unix_socketpair_denied": non_unix_socketpair_denied,
    "io_uring_setup_denied": io_uring_setup_denied,
    "unix_filesystem_connect_denied": unix_filesystem_connect_denied,
    "unix_abstract_connect_denied": unix_abstract_connect_denied,
    "unix_filesystem_bind_denied": unix_filesystem_bind_denied,
    "unix_abstract_bind_denied": unix_abstract_bind_denied,
    "unix_listen_denied": unix_listen_denied,
    "unix_filesystem_sendto_denied": unix_filesystem_sendto_denied,
    "unix_abstract_sendto_denied": unix_abstract_sendto_denied,
    "unix_named_sendmsg_denied": unix_named_sendmsg_denied,
    "direct_unix_connect_denied": direct_unix_connect_denied,
    "inherited_unix_accept_denied": inherited_unix_accept_denied,
    "inherited_unix_accept4_denied": inherited_unix_accept4_denied,
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
    "reducer_pickle_rejected": reducer_pickle_rejected,
    "reducer_never_executed": reducer_never_executed,
    "global_pickle_rejected": global_pickle_rejected,
    "persistent_pickle_rejected": persistent_pickle_rejected,
    "oversize_control_payload_rejected": oversize_payload_rejected,
    "file_io_persisted": sciprobe_network_file.read_text(encoding="utf-8") == "state-1764",
    "landlock_active": os.environ.get("SCIPROBE_LANDLOCK_FILTER_ACTIVE") == "1",
    "state_file_path": str(sciprobe_network_file),
    "state_value": sciprobe_network_state["value"],
    "trusted_driver_proc_hidden": trusted_driver_proc_hidden,
    "unshare_available": shutil.which("unshare") is not None,
    "open_fd_count": len(open_fds),
    "open_fds": open_fds,
    "socket_fd_count": len(socket_fds),
    "non_unix_socket_fd_count": len(non_unix_socket_fds),
}
sciprobe_security_proof = proof
print(json.dumps({
    "proof_key_count": len(sciprobe_security_proof),
    "proof_ready": True,
}, sort_keys=True))
"""


FOLLOWUP_CODE = r"""import ctypes
import errno
import json
import os
import socket
import stat

sciprobe_network_state["calls"] += 1
sciprobe_network_state["value"] += 2

libc = ctypes.CDLL(None, use_errno=True)
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
        1,
        39,
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
    "state_persisted": sciprobe_network_state == {"calls": 2, "value": 42},
    "state_calls": sciprobe_network_state["calls"],
    "state_value": sciprobe_network_state["value"],
    "file_io_persisted": sciprobe_network_file.read_text(
        encoding="utf-8"
    ) == "state-1764",
    "hook_loaded": os.environ.get("SCIPROBE_SECCOMP_HOOK_LOADED") == "1",
    "filter_active": os.environ.get("SCIPROBE_SECCOMP_FILTER_ACTIVE") == "1",
    "inherited_fds_clean": os.environ.get(
        "SCIPROBE_SECCOMP_INHERITED_FDS_CLEAN"
    ) == "1",
    "restricted_unpickler_active": os.environ.get(
        "SCIPROBE_RESTRICTED_MP_UNPICKLER_ACTIVE"
    ) == "1",
    "landlock_active": os.environ.get(
        "SCIPROBE_LANDLOCK_FILTER_ACTIVE"
    ) == "1",
    "state_file_path": str(sciprobe_network_file),
    "open_fd_count": len(open_fds),
    "open_fds": open_fds,
    "socket_fd_count": len(socket_fds),
    "non_unix_socket_fd_count": sum(
        item["domain"] != socket.AF_UNIX for item in socket_fds
    ),
}, sort_keys=True))
"""


CROSS_SESSION_CODE = r"""import ctypes
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
non_unix_socket_fds = [
    item for item in socket_fds if item["domain"] != socket.AF_UNIX
]

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
    "non_unix_socket_fd_count": len(non_unix_socket_fds),
    "only_control_unix_fd": (
        len(socket_fds) == 1
        and socket_fds[0]["domain"] == socket.AF_UNIX
    ),
}
for label, root in global_roots.items():
    proof[f"global_{label}_list_denied"] = filesystem_denied(lambda path=root: list(path.iterdir()))
    proof[f"global_{label}_create_denied"] = filesystem_denied(
        lambda path=root: (path / __GLOBAL_NAME__).write_text(
            "forbidden", encoding="utf-8"
        )
    )
sciprobe_cross_session_proof = proof
print(json.dumps({
    "proof_key_count": len(sciprobe_cross_session_proof),
    "proof_ready": True,
}, sort_keys=True))
"""


def _decode_proc_address(address_hex: str, family: socket.AddressFamily) -> str:
    if family == socket.AF_INET:
        packed = struct.pack("<I", int(address_hex, 16))
    elif family == socket.AF_INET6:
        if len(address_hex) != 32:
            raise ValueError(f"invalid IPv6 address in /proc/net/tcp6: {address_hex!r}")
        packed = b"".join(
            struct.pack("<I", int(address_hex[offset : offset + 8], 16))
            for offset in range(0, 32, 8)
        )
    else:
        raise ValueError(f"unsupported address family: {family}")
    return socket.inet_ntop(family, packed)


def _proc_tcp_listeners(
    path: Path, family: socket.AddressFamily
) -> list[dict[str, object]]:
    listeners: list[dict[str, object]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 4:
            raise RuntimeError(f"malformed socket row in {path}: {line!r}")
        if fields[3] != "0A":
            continue  # TCP_LISTEN
        local_address = fields[1]
        address_hex, separator, port_hex = local_address.rpartition(":")
        if not separator:
            raise RuntimeError(f"malformed local socket address in {path}: {line!r}")
        port = int(port_hex, 16)
        if port not in {6000, 6001}:
            continue
        normalized_host = _decode_proc_address(address_hex, family)
        listeners.append(
            {
                "address": normalized_host,
                "port": port,
                "raw": local_address,
                "source": str(path),
            }
        )
    return listeners


def _loopback_listeners() -> tuple[bool, list[dict[str, object]]]:
    listeners = _proc_tcp_listeners(Path("/proc/net/tcp"), socket.AF_INET)
    listeners.extend(_proc_tcp_listeners(Path("/proc/net/tcp6"), socket.AF_INET6))
    expected = {(str(item["address"]), int(item["port"])) for item in listeners}
    return (
        len(listeners) == 2 and expected == {("127.0.0.1", 6000), ("127.0.0.1", 6001)}
    ), listeners


async def _remote_request_denied(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs: object,
) -> bool:
    try:
        response = await client.request(method, url, **kwargs)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return True
    return response.status_code in {401, 403}


async def _verify_remote_ingress_denied(host: str, port: int) -> dict[str, bool]:
    if host in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("remote sandbox host resolved to loopback")
    base_url = f"http://{host}:{port}"
    timeout = httpx.Timeout(2.0)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        execute_denied, list_denied, delete_denied = await asyncio.gather(
            _remote_request_denied(
                client,
                "POST",
                f"{base_url}/execute",
                json={
                    "generated_code": "print(42)",
                    "timeout": 1,
                    "language": "python",
                },
            ),
            _remote_request_denied(client, "GET", f"{base_url}/sessions"),
            _remote_request_denied(
                client,
                "DELETE",
                f"{base_url}/sessions/sciprobe-unauthorized",
            ),
        )
    return {
        "remote_execute_denied": execute_denied,
        "remote_list_sessions_denied": list_denied,
        "remote_delete_session_denied": delete_denied,
    }


async def _collect_stateful_mapping(
    tool: DirectPythonTool,
    *,
    request_id: str,
    variable_name: str,
    key_count: object,
) -> dict[str, object]:
    allowed_variables = {
        "sciprobe_security_proof",
        "sciprobe_cross_session_proof",
    }
    if variable_name not in allowed_variables:
        raise ValueError(f"unsupported stateful proof variable: {variable_name!r}")
    if not isinstance(key_count, int) or isinstance(key_count, bool):
        raise TypeError("stateful proof key count must be an integer")
    if key_count < 1 or key_count > 256:
        raise ValueError(f"invalid stateful proof key count: {key_count}")

    proof: dict[str, object] = {}
    for start in range(0, key_count, STATEFUL_PROOF_CHUNK_ITEMS):
        stop = min(start + STATEFUL_PROOF_CHUNK_ITEMS, key_count)
        chunk_raw = await tool.execute(
            "stateful_python_code_exec",
            {
                "code": (
                    "import json\n"
                    f"_items = sorted({variable_name}.items())\n"
                    f"print(json.dumps(dict(_items[{start}:{stop}]), sort_keys=True))"
                )
            },
            extra_args={"request_id": request_id},
        )
        chunk = json.loads(chunk_raw)
        if not isinstance(chunk, dict):
            raise RuntimeError("stateful proof chunk is not an object")
        duplicate_keys = set(proof).intersection(chunk)
        if duplicate_keys:
            raise RuntimeError(
                "stateful proof returned duplicate keys: "
                + ", ".join(sorted(duplicate_keys))
            )
        proof.update(chunk)

    if len(proof) != key_count:
        raise RuntimeError(
            f"stateful proof returned {len(proof)} keys; expected {key_count}"
        )
    return proof


async def run(args: argparse.Namespace) -> dict[str, object]:
    loopback_listener_only, listeners = _loopback_listeners()
    remote_ingress = await _verify_remote_ingress_denied(
        args.remote_sandbox_host, args.sandbox_port
    )
    tool = DirectPythonTool(exec_timeout_s=30)
    tool.configure(
        context={
            "sandbox": {
                "sandbox_type": "local",
                "host": args.sandbox_host,
                "port": str(args.sandbox_port),
                "disable_session_restore": True,
            }
        }
    )
    request_id = "sciprobe-network-blocking-preflight"
    try:
        initialize_raw = await tool.execute(
            "stateful_python_code_exec",
            {"code": INITIALIZE_CODE},
            extra_args={"request_id": request_id},
        )
        proof_manifest_raw = await tool.execute(
            "stateful_python_code_exec",
            {"code": PROBE_CODE.replace("__TRUSTED_DRIVER_PID__", str(os.getpid()))},
            extra_args={"request_id": request_id},
        )
        proof_manifest = json.loads(proof_manifest_raw)
        if not isinstance(proof_manifest, dict):
            raise RuntimeError("sandbox first proof manifest is not an object")
        if proof_manifest.get("proof_ready") is not True:
            raise RuntimeError("sandbox first proof did not become ready")
        proof = await _collect_stateful_mapping(
            tool,
            request_id=request_id,
            variable_name="sciprobe_security_proof",
            key_count=proof_manifest.get("proof_key_count"),
        )
        followup_raw = await tool.execute(
            "stateful_python_code_exec",
            {"code": FOLLOWUP_CODE},
            extra_args={"request_id": request_id},
        )
        initialization_preview = json.loads(initialize_raw)
        if not isinstance(initialization_preview, dict):
            raise RuntimeError("sandbox initialization did not return an object")
        writer_file = initialization_preview.get("state_file_path")
        if not isinstance(writer_file, str) or not writer_file:
            raise RuntimeError("sandbox initialization omitted its private state path")
        writer_pid = initialization_preview.get("shell_pid")
        if not isinstance(writer_pid, int) or writer_pid <= 1:
            raise RuntimeError("sandbox initialization omitted its shell process id")
        cross_session_code = (
            CROSS_SESSION_CODE.replace("__WRITER_FILE__", json.dumps(writer_file))
            .replace("__WRITER_PID__", str(writer_pid))
            .replace(
                "__GLOBAL_NAME__",
                json.dumps(f"sciprobe-denied-{os.urandom(16).hex()}"),
            )
        )
        cross_session_manifest_raw = await tool.execute(
            "stateful_python_code_exec",
            {"code": cross_session_code},
            extra_args={"request_id": request_id + "-reader"},
        )
        cross_session_manifest = json.loads(cross_session_manifest_raw)
        if not isinstance(cross_session_manifest, dict):
            raise RuntimeError("sandbox cross-session proof manifest is not an object")
        if cross_session_manifest.get("proof_ready") is not True:
            raise RuntimeError("sandbox cross-session proof did not become ready")
        cross_session = await _collect_stateful_mapping(
            tool,
            request_id=request_id + "-reader",
            variable_name="sciprobe_cross_session_proof",
            key_count=cross_session_manifest.get("proof_key_count"),
        )
    finally:
        await tool.shutdown()

    initialization = json.loads(initialize_raw)
    followup = json.loads(followup_raw)
    if not all(
        isinstance(value, dict)
        for value in (initialization, proof, followup, cross_session)
    ):
        raise RuntimeError("sandbox did not return proof objects")

    required_true = (
        "hook_loaded",
        "filter_active",
        "inherited_fds_clean",
        "restricted_unpickler_active",
        "landlock_active",
        "seccomp_mode_filter",
        "python_ipv4_denied",
        "python_ipv6_denied",
        "native_socket_denied",
        "mro_base_denied",
        "raw_socket_denied",
        "python_unix_socket_denied",
        "direct_syscall_number_known",
        "direct_socket_syscall_denied",
        "direct_unix_socket_denied",
        "non_unix_socketpair_denied",
        "io_uring_setup_denied",
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
        "reducer_pickle_rejected",
        "reducer_never_executed",
        "global_pickle_rejected",
        "persistent_pickle_rejected",
        "oversize_control_payload_rejected",
        "file_io_persisted",
        "trusted_driver_proc_hidden",
        "unshare_available",
    )
    required_followup_true = (
        "state_persisted",
        "file_io_persisted",
        "hook_loaded",
        "filter_active",
        "inherited_fds_clean",
        "restricted_unpickler_active",
        "landlock_active",
    )
    required_cross_session_true = (
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
    )
    ingress = {
        "loopback_listener_only": loopback_listener_only,
        "listeners": listeners,
        **remote_ingress,
    }
    ingress_gate_passed = loopback_listener_only and all(remote_ingress.values())
    security_gate_passed = (
        ingress_gate_passed
        and initialization.get("file_write_ok") is True
        and initialization.get("landlock_active") is True
        and initialization.get("math_ok") is True
        and isinstance(initialization.get("state_file_path"), str)
        and all(proof.get(name) is True for name in required_true)
        and proof.get("state_file_path") == initialization.get("state_file_path")
        and proof.get("state_value") == 40
        and proof.get("socket_fd_count") == 1
        and proof.get("non_unix_socket_fd_count") == 0
        and all(followup.get(name) is True for name in required_followup_true)
        and followup.get("state_calls") == 2
        and followup.get("state_value") == 42
        and followup.get("socket_fd_count") == 1
        and followup.get("non_unix_socket_fd_count") == 0
        and followup.get("state_file_path") == initialization.get("state_file_path")
        and all(cross_session.get(name) is True for name in required_cross_session_true)
        and cross_session.get("socket_fd_count") == 1
        and cross_session.get("non_unix_socket_fd_count") == 0
    )
    if not security_gate_passed:
        raise RuntimeError(
            "live DirectPythonTool security proof failed: "
            f"initialization={initialization}; first={proof}; second={followup}; "
            f"cross_session={cross_session}"
        )
    result: dict[str, object] = {
        "initialization": initialization,
        "first": proof,
        "second": followup,
        "cross_session": cross_session,
        "ingress": ingress,
        "sandbox_parent_listener_accepted_later_call": True,
        "security_gate_passed": True,
        "status": "ok",
    }
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_output, args.output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox-host", default="127.0.0.1")
    parser.add_argument("--sandbox-port", type=int, default=6000)
    parser.add_argument("--remote-sandbox-host", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), sort_keys=True))


if __name__ == "__main__":
    main()
