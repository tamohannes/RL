from __future__ import annotations

import json
import runpy
import socket
import subprocess
import sys
from pathlib import Path

import pytest

VALIDATOR = Path("examples/validate_sciprobe_sandbox_seccomp.py")
HOOK = Path("examples/sandbox_seccomp_hook/sitecustomize.py")


def _evaluate_filter(
    program: object,
    *,
    syscall_number: int,
    audit_arch: int,
    argument_zero: int,
    argument_one: int = 0,
    argument_four: int = 0,
) -> int:
    accumulator = 0
    program_counter = 0
    argument_zero_unsigned = argument_zero & 0xFFFFFFFFFFFFFFFF
    argument_one_unsigned = argument_one & 0xFFFFFFFFFFFFFFFF
    values = {
        0: syscall_number,
        4: audit_arch,
        16: argument_zero_unsigned & 0xFFFFFFFF,
        20: argument_zero_unsigned >> 32,
        24: argument_one_unsigned & 0xFFFFFFFF,
        28: argument_one_unsigned >> 32,
        48: argument_four & 0xFFFFFFFF,
        52: argument_four >> 32,
    }
    while True:
        instruction = program[program_counter]
        if instruction.code == 0x20:
            accumulator = values[instruction.k]
            program_counter += 1
        elif instruction.code == 0x15:
            offset = instruction.jt if accumulator == instruction.k else instruction.jf
            program_counter += int(offset) + 1
        elif instruction.code == 0x45:
            offset = instruction.jt if accumulator & instruction.k else instruction.jf
            program_counter += int(offset) + 1
        elif instruction.code == 0x06:
            return int(instruction.k)
        else:
            raise AssertionError(f"unexpected BPF opcode: {instruction.code}")


def test_filter_program_supports_x86_64_and_aarch64(
    monkeypatch: object,
) -> None:
    monkeypatch.delenv("SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK", raising=False)
    hook = runpy.run_path(str(HOOK))
    deny = 0x00050000 | 101
    process_deny = 0x00050000 | 1
    not_supported = 0x00050000 | 38
    allow = 0x7FFF0000
    kill = 0x80000000

    for architecture in ("x86_64", "aarch64"):
        config = hook["_ARCH_CONFIG"][architecture]
        program = hook["_assemble_filter"](config)
        assert max(instruction.jt for instruction in program) <= 255
        assert max(instruction.jf for instruction in program) <= 255
        inputs = {
            "audit_arch": int(config["audit_arch"]),
            "argument_zero": 0,
        }
        assert (
            _evaluate_filter(
                program,
                syscall_number=int(config["socket"]),
                argument_zero=socket.AF_INET,
                audit_arch=inputs["audit_arch"],
            )
            == deny
        )
        assert (
            _evaluate_filter(
                program,
                syscall_number=int(config["socket"]),
                argument_zero=socket.AF_UNIX,
                audit_arch=inputs["audit_arch"],
            )
            == deny
        )
        assert (
            _evaluate_filter(
                program,
                syscall_number=int(config["socketpair"]),
                argument_zero=socket.AF_UNIX,
                audit_arch=inputs["audit_arch"],
            )
            == allow
        )
        assert (
            _evaluate_filter(
                program,
                syscall_number=int(config["socketpair"]),
                argument_zero=socket.AF_INET6,
                audit_arch=inputs["audit_arch"],
            )
            == deny
        )
        for syscall_name in (
            "connect",
            "bind",
            "listen",
            "accept",
            "accept4",
            "sendmsg",
            "sendmmsg",
        ):
            assert (
                _evaluate_filter(
                    program,
                    syscall_number=int(config[syscall_name]),
                    **inputs,
                )
                == deny
            )
        assert (
            _evaluate_filter(
                program,
                syscall_number=int(config["sendto"]),
                argument_four=0x1000,
                **inputs,
            )
            == deny
        )
        assert (
            _evaluate_filter(
                program,
                syscall_number=int(config["sendto"]),
                argument_four=0x100000000,
                **inputs,
            )
            == deny
        )
        assert (
            _evaluate_filter(
                program,
                syscall_number=int(config["sendto"]),
                argument_four=0,
                **inputs,
            )
            == allow
        )
        assert (
            _evaluate_filter(
                program,
                syscall_number=int(config["io_uring_setup"]),
                **inputs,
            )
            == deny
        )
        assert (
            _evaluate_filter(
                program,
                syscall_number=int(config["clone"]),
                argument_zero=0,
                audit_arch=inputs["audit_arch"],
            )
            == process_deny
        )
        assert (
            _evaluate_filter(
                program,
                syscall_number=int(config["clone"]),
                argument_zero=0x00010000,
                audit_arch=inputs["audit_arch"],
            )
            == allow
        )
        assert (
            _evaluate_filter(
                program,
                syscall_number=int(config["clone3"]),
                **inputs,
            )
            == not_supported
        )
        for syscall_number in config["process_denied"]:
            assert (
                _evaluate_filter(
                    program,
                    syscall_number=int(syscall_number),
                    **inputs,
                )
                == process_deny
            )
        for syscall_number in config["pid_zero_allowed"]:
            assert (
                _evaluate_filter(
                    program,
                    syscall_number=int(syscall_number),
                    **inputs,
                )
                == allow
            )
            assert (
                _evaluate_filter(
                    program,
                    syscall_number=int(syscall_number),
                    argument_zero=123,
                    audit_arch=inputs["audit_arch"],
                )
                == process_deny
            )
        for syscall_number in config["priority_self_only"]:
            assert (
                _evaluate_filter(
                    program,
                    syscall_number=int(syscall_number),
                    argument_zero=0,
                    argument_one=0,
                    audit_arch=inputs["audit_arch"],
                )
                == allow
            )
            assert (
                _evaluate_filter(
                    program,
                    syscall_number=int(syscall_number),
                    argument_zero=0,
                    argument_one=123,
                    audit_arch=inputs["audit_arch"],
                )
                == process_deny
            )
            assert (
                _evaluate_filter(
                    program,
                    syscall_number=int(syscall_number),
                    argument_zero=1,
                    argument_one=0,
                    audit_arch=inputs["audit_arch"],
                )
                == process_deny
            )
        for syscall_number in config["ioprio_self_only"]:
            assert (
                _evaluate_filter(
                    program,
                    syscall_number=int(syscall_number),
                    argument_zero=1,
                    argument_one=0,
                    audit_arch=inputs["audit_arch"],
                )
                == allow
            )
            assert (
                _evaluate_filter(
                    program,
                    syscall_number=int(syscall_number),
                    argument_zero=1,
                    argument_one=123,
                    audit_arch=inputs["audit_arch"],
                )
                == process_deny
            )
            assert (
                _evaluate_filter(
                    program,
                    syscall_number=int(syscall_number),
                    argument_zero=2,
                    argument_one=0,
                    audit_arch=inputs["audit_arch"],
                )
                == process_deny
            )
        assert (
            _evaluate_filter(
                program,
                syscall_number=0,
                **inputs,
            )
            == allow
        )
        assert (
            _evaluate_filter(
                program,
                syscall_number=0,
                audit_arch=0,
                argument_zero=0,
            )
            == kill
        )


def test_filter_assembler_rejects_long_jump_before_ctypes_truncation(
    monkeypatch: object,
) -> None:
    monkeypatch.delenv("SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK", raising=False)
    hook = runpy.run_path(str(HOOK))
    config = dict(hook["_ARCH_CONFIG"]["x86_64"])
    config["process_denied"] = tuple(range(10_000, 10_300))

    with pytest.raises(RuntimeError) as caught:
        hook["_assemble_filter"](config)
    assert "invalid seccomp BPF jump" in str(caught.value)


def test_seccomp_gate_blocks_kernel_bypasses_only_in_shell_child() -> None:
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "ok"
    assert result["sandbox_parent_inet_listener_allowed"] is True
    assert result["sandbox_parent_listener_still_usable"] is True
    assert result["sandbox_parent_unix_listeners_still_usable"] is True
    assert result["primitive_control_messages_allowed"] is True
    assert result["reducer_pickle_rejected"] is True
    assert result["reducer_never_executed"] is True
    assert result["global_pickle_rejected"] is True
    assert result["persistent_pickle_rejected"] is True
    assert result["oversize_control_payload_rejected"] is True
    assert result["filter_absent_in_sandbox_parent"] is True
    assert result["filter_active"] is True
    assert result["process_filter_active"] is True
    assert result["restricted_unpickler_active"] is True
    assert result["landlock_active"] is True
    assert result["writer_landlock_active"] is True
    assert result["home_is_private"] is True
    assert result["tmpdir_is_private"] is True
    assert result["session_directories_distinct"] is True
    assert result["sibling_read_denied"] is True
    assert result["sibling_write_denied"] is True
    assert result["sibling_list_denied"] is True
    assert result["sibling_delete_denied"] is True
    assert result["sibling_signal_zero_denied"] is True
    assert result["all_processes_signal_zero_denied"] is True
    assert result["sibling_prlimit_query_denied"] is True
    assert result["sibling_prlimit_mutation_denied"] is True
    assert result["sibling_affinity_query_denied"] is True
    assert result["sibling_affinity_mutation_denied"] is True
    assert result["sibling_priority_query_denied"] is True
    assert result["sibling_priority_mutation_denied"] is True
    assert result["sibling_ioprio_query_denied"] is True
    assert result["sibling_ioprio_mutation_denied"] is True
    assert result["parent_secret_pipe_fd_closed"] is True
    assert result["parent_sysv_shm_attach_read_denied"] is True
    for root in ("tmp", "var_tmp", "shm"):
        assert result[f"global_{root}_read_denied"] is True
        assert result[f"global_{root}_write_denied"] is True
        assert result[f"global_{root}_create_denied"] is True
        assert result[f"global_{root}_list_denied"] is True
    assert result["inherited_global_file_fd_closed"] is True
    assert result["trusted_parent_proc_read_denied"] is True
    assert result["trusted_parent_proc_list_denied"] is True
    assert result["self_proc_read_allowed"] is True
    assert result["readonly_probe_read_allowed"] is True
    assert result["readonly_probe_write_denied"] is True
    assert result["readonly_probe_truncate_denied"] is True
    assert result["readonly_probe_delete_denied"] is True
    assert result["readonly_probe_hardlink_denied"] is True
    assert result["python_imports_allowed"] is True
    assert result["own_state_writes_persist"] is True
    assert result["writer_stop_received"] is True
    assert result["writer_secret_intact"] is True
    assert result["direct_syscall_denied"] is True
    assert result["fork_denied"] is True
    assert result["clone3_denied"] is True
    assert result["ptrace_denied"] is True
    assert result["process_vm_readv_denied"] is True
    assert result["process_vm_writev_denied"] is True
    assert result["pidfd_send_signal_denied"] is True
    assert result["pidfd_open_denied"] is True
    assert result["pidfd_getfd_denied"] is True
    assert result["kill_syscall_denied"] is True
    assert result["tkill_syscall_denied"] is True
    assert result["tgkill_syscall_denied"] is True
    assert result["rt_sigqueueinfo_syscall_denied"] is True
    assert result["rt_tgsigqueueinfo_syscall_denied"] is True
    assert result["process_madvise_syscall_denied"] is True
    assert result["process_mrelease_syscall_denied"] is True
    assert result["prlimit64_syscall_denied"] is True
    assert result["sysv_ipc_syscalls_denied"] is True
    assert result["scheduler_priority_syscalls_denied"] is True
    assert result["self_process_queries_allowed"] is True
    assert result["numpy_pandas_csv_allowed"] is True
    assert result["direct_execve_denied"] is True
    assert result["direct_execveat_denied"] is True
    assert result["thread_clone_allowed"] is True
    assert result["raw_socket_denied"] is True
    assert result["mro_base_denied"] is True
    assert result["env_clearing_exec_denied"] is True
    assert result["env_clearing_exec_unix_denied"] is True
    assert result["curl_denied"] is True
    assert result["curl_unix_socket_denied"] is True
    assert result["python_unix_socket_denied"] is True
    assert result["direct_unix_socket_denied"] is True
    assert result["unix_filesystem_connect_denied"] is True
    assert result["unix_abstract_connect_denied"] is True
    assert result["unix_filesystem_bind_denied"] is True
    assert result["unix_abstract_bind_denied"] is True
    assert result["unix_filesystem_sendto_denied"] is True
    assert result["unix_abstract_sendto_denied"] is True
    assert result["unix_named_sendmsg_denied"] is True
    assert result["inherited_unix_accept_denied"] is True
    assert result["inherited_unix_accept4_denied"] is True
    assert result["inherited_unix_listeners_closed"] is True
    assert result["control_socket_preserved"] is True
    assert result["untrusted_inherited_socket_count"] == 0
    assert result["inherited_inet_listener_closed"] is True
    assert result["non_unix_inherited_socket_count"] == 0
    assert result["unix_socketpair_allowed"] is True
    assert result["multiprocessing_process_denied"] is True
    assert result["file_io_allowed"] is True
    assert result["stateful_math_allowed"] is True
    assert result["ipython_stateful_cells_allowed"] is True


def test_landlock_requires_truncation_isolation(
    monkeypatch: object,
) -> None:
    monkeypatch.delenv("SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK", raising=False)
    hook = runpy.run_path(str(HOOK))

    with pytest.raises(RuntimeError) as caught:
        hook["_landlock_handled_access"](2)
    assert "safe truncation isolation" in str(caught.value)


def test_landlock_rejects_global_temp_allowlist(
    monkeypatch: object,
) -> None:
    monkeypatch.delenv("SCIPROBE_REQUIRE_SECCOMP_NETWORK_BLOCK", raising=False)
    monkeypatch.setenv("SCIPROBE_SANDBOX_READONLY_PATHS", "/tmp")
    hook = runpy.run_path(str(HOOK))

    with pytest.raises(RuntimeError) as caught:
        hook["_configured_readonly_paths"]()
    assert "exposes private session roots" in str(caught.value)


def test_seccomp_preflight_fails_closed_when_hook_mount_is_absent(
    tmp_path: Path,
) -> None:
    missing_hook_dir = tmp_path / "missing-hook"
    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--hook-dir",
            str(missing_hook_dir),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode != 0
    assert "required seccomp hook is absent" in completed.stderr
