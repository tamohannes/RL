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

"""Probe sibling /proc access through the real DirectPythonTool sandbox path."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from nemo_skills.mcp.servers.python_tool import DirectPythonTool

EXPECTED_DATA_SHA256 = (
    "16713f67f959a4c276baea508c1fb64fa54bf622f4e14b0b4def77d6c152a590"
)


def _sandbox_code(target_pid: int, private_filename: str) -> str:
    return f"""import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

target_pid = {target_pid!r}
private_filename = {private_filename!r}

target_proc = Path("/proc") / str(target_pid)
try:
    target_environ = (target_proc / "environ").read_bytes()
    target_environ_readable = True
    target_env_sentinel_found = b"SCIPROBE_PROC_ENV_SENTINEL=" in target_environ
except OSError:
    target_environ_readable = False
    target_env_sentinel_found = False

target_private_path = target_proc / "root/workspace/private-sentinel" / private_filename
try:
    target_private_contents = target_private_path.read_bytes()
    target_private_file_readable = len(target_private_contents) == 32
except OSError:
    target_private_file_readable = False

any_sibling_env_sentinel_found = False
any_sibling_private_file_readable = False
for proc_entry in Path("/proc").iterdir():
    if not proc_entry.name.isdigit() or proc_entry.name == str(os.getpid()):
        continue
    try:
        environ_contents = (proc_entry / "environ").read_bytes()
    except OSError:
        pass
    else:
        if b"SCIPROBE_PROC_ENV_SENTINEL=" in environ_contents:
            any_sibling_env_sentinel_found = True
    try:
        private_contents = (
            proc_entry / "root/workspace/private-sentinel" / private_filename
        ).read_bytes()
    except OSError:
        pass
    else:
        if len(private_contents) == 32:
            any_sibling_private_file_readable = True

try:
    pid1_private_contents = (
        Path("/proc/1/root/workspace/private-sentinel") / private_filename
    ).read_bytes()
    private_file_readable_via_pid1_root = len(pid1_private_contents) == 32
except OSError:
    private_file_readable_via_pid1_root = False

mount_dir = Path(tempfile.mkdtemp(prefix="sciprobe-proc-mount-"))
mount_result = subprocess.run(
    ["mount", "-t", "proc", "proc", str(mount_dir)],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=False,
)
proc_remount_succeeded = mount_result.returncode == 0
target_visible_in_remounted_proc = (
    (mount_dir / str(target_pid)).exists() if proc_remount_succeeded else False
)
if proc_remount_succeeded:
    subprocess.run(
        ["umount", str(mount_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

data_root = Path("/workspace/sciprobe-probe/data")
digest = hashlib.sha256()
data_files = 0
data_bytes = 0
data_symlink_found = False
for path in sorted(data_root.rglob("*")):
    if path.is_symlink():
        data_symlink_found = True
        continue
    if path.is_dir():
        continue
    if not path.is_file():
        continue
    relative = path.relative_to(data_root).as_posix()
    contents = path.read_bytes()
    digest.update(relative.encode("utf-8"))
    digest.update(b"\\0")
    digest.update(str(len(contents)).encode("ascii"))
    digest.update(b"\\0")
    digest.update(contents)
    digest.update(b"\\0")
    data_files += 1
    data_bytes += len(contents)

write_blocked = False
try:
    with (data_root / "crispresso_output/README_pipeline.txt").open("ab") as handle:
        handle.write(b"proc-isolation-write-test")
except OSError:
    write_blocked = True

proof = {{
    "target_proc_visible": target_proc.exists(),
    "target_environ_readable": target_environ_readable,
    "target_env_sentinel_found": target_env_sentinel_found,
    "target_private_file_readable": target_private_file_readable,
    "any_sibling_env_sentinel_found": any_sibling_env_sentinel_found,
    "any_sibling_private_file_readable": any_sibling_private_file_readable,
    "private_file_readable_via_pid1_root": private_file_readable_via_pid1_root,
    "proc_remount_succeeded": proc_remount_succeeded,
    "target_visible_in_remounted_proc": target_visible_in_remounted_proc,
    "unshare_available": shutil.which("unshare") is not None,
    "uid_is_zero": os.geteuid() == 0,
    "data_mount_valid": (
        digest.hexdigest() == {EXPECTED_DATA_SHA256!r}
        and data_files == 25
        and data_bytes == 8137
        and not data_symlink_found
    ),
    "data_mount_write_blocked": write_blocked,
}}
assert all(isinstance(value, bool) for value in proof.values())
print(json.dumps(proof, sort_keys=True))
"""


async def _run(args: argparse.Namespace) -> dict[str, object]:
    target_pid = int(args.primary_pid_file.read_text(encoding="utf-8").strip())
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
    try:
        raw = await tool.execute(
            "stateful_python_code_exec",
            {"code": _sandbox_code(target_pid, args.private_filename)},
            extra_args={"request_id": f"proc-isolation-{args.variant}"},
        )
    finally:
        await tool.shutdown()
    proof = json.loads(raw)
    if not isinstance(proof, dict) or not proof:
        raise RuntimeError("sandbox did not return a proof object")
    if not all(isinstance(value, bool) for value in proof.values()):
        raise RuntimeError("sandbox proof must contain booleans only")
    result: dict[str, object] = {
        "status": "ok",
        "variant": args.variant,
        "proof": proof,
    }
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_output, args.output)
    return result


async def _serve(args: argparse.Namespace) -> None:
    args.primary_pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    for variant in args.variants:
        request = args.control_dir / f"{variant}.request"
        for _ in range(args.request_timeout_s):
            if request.exists():
                break
            await asyncio.sleep(1)
        else:
            raise TimeoutError(f"timed out waiting for {variant} request")

        await _run(
            argparse.Namespace(
                variant=variant,
                primary_pid_file=args.primary_pid_file,
                private_filename=args.private_filename,
                sandbox_host=args.sandbox_host,
                sandbox_port=args.sandbox_port,
                output=args.control_dir / f"{variant}.json",
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--variant")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--control-dir", type=Path)
    parser.add_argument("--request-timeout-s", type=int, default=600)
    parser.add_argument("--primary-pid-file", type=Path, required=True)
    parser.add_argument("--private-filename", required=True)
    parser.add_argument("--sandbox-host", default="127.0.0.1")
    parser.add_argument("--sandbox-port", type=int, default=6000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.serve:
        if not args.variants or args.control_dir is None:
            parser.error("--serve requires --variants and --control-dir")
        asyncio.run(_serve(args))
        print(json.dumps({"driver_completed": True}, sort_keys=True))
        return
    if args.variant is None or args.output is None:
        parser.error("single-run mode requires --variant and --output")
    print(json.dumps(asyncio.run(_run(args)), sort_keys=True))


if __name__ == "__main__":
    main()
