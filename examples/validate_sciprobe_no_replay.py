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

"""Prove side-effecting Gym requests are not replayed after execution."""

from __future__ import annotations

import asyncio
import json

import nemo_gym.server_utils as server_utils
from aiohttp import ClientSession


async def _prove_path(path: str) -> dict[str, object]:
    executions = 0
    received_path = ""
    executed = asyncio.Event()

    async def execute_then_disconnect(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        nonlocal executions, received_path
        try:
            raw_headers = await reader.readuntil(b"\r\n\r\n")
            request_line = raw_headers.split(b"\r\n", 1)[0].decode("ascii")
            _, received_path, _ = request_line.split(" ", 2)
            content_length = 0
            for line in raw_headers.split(b"\r\n"):
                name, separator, value = line.partition(b":")
                if separator and name.strip().lower() == b"content-length":
                    content_length = int(value.strip())
                    break
            if content_length:
                await reader.readexactly(content_length)
            executions += 1
            executed.set()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(
        execute_then_disconnect,
        host="127.0.0.1",
        port=0,
    )
    port = int(server.sockets[0].getsockname()[1])
    failure = ""
    async with ClientSession() as client:
        previous = server_utils._GLOBAL_AIOHTTP_CLIENT
        server_utils._GLOBAL_AIOHTTP_CLIENT = client
        try:
            try:
                await server_utils.request(
                    method="POST",
                    url=f"http://127.0.0.1:{port}{path}",
                    _internal=True,
                    json={"candidate": "answer"},
                )
            except Exception as error:
                failure = type(error).__name__
            else:
                raise AssertionError("disconnecting server unexpectedly returned")
            await asyncio.wait_for(executed.wait(), timeout=2)
            await asyncio.sleep(0.1)
        finally:
            server_utils._GLOBAL_AIOHTTP_CLIENT = previous
            server.close()
            await server.wait_closed()

    assert executions == 1, f"{path} executed {executions} times"
    assert received_path == path
    return {"path": path, "executions": executions, "failure": failure}


async def _main() -> dict[str, object]:
    proofs = [
        await _prove_path("/verify"),
        await _prove_path("/run"),
        await _prove_path("/ng-rollout/task-7/verify?attempt=1"),
        await _prove_path("/seed_session"),
        await _prove_path("/v1/responses"),
        await _prove_path("/ng-rollout/task-7/v1/responses"),
        await _prove_path("/stateful_python_code_exec"),
        await _prove_path("/ng-rollout/task-7/stateful_python_code_exec"),
    ]
    return {
        "status": "ok",
        "fail_closed_at_most_once": True,
        "proofs": proofs,
    }


def main() -> None:
    print(json.dumps(asyncio.run(_main()), sort_keys=True))


if __name__ == "__main__":
    main()
