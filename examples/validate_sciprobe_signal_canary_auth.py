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

"""Validate fail-closed, at-most-once SciProbe verification capabilities."""

from __future__ import annotations

import asyncio
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException
from nemo_gym.server_utils import ServerClient
from nemo_gym_extensions.resources_servers.sciprobe_ns_tools.app import (
    SciProbeNSToolsConfig,
    SciProbeNSToolsResourcesServer,
    _request_sha256,
    _sha256,
)
from resources_servers.ns_tools.app import NSToolsVerifyRequest
from sciprobe_capability import (
    CAPABILITY_HEADER,
    SIGNING_KEY_ENV,
    load_signing_key,
    mint_capability,
    verify_capability,
)
from sciprobe_capability_store import CapabilityResultStore

AUTH_HEADER = "X-SciProbe-Verifier-Token"
PROBE_ID = "q3:c013:d0"


def _body(*, candidate: str = "{}", rollout_index: int = 0) -> NSToolsVerifyRequest:
    return NSToolsVerifyRequest.model_validate(
        {
            "responses_create_params": {
                "input": [{"role": "user", "content": "preflight"}],
            },
            "response": {
                "id": "resp_auth_preflight",
                "created_at": 0.0,
                "model": "preflight",
                "object": "response",
                "output": [
                    {
                        "id": "msg_auth_preflight",
                        "content": [
                            {
                                "annotations": [],
                                "text": candidate,
                                "type": "output_text",
                            }
                        ],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
            "probe_id": PROBE_ID,
            "verifier_type": "sciprobe_checks",
            "_ng_task_index": 0,
            "_ng_rollout_index": rollout_index,
            "_ng_attempt_index": 0,
        }
    )


def _capability(body: NSToolsVerifyRequest) -> str:
    extra = body.model_extra or {}
    return mint_capability(
        signing_key=load_signing_key(),
        probe_id=extra["probe_id"],
        task_index=extra["_ng_task_index"],
        rollout_index=extra["_ng_rollout_index"],
        attempt_index=extra["_ng_attempt_index"],
        response=body.response,
    )


def _request(capability: str | None) -> MagicMock:
    headers = {} if capability is None else {CAPABILITY_HEADER: capability}
    return MagicMock(session={}, headers=headers)


def _response(payload: dict[str, object]) -> MagicMock:
    response = MagicMock(ok=True)
    response.json = AsyncMock(return_value=payload)
    return response


def _config() -> SciProbeNSToolsConfig:
    return SciProbeNSToolsConfig(
        host="127.0.0.1",
        port=1,
        entrypoint="app.py",
        name="ns_tools",
        num_workers=1,
        default_verifier="sciprobe_checks",
        verifiers={
            "sciprobe_checks": {
                "type": "resources_servers",
                "name": "sciprobe_checks",
            }
        },
        nemo_skills_tools=[],
        capability_wait_timeout_s=0.2,
        capability_poll_interval_s=0.005,
    )


def _server(client: MagicMock) -> SciProbeNSToolsResourcesServer:
    server = SciProbeNSToolsResourcesServer(config=_config(), server_client=client)
    server.tool_manager = None
    return server


async def _expect_http(
    status_code: int,
    server: SciProbeNSToolsResourcesServer,
    request: MagicMock,
    body: NSToolsVerifyRequest,
) -> None:
    try:
        await server.verify(request, body)
    except HTTPException as error:
        assert error.status_code == status_code
        expected_detail = (
            "forbidden" if status_code == 403 else "verification unavailable"
        )
        assert error.detail == expected_detail
    else:
        raise AssertionError(f"expected HTTP {status_code}")


async def _main() -> dict[str, object]:
    assert len(os.environ.get(SIGNING_KEY_ENV, "")) == 64
    assert os.environ.get("SCIPROBE_CAPABILITY_STORE_PATH", "")
    backend_token = os.environ.get("SCIPROBE_VERIFIER_TOKEN", "")
    assert len(backend_token) >= 32

    client = MagicMock(spec=ServerClient)
    client.post = AsyncMock(
        return_value=_response(
            {
                "reward": 1.0,
                "probe_id": PROBE_ID,
                "parse_status": "must-not-cross-boundary",
                "check_results": [["secret-check", True]],
            }
        )
    )
    server = _server(client)

    body = _body()
    valid = _capability(body)
    payload_part, signature_part = valid.split(".")
    tampered_signature = ("A" if signature_part[0] != "A" else "B") + signature_part[1:]
    tampered = f"{payload_part}.{tampered_signature}"
    wrong_candidate_body = _body(candidate="different")
    wrong_candidate_capability = _capability(wrong_candidate_body)
    for capability in (None, tampered, wrong_candidate_capability):
        await _expect_http(403, server, _request(capability), body)
    assert client.post.await_count == 0

    # A completed result can be requested again. A new server process returns
    # the scalar cached in SQLite without calling the verifier again.
    result = await server.verify(_request(valid), body)
    restarted_server = _server(client)
    retried_result = await restarted_server.verify(_request(valid), body)
    assert result.reward == retried_result.reward == 1.0
    assert result.delegated_response is retried_result.delegated_response is None
    assert client.post.await_count == 1
    delegated_call = client.post.await_args
    assert delegated_call.kwargs["headers"] == {AUTH_HEADER: backend_token}
    assert CAPABILITY_HEADER not in delegated_call.kwargs["headers"]

    # The signed candidate still matches, but a changed prompt makes the full
    # body different. The same JTI is rejected before delegation.
    mismatched_payload = body.model_dump(mode="json")
    mismatched_payload["responses_create_params"]["input"][0]["content"] = (
        "changed preflight"
    )
    mismatched_body = NSToolsVerifyRequest.model_validate(mismatched_payload)
    await _expect_http(403, restarted_server, _request(valid), mismatched_body)
    assert client.post.await_count == 1

    # Two independent server instances race on the same claim. One delegates;
    # the other waits for and returns the same cached scalar.
    started = asyncio.Event()
    release = asyncio.Event()
    concurrent_client = MagicMock(spec=ServerClient)

    async def delayed_post(**kwargs):
        started.set()
        await release.wait()
        return _response({"reward": 0.0, "check_results": [["hidden", False]]})

    concurrent_client.post = AsyncMock(side_effect=delayed_post)
    concurrent_body = _body(rollout_index=1)
    concurrent_capability = _capability(concurrent_body)
    first = asyncio.create_task(
        _server(concurrent_client).verify(
            _request(concurrent_capability), concurrent_body
        )
    )
    await started.wait()
    second = asyncio.create_task(
        _server(concurrent_client).verify(
            _request(concurrent_capability), concurrent_body
        )
    )
    await asyncio.sleep(0.02)
    release.set()
    concurrent_results = await asyncio.gather(first, second)
    assert [item.reward for item in concurrent_results] == [0.0, 0.0]
    assert concurrent_client.post.await_count == 1

    # A pending row left by a dead owner is never taken over or delegated.
    abandoned_body = _body(rollout_index=2)
    abandoned_capability = _capability(abandoned_body)
    abandoned_extra = abandoned_body.model_extra or {}
    abandoned_claims = verify_capability(
        abandoned_capability,
        signing_key=load_signing_key(),
        probe_id=abandoned_extra["probe_id"],
        task_index=abandoned_extra["_ng_task_index"],
        rollout_index=abandoned_extra["_ng_rollout_index"],
        attempt_index=abandoned_extra["_ng_attempt_index"],
        response=abandoned_body.response,
    )
    store = CapabilityResultStore(os.environ["SCIPROBE_CAPABILITY_STORE_PATH"])
    assert (
        store.claim(
            jti=abandoned_claims.jti,
            token_sha256=_sha256(abandoned_capability),
            request_sha256=_request_sha256(abandoned_body),
            expires_at=abandoned_claims.expires_at,
            now=int(time.time()),
        ).state
        == "owner"
    )
    abandoned_client = MagicMock(spec=ServerClient)
    abandoned_client.post = AsyncMock()
    await _expect_http(
        503,
        _server(abandoned_client),
        _request(abandoned_capability),
        abandoned_body,
    )
    assert abandoned_client.post.await_count == 0

    # Failed checker responses are terminal and retain no details.
    invalid_client = MagicMock(spec=ServerClient)
    invalid_client.post = AsyncMock()
    invalid_server = _server(invalid_client)
    for rollout_index, invalid_reward in enumerate(("1", float("nan"), 0.5), start=3):
        invalid_client.post.return_value = _response({"reward": invalid_reward})
        invalid_body = _body(rollout_index=rollout_index)
        invalid_capability = _capability(invalid_body)
        await _expect_http(
            503, invalid_server, _request(invalid_capability), invalid_body
        )
        calls_after_failure = invalid_client.post.await_count
        await _expect_http(
            503, invalid_server, _request(invalid_capability), invalid_body
        )
        assert invalid_client.post.await_count == calls_after_failure

    expiry_jti = "expiry-cleanup-id"
    now = int(time.time())
    assert (
        store.claim(
            jti=expiry_jti,
            token_sha256="a" * 64,
            request_sha256="b" * 64,
            expires_at=now + 1,
            now=now,
        ).state
        == "owner"
    )
    assert store.cleanup_expired(now=now) == 0
    assert store.cleanup_expired(now=now + 1) >= 1
    assert (
        store.lookup(
            jti=expiry_jti,
            token_sha256="a" * 64,
            request_sha256="b" * 64,
            now=now + 1,
        ).state
        == "missing"
    )

    serialized_results = json.dumps(
        [result.model_dump(), retried_result.model_dump()], sort_keys=True
    )
    for forbidden in (
        valid,
        backend_token,
        SIGNING_KEY_ENV,
        CAPABILITY_HEADER,
        "must-not-cross-boundary",
        "secret-check",
    ):
        assert forbidden not in serialized_results

    return {
        "status": "ok",
        "invalid_capabilities_rejected": 3,
        "completed_result_retry_cached": True,
        "concurrent_identical_retries_cached": True,
        "mismatched_body_rejected": True,
        "restart_shared_store": True,
        "ambiguous_pending_not_replayed": True,
        "invalid_rewards_terminal": 3,
        "expiry_cleanup": True,
        "capability_scrubbed": True,
        "backend_details_scrubbed": True,
    }


def main() -> None:
    print(json.dumps(asyncio.run(_main()), sort_keys=True))


if __name__ == "__main__":
    main()
