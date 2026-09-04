from __future__ import annotations

import asyncio
import json as jsonlib
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.sciprobe_simple_agent.app import (
    SciProbeSimpleAgent,
    SciProbeSimpleAgentConfig,
)
from responses_api_agents.simple_agent.app import (
    SimpleAgentRunRequest,
    SimpleAgentVerifyRequest,
)
from sciprobe_capability import (
    CAPABILITY_HEADER,
    CapabilityError,
    mint_capability,
    verify_capability,
)

SIGNING_KEY = bytes(range(32))


def _response(candidate: str = '{"answer": 7}') -> dict:
    return {
        "id": "response-1",
        "created_at": 1.0,
        "model": "model",
        "object": "response",
        "output": [
            {
                "id": "message-1",
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
    }


def _run_and_verify_requests() -> tuple[
    SimpleAgentRunRequest, SimpleAgentVerifyRequest
]:
    run_request = SimpleAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": "question"},
            "probe_id": "probe-1",
            "_ng_task_index": 3,
            "_ng_rollout_index": 5,
            "_ng_attempt_index": 1,
        }
    )
    verify_request = SimpleAgentVerifyRequest.model_validate(
        run_request.model_dump() | {"response": _response()}
    )
    return run_request, verify_request


def test_capability_is_bound_to_rollout_attempt_and_candidate() -> None:
    response = _response()
    token = mint_capability(
        signing_key=SIGNING_KEY,
        probe_id="probe-1",
        task_index=3,
        rollout_index=5,
        attempt_index=1,
        response=response,
        ttl_seconds=30,
        now=100,
        jti="0123456789abcdef",
    )

    claims = verify_capability(
        token,
        signing_key=SIGNING_KEY,
        probe_id="probe-1",
        task_index=3,
        rollout_index=5,
        attempt_index=1,
        response=response,
        now=110,
    )
    assert claims.jti == "0123456789abcdef"
    assert claims.expires_at == 130

    mismatches = (
        {"probe_id": "probe-2"},
        {"task_index": 4},
        {"rollout_index": 6},
        {"attempt_index": 2},
        {"response": _response("different")},
    )
    base = {
        "probe_id": "probe-1",
        "task_index": 3,
        "rollout_index": 5,
        "attempt_index": 1,
        "response": response,
    }
    for mismatch in mismatches:
        with pytest.raises(CapabilityError, match="another rollout"):
            verify_capability(
                token,
                signing_key=SIGNING_KEY,
                now=110,
                **(base | mismatch),
            )

    with pytest.raises(CapabilityError, match="expired"):
        verify_capability(
            token,
            signing_key=SIGNING_KEY,
            now=130,
            **base,
        )


def test_sciprobe_agent_mints_header_without_mutating_model_visible_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCIPROBE_CAPABILITY_SIGNING_KEY", SIGNING_KEY.hex())
    config = SciProbeSimpleAgentConfig(
        host="127.0.0.1",
        port=1,
        entrypoint="app.py",
        name="agent",
        resources_server=ResourcesServerRef(
            type="resources_servers",
            name="ns_tools",
        ),
        model_server=ModelServerRef(
            type="responses_api_models",
            name="policy_model",
        ),
    )
    agent = SciProbeSimpleAgent(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    )
    run_request, verify_request = _run_and_verify_requests()
    before = run_request.model_dump(mode="json")

    first = agent.verification_headers(run_request, verify_request)
    second = agent.verification_headers(run_request, verify_request)

    assert set(first) == {CAPABILITY_HEADER}
    assert first[CAPABILITY_HEADER] != second[CAPABILITY_HEADER]
    assert run_request.model_dump(mode="json") == before
    serialized_model_body = jsonlib.dumps(
        run_request.responses_create_params.model_dump()
    )
    assert first[CAPABILITY_HEADER] not in serialized_model_body
    verify_capability(
        first[CAPABILITY_HEADER],
        signing_key=SIGNING_KEY,
        probe_id="probe-1",
        task_index=3,
        rollout_index=5,
        attempt_index=1,
        response=verify_request.response,
    )


def test_sciprobe_agent_sends_capability_only_on_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCIPROBE_CAPABILITY_SIGNING_KEY", SIGNING_KEY.hex())
    config = SciProbeSimpleAgentConfig(
        host="127.0.0.1",
        port=1,
        entrypoint="app.py",
        name="agent",
        resources_server=ResourcesServerRef(
            type="resources_servers",
            name="ns_tools",
        ),
        model_server=ModelServerRef(
            type="responses_api_models",
            name="policy_model",
        ),
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": False}
    agent = SciProbeSimpleAgent(config=config, server_client=client)
    run_request, _ = _run_and_verify_requests()
    seen_capability: str | None = None

    def mock_http(payload: dict | None = None) -> MagicMock:
        response = MagicMock(status=200, cookies={}, ok=True)
        response.read = AsyncMock(return_value=jsonlib.dumps(payload or {}))
        return response

    async def post(*, url_path: str, json: dict, **kwargs):
        nonlocal seen_capability
        if url_path == "/seed_session":
            assert "headers" not in kwargs
            assert CAPABILITY_HEADER not in jsonlib.dumps(json)
            return mock_http()
        if url_path.endswith("/v1/responses"):
            assert "headers" not in kwargs
            assert json == run_request.responses_create_params
            return mock_http(_response())
        assert url_path == "/verify"
        assert CAPABILITY_HEADER not in json
        seen_capability = kwargs["headers"][CAPABILITY_HEADER]
        verify_capability(
            seen_capability,
            signing_key=SIGNING_KEY,
            probe_id="probe-1",
            task_index=3,
            rollout_index=5,
            attempt_index=1,
            response=json["response"],
        )
        return mock_http(json | {"reward": 1.0})

    client.post = AsyncMock(side_effect=post)
    result = asyncio.run(agent.run(MagicMock(cookies={}), run_request))

    assert result.reward == 1.0
    assert seen_capability is not None
    serialized = jsonlib.dumps(result.model_dump(mode="json"))
    assert seen_capability not in serialized
    assert CAPABILITY_HEADER not in serialized
