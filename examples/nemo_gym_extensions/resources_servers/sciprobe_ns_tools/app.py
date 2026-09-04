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

"""ns_tools variant that authenticates delegated SciProbe verification."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sqlite3
import time
from typing import Any

from fastapi import HTTPException, Request
from nemo_gym.server_utils import SESSION_ID_KEY, raise_for_status
from pydantic import Field
from resources_servers.ns_tools.app import (
    NSToolsConfig,
    NSToolsResourcesServer,
    NSToolsVerifyRequest,
    NSToolsVerifyResponse,
)
from sciprobe_capability import (
    CAPABILITY_HEADER,
    SIGNING_KEY_ENV,
    CapabilityError,
    load_signing_key,
    verify_capability,
)
from sciprobe_capability_store import CapabilityResultStore


class SciProbeNSToolsConfig(NSToolsConfig):
    verifier_auth_token_env_var: str = "SCIPROBE_VERIFIER_TOKEN"
    verifier_auth_header_name: str = "X-SciProbe-Verifier-Token"
    capability_signing_key_env_var: str = SIGNING_KEY_ENV
    capability_header_name: str = CAPABILITY_HEADER
    capability_store_path_env_var: str = "SCIPROBE_CAPABILITY_STORE_PATH"
    capability_wait_timeout_s: float = Field(default=30.0, gt=0, le=900)
    capability_poll_interval_s: float = Field(default=0.05, gt=0, le=1)


def _required_index(extra: dict[str, Any], key: str) -> int:
    value = extra.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CapabilityError(f"missing or invalid {key}")
    return value


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _request_sha256(body: NSToolsVerifyRequest) -> str:
    """Hash every normalized verify-body field without retaining the body."""
    serialized = json.dumps(
        body.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _sha256(serialized)


class SciProbeNSToolsResourcesServer(NSToolsResourcesServer):
    config: SciProbeNSToolsConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        store_path = os.environ.get(self.config.capability_store_path_env_var, "")
        if not store_path:
            raise RuntimeError("capability result store is not configured")
        self._capability_store = CapabilityResultStore(store_path)

    @staticmethod
    def _generic_unavailable() -> HTTPException:
        return HTTPException(status_code=503, detail="verification unavailable")

    async def _wait_for_cached_reward(
        self,
        *,
        jti: str,
        token_sha256: str,
        request_sha256: str,
        expires_at: int,
    ) -> float:
        deadline = min(
            asyncio.get_running_loop().time() + self.config.capability_wait_timeout_s,
            asyncio.get_running_loop().time() + max(0, expires_at - int(time.time())),
        )
        while asyncio.get_running_loop().time() < deadline:
            try:
                cached = await asyncio.to_thread(
                    self._capability_store.lookup,
                    jti=jti,
                    token_sha256=token_sha256,
                    request_sha256=request_sha256,
                    now=int(time.time()),
                )
            except (OSError, sqlite3.Error, ValueError):
                raise self._generic_unavailable() from None
            if cached.state == "completed":
                assert cached.reward in (0.0, 1.0)
                return cached.reward
            if cached.state == "mismatch":
                raise HTTPException(status_code=403, detail="forbidden")
            if cached.state in {"failed", "missing"}:
                raise self._generic_unavailable()
            await asyncio.sleep(self.config.capability_poll_interval_s)
        raise self._generic_unavailable()

    async def _delegate_reward(
        self,
        body: NSToolsVerifyRequest,
        sanitized_body: dict[str, Any],
    ) -> float:
        verifier_type = body.verifier_type or self.config.default_verifier
        if verifier_type not in self.config.verifiers:
            raise ValueError("unknown verifier")
        token = os.environ.get(self.config.verifier_auth_token_env_var, "")
        if len(token) < 32:
            raise RuntimeError("verifier authentication is not configured")

        verifier_ref = self.config.verifiers[verifier_type]
        response = await self.server_client.post(
            server_name=verifier_ref.name,
            url_path="/verify",
            json=sanitized_body,
            headers={self.config.verifier_auth_header_name: token},
        )
        await raise_for_status(response)
        result = await response.json()
        if not isinstance(result, dict):
            raise ValueError("invalid verifier response")
        raw_reward = result.get("reward")
        if isinstance(raw_reward, bool) or not isinstance(raw_reward, (int, float)):
            raise ValueError("invalid verifier reward")
        reward = float(raw_reward)
        if not math.isfinite(reward) or reward not in (0.0, 1.0):
            raise ValueError("invalid verifier reward")
        return reward

    async def verify(
        self, request: Request, body: NSToolsVerifyRequest
    ) -> NSToolsVerifyResponse:
        """Delegate verification with a credential unavailable to model code."""
        session_id = request.session.get(SESSION_ID_KEY) if request else None
        try:
            extra = body.model_extra or {}
            presented_capability = request.headers.get(
                self.config.capability_header_name, ""
            )
            try:
                claims = verify_capability(
                    presented_capability,
                    signing_key=load_signing_key(
                        self.config.capability_signing_key_env_var
                    ),
                    probe_id=extra.get("probe_id"),
                    task_index=_required_index(extra, "_ng_task_index"),
                    rollout_index=_required_index(extra, "_ng_rollout_index"),
                    attempt_index=_required_index(extra, "_ng_attempt_index"),
                    response=body.response,
                )
            except CapabilityError as error:
                raise HTTPException(status_code=403, detail="forbidden") from error
            sanitized_body = body.model_dump()
            token_sha256 = _sha256(presented_capability)
            request_sha256 = _request_sha256(body)
            try:
                claim = await asyncio.to_thread(
                    self._capability_store.claim,
                    jti=claims.jti,
                    token_sha256=token_sha256,
                    request_sha256=request_sha256,
                    expires_at=claims.expires_at,
                    now=int(time.time()),
                )
            except (OSError, sqlite3.Error, ValueError):
                raise self._generic_unavailable() from None

            if claim.state == "mismatch":
                raise HTTPException(status_code=403, detail="forbidden")
            if claim.state == "failed":
                raise self._generic_unavailable()
            if claim.state == "pending":
                reward = await self._wait_for_cached_reward(
                    jti=claims.jti,
                    token_sha256=token_sha256,
                    request_sha256=request_sha256,
                    expires_at=claims.expires_at,
                )
            elif claim.state == "completed":
                assert claim.reward in (0.0, 1.0)
                reward = claim.reward
            else:
                assert claim.state == "owner"
                try:
                    reward = await self._delegate_reward(body, sanitized_body)
                    stored = await asyncio.shield(
                        asyncio.to_thread(
                            self._capability_store.complete,
                            jti=claims.jti,
                            token_sha256=token_sha256,
                            request_sha256=request_sha256,
                            reward=reward,
                        )
                    )
                    if not stored:
                        raise RuntimeError("capability result claim was lost")
                except BaseException as error:
                    try:
                        await asyncio.shield(
                            asyncio.to_thread(
                                self._capability_store.fail,
                                jti=claims.jti,
                                token_sha256=token_sha256,
                                request_sha256=request_sha256,
                            )
                        )
                    except (OSError, sqlite3.Error, ValueError):
                        pass
                    if isinstance(error, asyncio.CancelledError):
                        raise
                    raise self._generic_unavailable() from None

            metrics = self._aggregate_timing_metrics(session_id)
            return NSToolsVerifyResponse(
                **sanitized_body,
                reward=reward,
                delegated_response=None,
                **metrics,
            )
        finally:
            if self.tool_manager is not None and session_id:
                await self.tool_manager.cleanup_request(session_id)


if __name__ == "__main__":
    SciProbeNSToolsResourcesServer.run_webserver()
