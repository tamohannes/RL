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

"""Authenticated deterministic verifier for the stateful-choice overfit task."""

from __future__ import annotations

import hmac
import os
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from pydantic import BaseModel, ConfigDict, Field

try:
    from .reward_contract import score_stateful_choice_trace
except ImportError:
    from reward_contract import score_stateful_choice_trace


class OverfitDefinition(BaseModel):
    tool_name: str
    state_name: str
    state_value: int
    expected_second_output: str
    choices: list[str] = Field(min_length=2, max_length=2)
    rewarded_choice: str


class OverfitChecksConfig(BaseResourcesServerConfig):
    auth_token_env_var: str = "SCIPROBE_VERIFIER_TOKEN"
    auth_header_name: str = "X-SciProbe-Verifier-Token"
    probes: dict[str, OverfitDefinition]


class OverfitChecksRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    probe_id: str


class OverfitChecksVerifyRequest(OverfitChecksRunRequest, BaseVerifyRequest):
    pass


class OverfitChecksVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    probe_id: str
    trace_valid: bool
    selected_choice: Optional[str] = None
    grader_status: str
    check_results: list[list[Any]]
    failed_checks: list[str]
    rewarded_choice: str


class OverfitChecksResourcesServer(SimpleResourcesServer):
    config: OverfitChecksConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        @app.middleware("http")
        async def require_verifier_token(request: Request, call_next):
            if request.url.path.rstrip("/").endswith("/verify"):
                expected = os.environ.get(self.config.auth_token_env_var, "")
                presented = request.headers.get(self.config.auth_header_name, "")
                if len(expected) < 32:
                    return JSONResponse(
                        status_code=503,
                        content={"detail": "verifier authentication unavailable"},
                    )
                if not hmac.compare_digest(presented, expected):
                    return JSONResponse(
                        status_code=403, content={"detail": "forbidden"}
                    )
            return await call_next(request)

        return app

    async def verify(
        self, body: OverfitChecksVerifyRequest
    ) -> OverfitChecksVerifyResponse:
        definition = self.config.probes.get(body.probe_id)
        if definition is None:
            raise ValueError(f"probe_id is not allowlisted: {body.probe_id!r}")
        reward, selected_choice, check_results = score_stateful_choice_trace(
            body.response,
            tool_name=definition.tool_name,
            state_name=definition.state_name,
            state_value=definition.state_value,
            expected_second_output=definition.expected_second_output,
            choices=definition.choices,
            rewarded_choice=definition.rewarded_choice,
        )
        failed_checks = [name for name, passed in check_results if not passed]
        return OverfitChecksVerifyResponse(
            **body.model_dump(),
            reward=reward,
            trace_valid=not failed_checks,
            selected_choice=selected_choice,
            grader_status="stateful_choice",
            check_results=check_results,
            failed_checks=failed_checks,
            rewarded_choice=definition.rewarded_choice,
        )


if __name__ == "__main__":
    OverfitChecksResourcesServer.run_webserver()
