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
"""NeMo Gym resource server for an execution-grounded SciProbe reward."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import sys
from pathlib import Path
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
from pydantic import BaseModel, ConfigDict
from sciprobe_capability import last_assistant_text

try:
    from .reward_contract import parse_candidate, passes_contract
except ImportError:
    from reward_contract import parse_candidate, passes_contract


RUNNER_PATH = Path(__file__).with_name("grader_runner.py")


class SciProbeDefinition(BaseModel):
    root: str
    checks_sha256: str
    data_tree_sha256: str
    source_ref: str
    answer_keys: list[str]


class SciProbeChecksConfig(BaseResourcesServerConfig):
    #: Outer bound, enforced here. Keep it above checker_process_timeout_s so the inner
    #: limit fires first and reports which probe and why, instead of a bare timeout.
    checker_timeout_s: float = 150.0
    #: Inner bound, passed to grader_runner and applied to one checks.py run.
    checker_process_timeout_s: float = 120.0
    #: Interpreter that runs checks.py. Empty means this process. Most probes recompute
    #: their gold with numpy, pandas, scipy or R, which the Gym venv does not carry, so
    #: point this at the grading environment.
    checker_python: str = ""
    auth_token_env_var: str = "SCIPROBE_VERIFIER_TOKEN"
    auth_header_name: str = "X-SciProbe-Verifier-Token"
    probes: dict[str, SciProbeDefinition]


class SciProbeChecksRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    probe_id: str


class SciProbeChecksVerifyRequest(SciProbeChecksRunRequest, BaseVerifyRequest):
    pass


class SciProbeChecksVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    probe_id: str
    parse_status: str
    exact_key_set: bool
    extracted_answer: Optional[Any] = None
    grader_status: str
    check_results: list[list[Any]]
    failed_checks: list[str]
    checks_sha256: str
    data_tree_sha256: str
    data_files: int
    data_bytes: int
    source_ref: str


async def _run_hidden_checker(
    definition: SciProbeDefinition,
    answer: Any,
    timeout_s: float,
    checker_python: str = "",
    checker_process_timeout_s: float | None = None,
) -> dict[str, Any]:
    argv = [
        str(RUNNER_PATH),
        "--probe-root",
        definition.root,
        "--checks-sha256",
        definition.checks_sha256,
        "--data-tree-sha256",
        definition.data_tree_sha256,
    ]
    if checker_python:
        argv += ["--checker-python", checker_python]
    if checker_process_timeout_s is not None:
        argv += ["--checker-timeout", str(checker_process_timeout_s)]
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(json.dumps(answer).encode("utf-8")),
            timeout=timeout_s,
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError(f"SciProbe checker timed out after {timeout_s}s") from None

    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"SciProbe checker exited {process.returncode}: {detail[:1000]}"
        )
    try:
        result = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("SciProbe checker returned invalid JSON") from error
    if (
        not isinstance(result, dict)
        or result.get("status") not in {"ok", "answer_rejected"}
        or not isinstance(result.get("checks"), list)
        or not all(
            isinstance(row, list)
            and len(row) == 2
            and isinstance(row[0], str)
            and isinstance(row[1], bool)
            for row in result["checks"]
        )
    ):
        raise RuntimeError("SciProbe checker returned an invalid result schema")
    if result.get("checks_sha256") != definition.checks_sha256:
        raise RuntimeError("SciProbe checker provenance mismatch for checks.py")
    if result.get("data_tree_sha256") != definition.data_tree_sha256:
        raise RuntimeError("SciProbe checker provenance mismatch for data tree")
    return result


class SciProbeChecksResourcesServer(SimpleResourcesServer):
    config: SciProbeChecksConfig

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
                        status_code=403,
                        content={"detail": "forbidden"},
                    )
            return await call_next(request)

        return app

    async def verify(
        self, body: SciProbeChecksVerifyRequest
    ) -> SciProbeChecksVerifyResponse:
        definition = self.config.probes.get(body.probe_id)
        if definition is None:
            raise ValueError(f"probe_id is not allowlisted: {body.probe_id!r}")

        text = last_assistant_text(body.response)
        parse_status, answer, exact_key_set = parse_candidate(
            text,
            definition.answer_keys,
        )
        grader_result = await _run_hidden_checker(
            definition,
            answer,
            self.config.checker_timeout_s,
            self.config.checker_python,
            self.config.checker_process_timeout_s,
        )
        check_results = grader_result["checks"]

        failed_checks = [name for name, passed in check_results if not passed]
        passed = passes_contract(parse_status, exact_key_set, check_results)
        return SciProbeChecksVerifyResponse(
            **body.model_dump(),
            reward=1.0 if passed else 0.0,
            parse_status=parse_status,
            exact_key_set=exact_key_set,
            extracted_answer=answer,
            grader_status=grader_result["status"],
            check_results=check_results,
            failed_checks=failed_checks,
            checks_sha256=grader_result["checks_sha256"],
            data_tree_sha256=grader_result["data_tree_sha256"],
            data_files=int(grader_result["data_files"]),
            data_bytes=int(grader_result["data_bytes"]),
            source_ref=definition.source_ref,
        )


if __name__ == "__main__":
    SciProbeChecksResourcesServer.run_webserver()
