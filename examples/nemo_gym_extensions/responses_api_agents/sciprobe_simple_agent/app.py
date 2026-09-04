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

"""SimpleAgent variant that mints a response-bound verification capability."""

from __future__ import annotations

from typing import Any

from nemo_gym.global_config import (
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from pydantic import Field
from responses_api_agents.simple_agent.app import (
    SimpleAgent,
    SimpleAgentConfig,
    SimpleAgentRunRequest,
    SimpleAgentVerifyRequest,
)
from sciprobe_capability import (
    CAPABILITY_HEADER,
    MAX_CAPABILITY_TTL_SECONDS,
    SIGNING_KEY_ENV,
    CapabilityError,
    load_signing_key,
    mint_capability,
)


class SciProbeSimpleAgentConfig(SimpleAgentConfig):
    capability_signing_key_env_var: str = SIGNING_KEY_ENV
    capability_header_name: str = CAPABILITY_HEADER
    capability_ttl_seconds: int = Field(
        default=300,
        ge=1,
        le=MAX_CAPABILITY_TTL_SECONDS,
    )


def _required_index(extra: dict[str, Any], key: str) -> int:
    value = extra.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CapabilityError(f"missing or invalid {key}")
    return value


class SciProbeSimpleAgent(SimpleAgent):
    config: SciProbeSimpleAgentConfig

    def verification_headers(
        self,
        body: SimpleAgentRunRequest,
        verify_request: SimpleAgentVerifyRequest,
    ) -> dict[str, str]:
        """Mint only after generation and attach only to the verifier request."""
        extra = body.model_extra or {}
        probe_id = extra.get("probe_id")
        if not isinstance(probe_id, str) or not probe_id:
            raise CapabilityError("missing or invalid probe_id")
        capability = mint_capability(
            signing_key=load_signing_key(self.config.capability_signing_key_env_var),
            probe_id=probe_id,
            task_index=_required_index(extra, TASK_INDEX_KEY_NAME),
            rollout_index=_required_index(extra, ROLLOUT_INDEX_KEY_NAME),
            attempt_index=_required_index(extra, ATTEMPT_INDEX_KEY_NAME),
            response=verify_request.response,
            ttl_seconds=self.config.capability_ttl_seconds,
        )
        return {self.config.capability_header_name: capability}


if __name__ == "__main__":
    SciProbeSimpleAgent.run_webserver()
