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

"""Short-lived, response-bound capabilities for SciProbe verification."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any

CAPABILITY_AUDIENCE = "sciprobe.verify"
CAPABILITY_HEADER = "X-SciProbe-Rollout-Capability"
SIGNING_KEY_ENV = "SCIPROBE_CAPABILITY_SIGNING_KEY"
CAPABILITY_VERSION = 1
MAX_CAPABILITY_BYTES = 2048
MAX_CAPABILITY_TTL_SECONDS = 900
_CLAIM_KEYS = {
    "aud",
    "attempt_index",
    "candidate_sha256",
    "exp",
    "iat",
    "jti",
    "probe_id",
    "rollout_index",
    "task_index",
    "v",
}


class CapabilityError(ValueError):
    """A capability is missing, malformed, expired, or bound to another rollout."""


@dataclass(frozen=True)
class CapabilityClaims:
    probe_id: str
    task_index: int
    rollout_index: int
    attempt_index: int
    candidate_sha256: str
    jti: str
    issued_at: int
    expires_at: int


def last_assistant_text(response: Any) -> str:
    """Return the final visible assistant text from a NeMo Gym response."""
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    if not isinstance(output, list):
        return ""

    for item in reversed(output):
        item_type = getattr(item, "type", None)
        item_role = getattr(item, "role", None)
        content = getattr(item, "content", None)
        if isinstance(item, dict):
            item_type = item.get("type")
            item_role = item.get("role")
            content = item.get("content")
        if item_type != "message" or item_role != "assistant":
            continue

        texts: list[str] = []
        if isinstance(content, list):
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(part, dict):
                    text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)
        elif isinstance(content, str):
            texts.append(content)
        if texts:
            return "\n".join(texts).strip()
    return ""


def candidate_sha256(response: Any) -> str:
    """Hash only the visible final answer used by the SciProbe checker."""
    return hashlib.sha256(last_assistant_text(response).encode("utf-8")).hexdigest()


def load_signing_key(env_var: str = SIGNING_KEY_ENV) -> bytes:
    """Load a 32-byte hexadecimal signing key without exposing its value."""
    encoded = os.environ.get(env_var, "")
    try:
        key = bytes.fromhex(encoded)
    except ValueError as error:
        raise RuntimeError(
            f"{env_var} must contain a 32-byte hexadecimal key"
        ) from error
    if len(key) != 32 or len(encoded) != 64:
        raise RuntimeError(f"{env_var} must contain a 32-byte hexadecimal key")
    return key


def _require_index(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CapabilityError(f"{name} must be an integer")
    if value < 0:
        raise CapabilityError(f"{name} must be non-negative")
    return value


def _require_probe_id(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 256:
        raise CapabilityError(
            "probe_id must be a non-empty string of at most 256 bytes"
        )
    return value


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise CapabilityError("capability is malformed")
    padding = "=" * (-len(value) % 4)
    try:
        return base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (ValueError, TypeError) as error:
        raise CapabilityError("capability is malformed") from error


def mint_capability(
    *,
    signing_key: bytes,
    probe_id: str,
    task_index: int,
    rollout_index: int,
    attempt_index: int,
    response: Any,
    ttl_seconds: int = 300,
    now: int | None = None,
    jti: str | None = None,
) -> str:
    """Mint an opaque capability bound to one rollout attempt and candidate."""
    if len(signing_key) != 32:
        raise CapabilityError("signing key must be 32 bytes")
    probe_id = _require_probe_id(probe_id)
    task_index = _require_index(task_index, "task_index")
    rollout_index = _require_index(rollout_index, "rollout_index")
    attempt_index = _require_index(attempt_index, "attempt_index")
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
        raise CapabilityError("ttl_seconds must be an integer")
    if not 1 <= ttl_seconds <= MAX_CAPABILITY_TTL_SECONDS:
        raise CapabilityError(
            f"ttl_seconds must be between 1 and {MAX_CAPABILITY_TTL_SECONDS}"
        )
    issued_at = int(time.time()) if now is None else _require_index(now, "now")
    token_id = jti or secrets.token_urlsafe(16)
    if not isinstance(token_id, str) or not 16 <= len(token_id) <= 128:
        raise CapabilityError("jti has an invalid length")

    payload = {
        "aud": CAPABILITY_AUDIENCE,
        "attempt_index": attempt_index,
        "candidate_sha256": candidate_sha256(response),
        "exp": issued_at + ttl_seconds,
        "iat": issued_at,
        "jti": token_id,
        "probe_id": probe_id,
        "rollout_index": rollout_index,
        "task_index": task_index,
        "v": CAPABILITY_VERSION,
    }
    payload_bytes = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    signature = hmac.digest(signing_key, payload_bytes, "sha256")
    return f"{_b64encode(payload_bytes)}.{_b64encode(signature)}"


def verify_capability(
    token: str,
    *,
    signing_key: bytes,
    probe_id: str,
    task_index: int,
    rollout_index: int,
    attempt_index: int,
    response: Any,
    now: int | None = None,
) -> CapabilityClaims:
    """Verify signature, lifetime, and exact rollout/candidate binding."""
    if not isinstance(token, str) or not token or len(token) > MAX_CAPABILITY_BYTES:
        raise CapabilityError("capability is missing or too large")
    if len(signing_key) != 32:
        raise CapabilityError("signing key must be 32 bytes")
    parts = token.split(".")
    if len(parts) != 2:
        raise CapabilityError("capability is malformed")
    payload_bytes = _b64decode(parts[0])
    signature = _b64decode(parts[1])
    expected_signature = hmac.digest(signing_key, payload_bytes, "sha256")
    if len(signature) != len(expected_signature) or not hmac.compare_digest(
        signature, expected_signature
    ):
        raise CapabilityError("capability signature is invalid")
    try:
        payload = json.loads(payload_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CapabilityError("capability payload is malformed") from error
    if not isinstance(payload, dict) or set(payload) != _CLAIM_KEYS:
        raise CapabilityError("capability claims are invalid")
    if payload["v"] != CAPABILITY_VERSION or payload["aud"] != CAPABILITY_AUDIENCE:
        raise CapabilityError("capability audience or version is invalid")

    claimed_probe_id = _require_probe_id(payload["probe_id"])
    claimed_task_index = _require_index(payload["task_index"], "task_index")
    claimed_rollout_index = _require_index(payload["rollout_index"], "rollout_index")
    claimed_attempt_index = _require_index(payload["attempt_index"], "attempt_index")
    issued_at = _require_index(payload["iat"], "iat")
    expires_at = _require_index(payload["exp"], "exp")
    current_time = int(time.time()) if now is None else _require_index(now, "now")
    if expires_at <= issued_at or expires_at - issued_at > MAX_CAPABILITY_TTL_SECONDS:
        raise CapabilityError("capability lifetime is invalid")
    if issued_at > current_time + 5 or expires_at <= current_time:
        raise CapabilityError("capability is expired or not yet valid")
    token_id = payload["jti"]
    if not isinstance(token_id, str) or not 16 <= len(token_id) <= 128:
        raise CapabilityError("capability jti is invalid")
    claimed_digest = payload["candidate_sha256"]
    if not isinstance(claimed_digest, str) or len(claimed_digest) != 64:
        raise CapabilityError("candidate digest is invalid")

    expected = (
        _require_probe_id(probe_id),
        _require_index(task_index, "task_index"),
        _require_index(rollout_index, "rollout_index"),
        _require_index(attempt_index, "attempt_index"),
        candidate_sha256(response),
    )
    claimed = (
        claimed_probe_id,
        claimed_task_index,
        claimed_rollout_index,
        claimed_attempt_index,
        claimed_digest,
    )
    if claimed != expected:
        raise CapabilityError("capability is bound to another rollout")

    return CapabilityClaims(
        probe_id=claimed_probe_id,
        task_index=claimed_task_index,
        rollout_index=claimed_rollout_index,
        attempt_index=claimed_attempt_index,
        candidate_sha256=claimed_digest,
        jti=token_id,
        issued_at=issued_at,
        expires_at=expires_at,
    )
