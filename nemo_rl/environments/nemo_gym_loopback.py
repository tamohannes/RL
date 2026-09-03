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

"""Fail-closed loopback binding checks for colocated NeMo Gym services."""

import socket
from typing import Any

NEMO_GYM_LOOPBACK_HOST = "127.0.0.1"


def validate_nemo_gym_loopback_bindings(
    *,
    global_config_dict: Any,
    server_instance_configs: list[Any],
    head_server_key_name: str,
) -> list[tuple[str, int]]:
    """Return all Gym bindings after requiring an IPv4 loopback host.

    This validation runs after Gym has merged every component config but before
    ``RunHelper`` starts any HTTP server. It therefore catches an explicit
    component ``host`` that would otherwise override ``default_host`` and
    expose the service despite the canary's loopback-only setting.
    """
    bindings: list[tuple[str, int]] = []
    head_server = global_config_dict.get(head_server_key_name)
    if head_server is None:
        raise RuntimeError("loopback-only NeMo Gym config has no head server")

    head_host = head_server.get("host")
    head_port = head_server.get("port")
    if head_host != NEMO_GYM_LOOPBACK_HOST or not isinstance(head_port, int):
        raise RuntimeError(
            "loopback-only NeMo Gym requires the head server to bind "
            f"{NEMO_GYM_LOOPBACK_HOST}; got {head_host!r}:{head_port!r}"
        )
    bindings.append((head_server_key_name, head_port))

    for instance in server_instance_configs:
        config = instance.get_inner_run_server_config_dict()
        host = config.get("host")
        port = config.get("port")
        if host != NEMO_GYM_LOOPBACK_HOST or not isinstance(port, int):
            raise RuntimeError(
                "loopback-only NeMo Gym requires every child server to bind "
                f"{NEMO_GYM_LOOPBACK_HOST}; {instance.name!r} resolved to "
                f"{host!r}:{port!r}"
            )
        bindings.append((instance.name, port))

    return bindings


def assert_nemo_gym_node_ip_ingress_denied(
    *, node_ip: str, bindings: list[tuple[str, int]]
) -> None:
    """Fail if a loopback-only Gym service accepts TCP via the Ray node IP."""
    if node_ip in {NEMO_GYM_LOOPBACK_HOST, "::1", "localhost"}:
        # Local-only development has no distinct node interface to probe. The
        # resolved binding validation above remains mandatory.
        return

    exposed: list[str] = []
    address_family = socket.AF_INET6 if ":" in node_ip else socket.AF_INET
    for name, port in bindings:
        with socket.socket(address_family, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.25)
            address: Any = (
                (node_ip, port, 0, 0)
                if address_family == socket.AF_INET6
                else (node_ip, port)
            )
            try:
                connect_result = probe.connect_ex(address)
            except OSError:
                # Refusal, timeout, or an unreachable interface all prove the
                # service cannot be reached through this address.
                continue
            if connect_result == 0:
                exposed.append(f"{name}={node_ip}:{port}")

    if exposed:
        raise RuntimeError(
            "loopback-only NeMo Gym service is reachable through the node IP: "
            + ", ".join(exposed)
        )
