from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

import nemo_rl.environments.nemo_gym_loopback as nemo_gym_loopback_module
from nemo_rl.environments.nemo_gym_loopback import (
    assert_nemo_gym_node_ip_ingress_denied,
    validate_nemo_gym_loopback_bindings,
)

NEMO_GYM_SOURCE = Path("nemo_rl/environments/nemo_gym.py")


def test_loopback_mode_does_not_use_hidden_non_none_get_defaults() -> None:
    source = NEMO_GYM_SOURCE.read_text(encoding="utf-8")
    assert '.get("loopback_only", False)' not in source
    assert '.pop("loopback_only", False)' not in source
    assert "loopback_only = False" not in source


@dataclass
class _FakeServerInstance:
    name: str
    config: Any

    def get_inner_run_server_config_dict(self) -> Any:
        return self.config


def _resolved_config(*, child_host: str = "127.0.0.1") -> tuple[Any, list[Any]]:
    config = OmegaConf.create(
        {
            "head_server": {"host": "127.0.0.1", "port": 5100},
            "child": {"host": child_host, "port": 5101},
        }
    )
    return config, [_FakeServerInstance("child", config.child)]


def test_loopback_binding_validation_returns_head_and_child_ports() -> None:
    config, instances = _resolved_config()

    assert validate_nemo_gym_loopback_bindings(
        global_config_dict=config,
        server_instance_configs=instances,
        head_server_key_name="head_server",
    ) == [("head_server", 5100), ("child", 5101)]


def test_loopback_binding_validation_rejects_explicit_child_exposure() -> None:
    config, instances = _resolved_config(child_host="0.0.0.0")

    with pytest.raises(RuntimeError, match="every child server"):
        validate_nemo_gym_loopback_bindings(
            global_config_dict=config,
            server_instance_configs=instances,
            head_server_key_name="head_server",
        )


def test_node_ip_probe_checks_every_loopback_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked: list[tuple[str, int]] = []

    class _ProbeSocket:
        def __enter__(self) -> "_ProbeSocket":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def settimeout(self, _timeout: float) -> None:
            return None

        def connect_ex(self, address: tuple[str, int]) -> int:
            checked.append(address)
            return 111

    monkeypatch.setattr(
        nemo_gym_loopback_module.socket,
        "socket",
        lambda *_args: _ProbeSocket(),
    )

    assert_nemo_gym_node_ip_ingress_denied(
        node_ip="10.2.3.4",
        bindings=[("head_server", 5100), ("child", 5101)],
    )

    assert checked == [("10.2.3.4", 5100), ("10.2.3.4", 5101)]


def test_node_ip_probe_fails_when_global_config_head_is_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ProbeSocket:
        def __enter__(self) -> "_ProbeSocket":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def settimeout(self, _timeout: float) -> None:
            return None

        def connect_ex(self, _address: tuple[str, int]) -> int:
            return 0

    monkeypatch.setattr(
        nemo_gym_loopback_module.socket,
        "socket",
        lambda *_args: _ProbeSocket(),
    )

    with pytest.raises(RuntimeError, match="head_server=10.2.3.4:5100"):
        assert_nemo_gym_node_ip_ingress_denied(
            node_ip="10.2.3.4",
            bindings=[("head_server", 5100)],
        )
