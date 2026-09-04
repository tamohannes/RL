from __future__ import annotations

import asyncio
import importlib.util
import json
import socket
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

VALIDATOR = Path("examples/validate_sciprobe_network_blocking.py")


def _load_validator() -> ModuleType:
    module_names = (
        "httpx",
        "nemo_skills",
        "nemo_skills.mcp",
        "nemo_skills.mcp.servers",
        "nemo_skills.mcp.servers.python_tool",
    )
    previous = {name: sys.modules.get(name) for name in module_names}
    for name in module_names:
        stub = ModuleType(name)
        if name in {"nemo_skills", "nemo_skills.mcp", "nemo_skills.mcp.servers"}:
            stub.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = stub
    sys.modules["httpx"].AsyncClient = object  # type: ignore[attr-defined]
    sys.modules["httpx"].ConnectError = OSError  # type: ignore[attr-defined]
    sys.modules["httpx"].ConnectTimeout = TimeoutError  # type: ignore[attr-defined]
    sys.modules["nemo_skills.mcp.servers.python_tool"].DirectPythonTool = (  # type: ignore[attr-defined]
        SimpleNamespace
    )
    spec = importlib.util.spec_from_file_location(
        "sciprobe_network_blocking", VALIDATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


def test_proc_tcp_listener_parser_decodes_ipv4_and_filters_state(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    proc_tcp = tmp_path / "tcp"
    proc_tcp.write_text(
        "  sl  local_address rem_address   st tx_queue rx_queue\n"
        "   0: 0100007F:1770 00000000:0000 0A 00000000:00000000\n"
        "   1: 0100007F:1771 00000000:0000 01 00000000:00000000\n",
        encoding="utf-8",
    )

    assert validator._proc_tcp_listeners(proc_tcp, socket.AF_INET) == [
        {
            "address": "127.0.0.1",
            "port": 6000,
            "raw": "0100007F:1770",
            "source": str(proc_tcp),
        }
    ]


def test_proc_tcp_listener_parser_exposes_non_loopback_and_ipv6(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    proc_tcp = tmp_path / "tcp"
    proc_tcp6 = tmp_path / "tcp6"
    proc_tcp.write_text(
        "  sl  local_address rem_address   st tx_queue rx_queue\n"
        "   0: 00000000:1771 00000000:0000 0A 00000000:00000000\n",
        encoding="utf-8",
    )
    proc_tcp6.write_text(
        "  sl  local_address rem_address   st tx_queue rx_queue\n"
        "   0: 00000000000000000000000001000000:1770 00000000000000000000000000000000:0000 0A 00000000:00000000\n",
        encoding="utf-8",
    )

    assert (
        validator._proc_tcp_listeners(proc_tcp, socket.AF_INET)[0]["address"]
        == "0.0.0.0"
    )
    assert (
        validator._proc_tcp_listeners(proc_tcp6, socket.AF_INET6)[0]["address"] == "::1"
    )


def test_stateful_proof_is_collected_below_the_tool_output_cap() -> None:
    validator = _load_validator()

    class FakeTool:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object], dict[str, object]]] = []
            self.chunks = iter(
                [
                    {"alpha": True},
                    {"beta": 42},
                    {"gamma": "complete"},
                ]
            )

        async def execute(
            self,
            tool_name: str,
            arguments: dict[str, object],
            extra_args: dict[str, object],
        ) -> str:
            self.calls.append((tool_name, arguments, extra_args))
            return json.dumps(next(self.chunks), sort_keys=True)

    tool = FakeTool()
    proof = asyncio.run(
        validator._collect_stateful_mapping(
            tool,
            request_id="proof-request",
            variable_name="sciprobe_security_proof",
            key_count=3,
        )
    )

    assert proof == {"alpha": True, "beta": 42, "gamma": "complete"}
    assert len(tool.calls) == 3
    assert all(call[0] == "stateful_python_code_exec" for call in tool.calls)
    assert all(call[2] == {"request_id": "proof-request"} for call in tool.calls)
    assert all("sciprobe_security_proof" in str(call[1]["code"]) for call in tool.calls)
