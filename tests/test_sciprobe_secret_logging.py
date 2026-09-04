from __future__ import annotations

import runpy
from pathlib import Path

import pytest

RAY_SUB = Path("ray.sub")
LAUNCHER = Path("examples/launch_sciprobe_lightning_signal_canary.sh")
OUTPUT_VALIDATOR = Path("examples/validate_sciprobe_signal_canary_outputs.py")


def test_ray_launcher_never_dumps_the_environment() -> None:
    source = RAY_SUB.read_text(encoding="utf-8")
    assert "umask 077" in source
    assert "RAY_SUB_DEBUG_ENV" not in source
    assert "env |" not in source
    assert "printenv" not in source


def test_signal_launcher_captures_slurm_output_under_run_root() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "umask 077" in source
    assert '--output="${RUN_ROOT}/slurm-%j.out"' in source
    assert '--error="${RUN_ROOT}/slurm-%j.err"' in source


def test_log_scanner_checks_every_fresh_secret_value(tmp_path: Path) -> None:
    module = runpy.run_path(str(OUTPUT_VALIDATOR))
    scan = module["_validate_secrets_absent_from_logs"]
    secrets = {
        "SCIPROBE_VERIFIER_TOKEN": "a" * 64,
        "SCIPROBE_CAPABILITY_SIGNING_KEY": "b" * 64,
        "SCIPROBE_TRUSTED_INGRESS_TOKEN": "c" * 64,
        "SCIPROBE_POLICY_GENERATION_TOKEN": "d" * 64,
    }
    for directory_name in ("logs", "nemo-gym", "ray"):
        directory = tmp_path / directory_name
        directory.mkdir()
        (directory / "safe.log").write_text("no credentials here\n", encoding="utf-8")
    (tmp_path / "slurm-123.out").write_text("safe stdout\n", encoding="utf-8")

    result = scan(tmp_path, secrets)
    assert result["secret_values_checked"] == sorted(secrets)
    assert result["files"] == 4

    leak_path = tmp_path / "ray" / "ray-head.log"
    for offset, secret in enumerate(secrets.values()):
        leak_path.write_bytes(
            b"x" * (1024 * 1024 - offset) + secret.encode("utf-8") + b"suffix"
        )
        with pytest.raises(AssertionError):
            scan(tmp_path, secrets)
        leak_path.unlink()
