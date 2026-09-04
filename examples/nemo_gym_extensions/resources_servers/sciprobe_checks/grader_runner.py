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
"""Run one exact, provenance-pinned SciProbe checker in a fresh temp."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: The old hardcoded 8s was below what two probes need: one takes 16.6s to 47s and
#: another 9.6s to 23.9s, so they failed closed regardless of the environment.
DEFAULT_CHECKER_TIMEOUT_S = 120.0

#: A checker that cannot run is not a wrong answer. If checks.py fails to import a
#: package or cannot find a binary or a data file, scoring 0 would charge the model for
#: our environment. Under GRPO that probe then becomes permanently unsolvable and drags
#: the gradient every step it appears, while looking exactly like a hard probe. Those
#: cases are reported as infra_error and raised; only a genuine exception from check()
#: on the submitted answer counts as a rejection.
CHECK_RUNNER = """import json, sys
sys.path.insert(0, '.')
INFRA = (ImportError, FileNotFoundError, NotADirectoryError, PermissionError, MemoryError)
def _fmt(exc):
    return '%s: %s' % (type(exc).__name__, exc)
try:
    import checks
except BaseException as exc:
    print(json.dumps({'answer': {'checks': None, 'infra_error': _fmt(exc)}}))
    raise SystemExit(0)
ans = json.load(open('_answer.json'))
try:
    raw = checks.check(ans)
except INFRA as exc:
    out = {'checks': None, 'infra_error': _fmt(exc)}
except OSError as exc:
    out = {'checks': None, 'infra_error': _fmt(exc)}
except Exception as exc:
    out = {'checks': None, 'answer_rejected': _fmt(exc)}
else:
    try:
        out = {'checks': [[str(n), bool(v)] for n, v in raw]}
    except Exception as exc:
        out = {'checks': None, 'bad_return': '%s: %s' % (type(exc).__name__, exc)}
print(json.dumps({'answer': out}))
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _data_tree_sha256(root: Path) -> tuple[str, int, int]:
    """Hash sorted relative paths, sizes, and contents; reject every symlink."""
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(
                f"probe data contains a symlink: {path.relative_to(root)}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(
                f"probe data contains a non-regular entry: {path.relative_to(root)}"
            )
        relative = path.relative_to(root).as_posix()
        contents = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(contents)).encode("ascii"))
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
        files += 1
        total_bytes += len(contents)
    if files == 0:
        raise RuntimeError(f"probe data tree is empty: {root}")
    return digest.hexdigest(), files, total_bytes


def _normalize_submitted(answer: object, depth: int = 4) -> object:
    """Match SciProbe grade_submitted's JSON-string and answer-wrapper handling."""
    for _ in range(depth):
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except (TypeError, json.JSONDecodeError):
                return answer
        elif isinstance(answer, dict) and len(answer) == 1 and "answer" in answer:
            answer = answer["answer"]
        else:
            break
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--checks-sha256", required=True)
    parser.add_argument("--data-tree-sha256", required=True)
    parser.add_argument(
        "--checker-python",
        default=sys.executable,
        help=(
            "interpreter that runs the probe's checks.py; defaults to this process. "
            "Point it at the grading environment when checks.py needs the science stack."
        ),
    )
    parser.add_argument(
        "--checker-timeout",
        type=float,
        default=DEFAULT_CHECKER_TIMEOUT_S,
        help="seconds allowed for one checks.py run (default %(default)s)",
    )
    args = parser.parse_args()

    probe_root = args.probe_root.resolve(strict=True)
    data_root = probe_root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"missing probe data directory: {data_root}")
    checks_path = probe_root / "checks.py"
    if not checks_path.is_file():
        raise FileNotFoundError(f"missing probe checker: {checks_path}")

    checks_sha256 = _sha256(checks_path)
    if checks_sha256 != args.checks_sha256:
        raise RuntimeError(
            f"checks.py hash drift: got {checks_sha256}, expected {args.checks_sha256}"
        )
    data_tree_sha256, data_files, data_bytes = _data_tree_sha256(data_root)
    if data_tree_sha256 != args.data_tree_sha256:
        raise RuntimeError(
            "data tree hash drift: "
            f"got {data_tree_sha256}, expected {args.data_tree_sha256}"
        )

    answer = _normalize_submitted(json.load(sys.stdin))
    with tempfile.TemporaryDirectory(prefix="sciprobe_grade_") as temp_name:
        temp_root = Path(temp_name)
        shutil.copytree(data_root, temp_root / "data")
        shutil.copyfile(checks_path, temp_root / "checks.py")
        (temp_root / "_answer.json").write_text(json.dumps(answer), encoding="utf-8")
        (temp_root / "_run_checks.py").write_text(CHECK_RUNNER, encoding="utf-8")
        completed = subprocess.run(
            [args.checker_python, str(temp_root / "_run_checks.py")],
            cwd=temp_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=args.checker_timeout,
        )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"checks.py process exited {completed.returncode}: {detail[-1000:]}"
        )
    try:
        payload = json.loads(completed.stdout)
        outcome = payload["answer"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("checks.py process returned an invalid envelope") from error

    if outcome.get("infra_error"):
        raise RuntimeError(
            "checks.py could not run in this environment: " + str(outcome["infra_error"])
        )
    if outcome.get("answer_rejected"):
        status = "answer_rejected"
        results = [["answer_schema", False]]
    elif outcome.get("bad_return"):
        raise RuntimeError(
            "check() must return [[name, bool], ...]: " + str(outcome["bad_return"])
        )
    else:
        status = "ok"
        results = outcome.get("checks")
        if not isinstance(results, list) or not results:
            raise RuntimeError("check() returned no checks")
        if not all(
            isinstance(row, list)
            and len(row) == 2
            and isinstance(row[0], str)
            and isinstance(row[1], bool)
            for row in results
        ):
            raise RuntimeError("checker returned an invalid result schema")

    json.dump(
        {
            "status": status,
            "checks": results,
            "checks_sha256": checks_sha256,
            "data_tree_sha256": data_tree_sha256,
            "data_files": data_files,
            "data_bytes": data_bytes,
        },
        sys.stdout,
        sort_keys=True,
    )


if __name__ == "__main__":
    main()
