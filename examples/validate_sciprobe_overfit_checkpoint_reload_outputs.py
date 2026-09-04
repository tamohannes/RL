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

"""Validate a fresh-process Lightning checkpoint-reload rollout proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import os
from pathlib import Path
from typing import Any

import validate_sciprobe_overfit_canary_outputs as canary

EXPECTED_SOURCE_RUN_PREFIX = "lightning-overfit-canary-"
EXPECTED_RELOAD_RUN_PREFIX = "lightning-overfit-checkpoint-reload-r"
EXPECTED_STEP = 6
EXPECTED_VALIDATION_ROLLOUTS = 32


#: Some clusters expose one filesystem under two mount roots, so a path compared as a
#: string can differ while naming the same storage. Set STORAGE_ALIASES to a
#: colon-separated list of those roots and STORAGE_CANONICAL_ROOT to the one they
#: normalize to. Both default to empty, which disables aliasing entirely; the mount
#: layout is a property of the cluster, not of this repository.
STORAGE_ALIASES = tuple(
    Path(root)
    for root in os.environ.get("STORAGE_ALIASES", "").split(":")
    if root
)
STORAGE_CANONICAL_ROOT = os.environ.get("STORAGE_CANONICAL_ROOT", "")


def _canonical_checkpoint_root(path: str) -> str:
    """Normalize configured mount aliases that name the same storage."""
    checkpoint_root = Path(path)
    assert checkpoint_root.is_absolute(), (
        f"checkpoint root must be absolute: {checkpoint_root}"
    )
    if not STORAGE_CANONICAL_ROOT:
        return checkpoint_root.as_posix()
    for alias in STORAGE_ALIASES:
        try:
            relative = checkpoint_root.relative_to(alias)
        except ValueError:
            continue
        return (Path(STORAGE_CANONICAL_ROOT) / relative).as_posix()
    return checkpoint_root.as_posix()


def _assert_scoped_directory(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    assert resolved.is_dir(), f"{label} is not a directory: {resolved}"
    assert resolved.is_relative_to(root), f"{label} escapes reload root: {resolved}"
    return resolved


def _assert_private_regular_file(path: Path, label: str) -> Path:
    file_stat = path.lstat()
    assert not stat.S_ISLNK(file_stat.st_mode), f"{label} is a symlink: {path}"
    assert stat.S_ISREG(file_stat.st_mode), f"{label} is not regular: {path}"
    assert stat.S_IMODE(file_stat.st_mode) == 0o600, (
        f"{label} must be mode 0600: {path}"
    )
    assert file_stat.st_nlink == 1, f"{label} must have one hard link: {path}"
    return path.resolve(strict=True)


def _step_directories(checkpoint_root: Path) -> dict[int, Path]:
    steps: dict[int, Path] = {}
    for path in checkpoint_root.iterdir():
        if not path.is_dir() or not path.name.startswith("step_"):
            continue
        step_text = path.name.removeprefix("step_")
        if not step_text.isdigit():
            continue
        step = int(step_text)
        assert step not in steps, f"duplicate checkpoint step {step}"
        steps[step] = path
    return steps


def _checkpoint_manifest(checkpoint_root: Path) -> dict[str, Any]:
    checkpoint_root = checkpoint_root.resolve(strict=True)
    directories: list[str] = []
    files: list[dict[str, Any]] = []
    for path in sorted(
        checkpoint_root.rglob("*"),
        key=lambda candidate: candidate.relative_to(checkpoint_root).as_posix(),
    ):
        path_stat = path.lstat()
        relative_path = path.relative_to(checkpoint_root).as_posix()
        assert not stat.S_ISLNK(path_stat.st_mode), (
            f"source checkpoint contains a symlink: {relative_path}"
        )
        if stat.S_ISDIR(path_stat.st_mode):
            directories.append(relative_path)
            continue
        assert stat.S_ISREG(path_stat.st_mode), (
            f"source checkpoint contains a non-regular entry: {relative_path}"
        )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        files.append(
            {
                "path": relative_path,
                "bytes": path_stat.st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return {
        "version": 1,
        "checkpoint_root": str(checkpoint_root),
        "step_inventory": sorted(_step_directories(checkpoint_root)),
        "directories": directories,
        "files": files,
    }


def _validate_checkpoint_immutability(
    *,
    checkpoint_root: Path,
    manifest_path: Path,
    reload_root: Path,
) -> dict[str, Any]:
    manifest_path = _assert_private_regular_file(
        manifest_path, "source checkpoint manifest"
    )
    assert manifest_path.is_relative_to(reload_root), (
        "source checkpoint manifest is outside the fresh reload root"
    )
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(recorded, dict), "source checkpoint manifest is not an object"
    assert set(recorded) == {
        "version",
        "checkpoint_root",
        "step_inventory",
        "directories",
        "files",
    }, f"source checkpoint manifest has unexpected fields: {sorted(recorded)}"
    actual = _checkpoint_manifest(checkpoint_root)
    assert recorded.get("version") == 1
    recorded_checkpoint_root = recorded.get("checkpoint_root")
    assert isinstance(recorded_checkpoint_root, str), (
        "source checkpoint manifest checkpoint_root is not a string"
    )
    recorded_root_identity = _canonical_checkpoint_root(recorded_checkpoint_root)
    actual_root_identity = _canonical_checkpoint_root(actual["checkpoint_root"])
    assert recorded_root_identity == actual_root_identity, (
        "source checkpoint root differs from the pre-run manifest after known "
        "cluster path-alias normalization: "
        f"before={recorded_checkpoint_root} after={actual['checkpoint_root']}"
    )
    assert recorded.get("step_inventory") == actual["step_inventory"], (
        "source checkpoint step inventory changed during reload: "
        f"before={recorded.get('step_inventory')} "
        f"after={actual['step_inventory']}"
    )
    assert recorded.get("directories") == actual["directories"], (
        "source checkpoint directory inventory changed during reload"
    )

    recorded_files = recorded.get("files")
    assert isinstance(recorded_files, list), (
        "source checkpoint manifest files field is not a list"
    )
    recorded_by_path = {
        item["path"]: item
        for item in recorded_files
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    actual_by_path = {item["path"]: item for item in actual["files"]}
    assert len(recorded_by_path) == len(recorded_files), (
        "source checkpoint manifest has malformed or duplicate file entries"
    )
    assert set(recorded_by_path) == set(actual_by_path), (
        "source checkpoint file inventory changed during reload"
    )
    changed_files = sorted(
        path
        for path in recorded_by_path
        if recorded_by_path[path] != actual_by_path[path]
    )
    assert not changed_files, (
        f"source checkpoint file size or SHA256 changed during reload: {changed_files}"
    )
    return {
        "manifest": str(manifest_path),
        "manifest_checkpoint_root": recorded_checkpoint_root,
        "validated_checkpoint_root": actual["checkpoint_root"],
        "canonical_checkpoint_root": actual_root_identity,
        "path_alias_normalized": recorded_checkpoint_root != actual["checkpoint_root"],
        "read_only_mount_required_by_launcher": True,
        "step_inventory": actual["step_inventory"],
        "directories": len(actual["directories"]),
        "files": len(actual["files"]),
        "bytes": sum(item["bytes"] for item in actual["files"]),
        "sha256_and_size_unchanged": True,
    }


def _validate_source_checkpoint(
    source_root: Path, expected_step: int
) -> dict[str, Any]:
    checkpoint_root = source_root / "checkpoints"
    assert checkpoint_root.is_dir(), f"missing source checkpoints: {checkpoint_root}"
    step_directories = _step_directories(checkpoint_root)
    assert step_directories, f"no source step checkpoints: {checkpoint_root}"
    assert max(step_directories) == expected_step, (
        f"latest source checkpoint is step {max(step_directories)}, "
        f"expected step {expected_step}"
    )

    training_info_path = step_directories[expected_step] / "training_info.json"
    training_info = json.loads(training_info_path.read_text(encoding="utf-8"))
    assert isinstance(training_info, dict), (
        f"source training info is not an object: {training_info_path}"
    )
    total_steps = canary._integer(
        training_info.get("total_steps"), "source training_info.total_steps"
    )
    assert total_steps == expected_step, (
        f"source training_info.total_steps is {total_steps}, expected {expected_step}"
    )

    checkpoint = canary._validate_checkpoint(source_root, expected_step)
    return {
        "checkpoint_root": str(checkpoint_root.resolve()),
        "step_inventory": sorted(step_directories),
        "training_info": str(training_info_path.resolve()),
        "total_steps": total_steps,
        "safetensor_files": len(checkpoint["safetensors"]),
        "safetensor_bytes": sum(item["bytes"] for item in checkpoint["safetensors"]),
    }


def _validate_runtime_isolation(source_root: Path, reload_root: Path) -> dict[str, Any]:
    source_runtime = source_root / "runtime" / "train.jsonl"
    source_capability_store = source_root / "runtime" / "capability-results.sqlite3"
    reload_runtime_dir = _assert_scoped_directory(
        reload_root / "runtime", reload_root, "reload runtime directory"
    )
    reload_runtime = _assert_private_regular_file(
        reload_runtime_dir / "train.jsonl", "reload runtime dataset"
    )
    reload_capability_store = _assert_private_regular_file(
        reload_runtime_dir / "capability-results.sqlite3",
        "reload capability database",
    )
    assert reload_runtime != source_runtime.resolve(strict=True), (
        "reload runtime dataset reuses the source path"
    )
    assert reload_capability_store != source_capability_store.resolve(strict=True), (
        "reload capability database reuses the source path"
    )
    assert not reload_runtime.samefile(source_runtime), (
        "reload runtime dataset reuses the source file"
    )
    assert not reload_capability_store.samefile(source_capability_store), (
        "reload capability database reuses the source file"
    )

    source_rows = canary._load_jsonl(source_runtime)
    rows = canary._load_jsonl(reload_runtime)
    assert len(source_rows) == 1, (
        f"source runtime dataset has {len(source_rows)} rows, expected 1"
    )
    assert len(rows) == 1, f"reload runtime dataset has {len(rows)} rows, expected 1"
    assert rows == source_rows, "reload runtime dataset differs from the source prompt"
    row = rows[0]
    assert row.get("id") == canary.PROBE_ID
    assert row.get("probe_id") == canary.PROBE_ID
    assert not {
        "_sciprobe_verifier_capability",
        "sciprobe_capability",
    }.intersection(row), "reload runtime dataset persists a verifier capability"

    return {
        "runtime_dataset": str(reload_runtime),
        "capability_database": str(reload_capability_store),
        "rows": len(rows),
        "probe_id": row["probe_id"],
        "matches_source_prompt": True,
    }


def _validate_tensorboard(
    log_root: Path, expected_step: int, expected_pass_at_1: float
) -> dict[str, Any]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_paths = sorted(log_root.rglob("events.out.tfevents.*"))
    assert event_paths, f"no reload TensorBoard event files under {log_root}"
    scalar_tags: set[str] = set()
    validation_values: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for path in event_paths:
        try:
            accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
            accumulator.Reload()
            tags = set(accumulator.Tags().get("scalars", []))
            scalar_tags.update(tags)
            if "validation/accuracy" not in tags:
                continue
            for event in accumulator.Scalars("validation/accuracy"):
                validation_values.append(
                    {
                        "path": str(path),
                        "step": int(event.step),
                        "value": float(event.value),
                    }
                )
        except Exception as error:
            errors[str(path)] = f"{type(error).__name__}: {error}"

    train_tags = sorted(tag for tag in scalar_tags if tag.startswith("train/"))
    assert not train_tags, f"reload logger contains train metrics: {train_tags}"
    matching = [
        item
        for item in validation_values
        if item["step"] == expected_step
        and math.isclose(item["value"], expected_pass_at_1, rel_tol=0.0, abs_tol=1e-6)
    ]
    assert matching, (
        "reload TensorBoard validation/accuracy does not match the structured "
        f"step-{expected_step} pass@1; values={validation_values}, errors={errors}"
    )
    return {
        "event_files": [str(path) for path in event_paths],
        "scalar_tags": sorted(scalar_tags),
        "validation_accuracy": matching,
    }


def _validate_validation_outputs(
    *,
    log_root: Path,
    expected_step: int,
    expected_rollouts: int,
    require_no_train_outputs: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    full_results, table_paths = canary._load_full_result_tables(log_root)
    if require_no_train_outputs:
        assert not full_results["train"], (
            f"reload logger contains train Full-result tables: {table_paths['train']}"
        )
    validation_steps = set(full_results["validation"])
    if require_no_train_outputs:
        assert validation_steps == {expected_step}, (
            "reload validation Full-result steps are "
            f"{sorted(validation_steps)}, expected [{expected_step}]"
        )
    else:
        assert expected_step in validation_steps, (
            f"source validation has no Full-result table for step {expected_step}; "
            f"found {sorted(validation_steps)}"
        )
    assert len(full_results["validation"][expected_step]) == 1, (
        f"expected one reload step-{expected_step} Full-result table, found "
        f"{len(full_results['validation'][expected_step])}"
    )
    assert len(table_paths["validation"][expected_step]) == 1

    train_data_paths = sorted(
        path for path in log_root.rglob("train_data_step*") if path.is_file()
    )
    if require_no_train_outputs:
        assert not train_data_paths, (
            f"reload logger contains train_data_step files: {train_data_paths}"
        )
    validation_paths = canary._indexed_jsonl_paths(log_root, "val_data_step")
    validation_steps = set(validation_paths)
    if require_no_train_outputs:
        assert validation_steps == {expected_step}, (
            f"reload validation JSONL steps are {sorted(validation_paths)}, "
            f"expected [{expected_step}]"
        )
    else:
        assert expected_step in validation_steps, (
            f"source validation has no JSONL for step {expected_step}; "
            f"found {sorted(validation_paths)}"
        )
    all_validation_paths = sorted(
        path for path in log_root.rglob("val_data_step*") if path.is_file()
    )
    if require_no_train_outputs:
        assert all_validation_paths == [validation_paths[expected_step]], (
            f"unexpected reload validation data files: {all_validation_paths}"
        )

    results = full_results["validation"][expected_step][0]
    summary = canary._validate_validation_step(
        step=expected_step,
        results=results,
        validation_path=validation_paths[expected_step],
        expected_rollouts=expected_rollouts,
    )
    audited = [
        canary._validate_full_result(
            result, f"validation step {expected_step} rollout {position}"
        )
        for position, result in enumerate(results)
    ]
    assert len(audited) == expected_rollouts
    audit = {
        "rollouts": len(audited),
        "turns_min": min(item["audit"]["turns"] for item in audited),
        "turns_max": max(item["audit"]["turns"] for item in audited),
        "tokens_min": min(item["audit"]["tokens"] for item in audited),
        "tokens_max": max(item["audit"]["tokens"] for item in audited),
        "generation_tokens": sum(
            item["audit"]["generation_tokens"] for item in audited
        ),
    }
    return summary, {
        "full_result_table": str(table_paths["validation"][expected_step][0]),
        "validation_jsonl": str(validation_paths[expected_step]),
        "raw_token_audits": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--reload-run-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-run-id", required=True)
    parser.add_argument("--expected-step", type=int, default=EXPECTED_STEP)
    parser.add_argument(
        "--expected-validation-rollouts",
        type=int,
        default=EXPECTED_VALIDATION_ROLLOUTS,
    )
    args = parser.parse_args()

    assert args.expected_step == EXPECTED_STEP, (
        f"checkpoint-reload proof requires step {EXPECTED_STEP}"
    )
    assert args.expected_validation_rollouts == EXPECTED_VALIDATION_ROLLOUTS, (
        "checkpoint-reload proof requires exactly "
        f"{EXPECTED_VALIDATION_ROLLOUTS} validation rollouts"
    )
    source_root = args.source_run_root.resolve(strict=True)
    reload_root = args.reload_run_root.resolve(strict=True)
    assert source_root.is_dir() and reload_root.is_dir()
    source_run_suffix = args.expected_source_run_id.removeprefix("r")
    assert (
        args.expected_source_run_id.startswith("r") and source_run_suffix.isdigit()
    ), (
        "expected source run id must match r<positive-integer>, found "
        f"{args.expected_source_run_id!r}"
    )
    assert int(source_run_suffix) > 0, (
        "expected source run id must be positive, found "
        f"{args.expected_source_run_id!r}"
    )
    expected_source_run_name = (
        f"{EXPECTED_SOURCE_RUN_PREFIX}{args.expected_source_run_id}"
    )
    assert source_root.name == expected_source_run_name, (
        f"source run must be {expected_source_run_name}, found {source_root.name}"
    )
    reload_id = reload_root.name.removeprefix(EXPECTED_RELOAD_RUN_PREFIX)
    assert reload_root.name.startswith(EXPECTED_RELOAD_RUN_PREFIX), (
        f"reload run name must start with {EXPECTED_RELOAD_RUN_PREFIX}"
    )
    assert reload_id.isdigit() and int(reload_id) > 0, (
        f"reload run suffix must be a positive integer, found {reload_id!r}"
    )
    assert source_root != reload_root, "reload run root reuses the source run root"
    assert source_root.parent == reload_root.parent, (
        "source and reload runs must be siblings under the same runs directory"
    )
    assert source_root.parent.name == "runs", "source run is not under runs/"

    source_log_root = (source_root / "logs").resolve(strict=True)
    reload_log_root = _assert_scoped_directory(
        reload_root / "logs", reload_root, "reload logger root"
    )
    assert source_log_root != reload_log_root, (
        "reload validation reuses the source logger root"
    )
    assert not reload_log_root.is_relative_to(source_root), (
        "reload logger is nested under the source run"
    )
    ray_log_root = _assert_scoped_directory(
        reload_root / "ray", reload_root, "reload Ray log root"
    )
    gym_log_root = _assert_scoped_directory(
        reload_root / "nemo-gym", reload_root, "reload Gym log root"
    )
    assert not (reload_root / "checkpoints").exists(), (
        "reload run unexpectedly has a local checkpoint directory"
    )

    source_checkpoint = _validate_source_checkpoint(source_root, args.expected_step)
    checkpoint_immutability = _validate_checkpoint_immutability(
        checkpoint_root=source_root / "checkpoints",
        manifest_path=args.source_checkpoint_manifest,
        reload_root=reload_root,
    )
    assert (
        checkpoint_immutability["step_inventory"] == source_checkpoint["step_inventory"]
    )
    runtime = _validate_runtime_isolation(source_root, reload_root)
    source_step0_validation, source_step0_artifacts = _validate_validation_outputs(
        log_root=source_log_root,
        expected_step=0,
        expected_rollouts=args.expected_validation_rollouts,
        require_no_train_outputs=False,
    )
    source_step6_validation, source_step6_artifacts = _validate_validation_outputs(
        log_root=source_log_root,
        expected_step=args.expected_step,
        expected_rollouts=args.expected_validation_rollouts,
        require_no_train_outputs=False,
    )
    reload_validation, reload_artifacts = _validate_validation_outputs(
        log_root=reload_log_root,
        expected_step=args.expected_step,
        expected_rollouts=args.expected_validation_rollouts,
        require_no_train_outputs=True,
    )

    source_step0_pass_at_1 = float(source_step0_validation["reward_mean"])
    source_step6_pass_at_1 = float(source_step6_validation["reward_mean"])
    reload_pass_at_1 = float(reload_validation["reward_mean"])
    assert source_step6_pass_at_1 > source_step0_pass_at_1, (
        "source step-6 pass@1 did not exceed source step-0 pass@1: "
        f"{source_step6_pass_at_1} <= {source_step0_pass_at_1}"
    )
    assert reload_pass_at_1 > source_step0_pass_at_1, (
        "fresh checkpoint-reload pass@1 did not exceed source step-0 pass@1: "
        f"{reload_pass_at_1} <= {source_step0_pass_at_1}"
    )
    # These are independent 32-sample binomial estimates. At the maximum
    # variance p=0.5, two standard errors for their difference is exactly
    # 0.25. Larger drift is inconsistent with ordinary sampling noise for
    # this smoke proof and should fail closed.
    reload_source_tolerance = 2.0 * math.sqrt(
        2.0 * 0.25 / args.expected_validation_rollouts
    )
    reload_source_delta = abs(reload_pass_at_1 - source_step6_pass_at_1)
    assert reload_source_delta <= reload_source_tolerance, (
        "fresh reload step-6 pass@1 differs too much from source step-6 "
        f"pass@1: delta={reload_source_delta}, "
        f"two-standard-error tolerance={reload_source_tolerance}"
    )
    tensorboard = _validate_tensorboard(
        reload_log_root, args.expected_step, reload_pass_at_1
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "proof": "fresh-process checkpoint reload, validation only",
                "metric_provenance": (
                    "execution-grounded deterministic binary reward; no LLM judge"
                ),
                "source_run_root": str(source_root),
                "reload_run_root": str(reload_root),
                "source_checkpoint": source_checkpoint,
                "checkpoint_immutability": checkpoint_immutability,
                "isolation": {
                    "source_logger_root": str(source_log_root),
                    "reload_logger_root": str(reload_log_root),
                    "reload_ray_log_root": str(ray_log_root),
                    "reload_gym_log_root": str(gym_log_root),
                    **runtime,
                },
                "source_step0": {
                    "pass_at_1": source_step0_pass_at_1,
                    "rollouts": args.expected_validation_rollouts,
                    "temperature": 1.3,
                    "summary": source_step0_validation,
                    "artifacts": source_step0_artifacts,
                },
                "source_step6": {
                    "pass_at_1": source_step6_pass_at_1,
                    "rollouts": args.expected_validation_rollouts,
                    "temperature": 1.3,
                    "summary": source_step6_validation,
                    "artifacts": source_step6_artifacts,
                },
                "reload_step6": {
                    "pass_at_1": reload_pass_at_1,
                    "rollouts": args.expected_validation_rollouts,
                    "temperature": 1.3,
                    "summary": reload_validation,
                    "artifacts": reload_artifacts,
                },
                "source_pass_at_1_shift": (
                    source_step6_pass_at_1 - source_step0_pass_at_1
                ),
                "reload_pass_at_1_shift_from_source_step0": (
                    reload_pass_at_1 - source_step0_pass_at_1
                ),
                "reload_source_step6_comparison": {
                    "absolute_delta": reload_source_delta,
                    "two_standard_error_tolerance": reload_source_tolerance,
                    "within_tolerance": True,
                },
                "optimizer_steps": 0,
                "optimizer_evidence": (
                    "validation-only runner branch does not invoke grpo_train; "
                    "reload logs contain no train data or train metrics"
                ),
                "no_train_data_step_files": True,
                "tensorboard": tensorboard,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
