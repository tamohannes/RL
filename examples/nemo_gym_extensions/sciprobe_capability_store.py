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

"""Private cross-process ledger for fail-closed, at-most-once verification."""

from __future__ import annotations

import os
import sqlite3
import stat
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ClaimState = Literal[
    "owner",
    "pending",
    "completed",
    "failed",
    "mismatch",
    "missing",
]


@dataclass(frozen=True)
class CapabilityResult:
    state: ClaimState
    reward: float | None = None


class CapabilityResultStore:
    """SQLite-backed claim/result store shared by all verifier processes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._prepare_private_file()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS capability_results (
                    jti TEXT PRIMARY KEY,
                    token_sha256 TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (
                        state IN ('pending', 'completed', 'failed')
                    ),
                    reward INTEGER CHECK (reward IN (0, 1)),
                    expires_at INTEGER NOT NULL,
                    CHECK (
                        (state = 'completed' AND reward IS NOT NULL)
                        OR (state != 'completed' AND reward IS NULL)
                    )
                )
                """
            )
            connection.execute("PRAGMA user_version = 1")
        self._validate_private_file()

    def _prepare_private_file(self) -> None:
        if not self.path.is_absolute():
            raise ValueError("capability store path must be absolute")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        parent_stat = self.path.parent.stat()
        if not stat.S_ISDIR(parent_stat.st_mode) or parent_stat.st_mode & 0o077:
            raise PermissionError("capability store directory must be private")

        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except FileExistsError:
            self._validate_private_file()
        else:
            try:
                os.fchmod(descriptor, 0o600)
            finally:
                os.close(descriptor)

    def _validate_private_file(self) -> None:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags)
        try:
            file_stat = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise PermissionError("capability store must be a regular file")
        if stat.S_IMODE(file_stat.st_mode) != 0o600:
            raise PermissionError("capability store must have mode 0600")
        if file_stat.st_uid != os.geteuid() or file_stat.st_nlink != 1:
            raise PermissionError("capability store must be private to this user")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=5.0,
            isolation_level=None,
        )
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA trusted_schema = OFF")
        return connection

    @staticmethod
    def _validate_key(value: str, name: str, *, minimum: int, maximum: int) -> None:
        if not isinstance(value, str) or not minimum <= len(value) <= maximum:
            raise ValueError(f"{name} is invalid")

    def claim(
        self,
        *,
        jti: str,
        token_sha256: str,
        request_sha256: str,
        expires_at: int,
        now: int,
    ) -> CapabilityResult:
        """Atomically become owner or observe an existing immutable claim."""
        self._validate_key(jti, "jti", minimum=16, maximum=128)
        self._validate_key(token_sha256, "token_sha256", minimum=64, maximum=64)
        self._validate_key(request_sha256, "request_sha256", minimum=64, maximum=64)
        if isinstance(expires_at, bool) or not isinstance(expires_at, int):
            raise ValueError("expires_at is invalid")
        if isinstance(now, bool) or not isinstance(now, int) or expires_at <= now:
            raise ValueError("claim is expired")

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "DELETE FROM capability_results WHERE expires_at <= ?",
                    (now,),
                )
                row = connection.execute(
                    """
                    SELECT token_sha256, request_sha256, state, reward
                    FROM capability_results
                    WHERE jti = ?
                    """,
                    (jti,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO capability_results (
                            jti, token_sha256, request_sha256,
                            state, reward, expires_at
                        ) VALUES (?, ?, ?, 'pending', NULL, ?)
                        """,
                        (jti, token_sha256, request_sha256, expires_at),
                    )
                    result = CapabilityResult("owner")
                elif row[0] != token_sha256 or row[1] != request_sha256:
                    result = CapabilityResult("mismatch")
                elif row[2] == "completed":
                    result = CapabilityResult("completed", float(row[3]))
                else:
                    result = CapabilityResult(row[2])
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise

    def lookup(
        self,
        *,
        jti: str,
        token_sha256: str,
        request_sha256: str,
        now: int,
    ) -> CapabilityResult:
        """Read an existing claim without taking ownership."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT token_sha256, request_sha256, state, reward, expires_at
                FROM capability_results
                WHERE jti = ?
                """,
                (jti,),
            ).fetchone()
        if row is None or row[4] <= now:
            return CapabilityResult("missing")
        if row[0] != token_sha256 or row[1] != request_sha256:
            return CapabilityResult("mismatch")
        if row[2] == "completed":
            return CapabilityResult("completed", float(row[3]))
        return CapabilityResult(row[2])

    def complete(
        self,
        *,
        jti: str,
        token_sha256: str,
        request_sha256: str,
        reward: float,
    ) -> bool:
        """Cache only a binary scalar for the owner of a pending claim."""
        if isinstance(reward, bool) or reward not in (0, 1, 0.0, 1.0):
            raise ValueError("reward must be binary")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    UPDATE capability_results
                    SET state = 'completed', reward = ?
                    WHERE jti = ?
                      AND token_sha256 = ?
                      AND request_sha256 = ?
                      AND state = 'pending'
                    """,
                    (int(reward), jti, token_sha256, request_sha256),
                )
                connection.commit()
                return cursor.rowcount == 1
            except BaseException:
                connection.rollback()
                raise

    def fail(
        self,
        *,
        jti: str,
        token_sha256: str,
        request_sha256: str,
    ) -> bool:
        """Permanently close a pending claim without storing failure details."""
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    UPDATE capability_results
                    SET state = 'failed', reward = NULL
                    WHERE jti = ?
                      AND token_sha256 = ?
                      AND request_sha256 = ?
                      AND state = 'pending'
                    """,
                    (jti, token_sha256, request_sha256),
                )
                connection.commit()
                return cursor.rowcount == 1
            except BaseException:
                connection.rollback()
                raise

    def cleanup_expired(self, *, now: int) -> int:
        """Delete claims only after their signed token lifetime ends."""
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    "DELETE FROM capability_results WHERE expires_at <= ?",
                    (now,),
                )
                connection.commit()
                return cursor.rowcount
            except BaseException:
                connection.rollback()
                raise
