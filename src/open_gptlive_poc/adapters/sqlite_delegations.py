"""Same-host callback queue shared by local server workers."""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


_WORKER_LEASE_SECONDS = 30


class SQLiteDelegations:
    """Persist delegation ownership and relay callback results between workers."""

    def __init__(self, database_path: str, *, session_ttl_seconds: int = 3600) -> None:
        self.database_path = database_path
        self.session_ttl_seconds = session_ttl_seconds
        self.worker_id = uuid4().hex

    def initialize(self) -> None:
        # Refresh after startup as app factories may have been imported pre-fork.
        self.worker_id = f"{os.getpid()}-{uuid4().hex}"
        path = Path(self.database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    heartbeat REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS delegations (
                    delegation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    worker_id TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('active', 'pending'))
                );
                CREATE TABLE IF NOT EXISTS callbacks (
                    callback_id TEXT PRIMARY KEY,
                    delegation_id TEXT NOT NULL UNIQUE,
                    worker_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'accepted', 'rejected')),
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS callbacks_worker_status
                    ON callbacks(worker_id, status);
                """
            )
        try:
            path.chmod(0o600)
        except OSError:
            pass
        self.heartbeat()

    def register(self, delegation_id: str, session_id: str) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO workers(worker_id, heartbeat) VALUES (?, ?) "
                "ON CONFLICT(worker_id) DO UPDATE SET heartbeat=excluded.heartbeat",
                (self.worker_id, now),
            )
            connection.execute(
                "INSERT INTO delegations VALUES (?, ?, ?, ?, 'active')",
                (delegation_id, session_id, self.worker_id, now + self.session_ttl_seconds),
            )

    def enqueue(self, delegation_id: str, content: str) -> str | None:
        now = time.time()
        callback_id = uuid4().hex
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._prune(connection, now)
            owner = connection.execute(
                "SELECT worker_id, status FROM delegations WHERE delegation_id=?",
                (delegation_id,),
            ).fetchone()
            if owner is None:
                return None
            if owner[1] == "pending":
                existing = connection.execute(
                    "SELECT callback_id, content FROM callbacks "
                    "WHERE delegation_id=? AND status='pending'",
                    (delegation_id,),
                ).fetchone()
                return existing[0] if existing is not None and existing[1] == content else None
            connection.execute(
                "UPDATE delegations SET status='pending' WHERE delegation_id=?",
                (delegation_id,),
            )
            connection.execute(
                "INSERT INTO callbacks VALUES (?, ?, ?, ?, 'pending', ?)",
                (callback_id, delegation_id, owner[0], content, now),
            )
        return callback_id

    def unregister_delegation(self, delegation_id: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE callbacks SET status='rejected' WHERE delegation_id=? AND worker_id=? AND status='pending'",
                (delegation_id, self.worker_id),
            )
            connection.execute(
                "DELETE FROM delegations WHERE delegation_id=? AND worker_id=?",
                (delegation_id, self.worker_id),
            )

    def pending(self) -> list[tuple[str, str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT callback_id, delegation_id, content FROM callbacks "
                "WHERE worker_id=? AND status='pending'",
                (self.worker_id,),
            )
            return list(rows)

    def complete(self, callback_id: str, accepted: bool) -> None:
        status = "accepted" if accepted else "rejected"
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT delegation_id FROM callbacks WHERE callback_id=? AND worker_id=? AND status='pending'",
                (callback_id, self.worker_id),
            ).fetchone()
            if row is None:
                return
            connection.execute("UPDATE callbacks SET status=? WHERE callback_id=?", (status, callback_id))
            connection.execute("DELETE FROM delegations WHERE delegation_id=?", (row[0],))

    def result(self, callback_id: str) -> bool | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM callbacks WHERE callback_id=?", (callback_id,)
            ).fetchone()
            if row is None or row[0] == "pending":
                return None
        return row[0] == "accepted"

    def unregister_session(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            ids = connection.execute(
                "SELECT delegation_id FROM delegations WHERE session_id=? AND worker_id=?",
                (session_id, self.worker_id),
            ).fetchall()
            for (delegation_id,) in ids:
                connection.execute(
                    "UPDATE callbacks SET status='rejected' WHERE delegation_id=? AND status='pending'",
                    (delegation_id,),
                )
                connection.execute("DELETE FROM delegations WHERE delegation_id=?", (delegation_id,))

    def heartbeat(self) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = time.time()
            self._prune(connection, now)
            connection.execute(
                "INSERT INTO workers(worker_id, heartbeat) VALUES (?, ?) "
                "ON CONFLICT(worker_id) DO UPDATE SET heartbeat=excluded.heartbeat",
                (self.worker_id, now),
            )

    def close_worker(self) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE callbacks SET status='rejected' WHERE worker_id=? AND status='pending'",
                (self.worker_id,),
            )
            connection.execute("DELETE FROM delegations WHERE worker_id=?", (self.worker_id,))
            connection.execute("DELETE FROM workers WHERE worker_id=?", (self.worker_id,))

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=2.0)
        connection.execute("PRAGMA busy_timeout=2000")
        try:
            yield connection
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _prune(connection: sqlite3.Connection, now: float) -> None:
        stale = "expires_at <= ? OR worker_id NOT IN " \
            "(SELECT worker_id FROM workers WHERE heartbeat > ?)"
        connection.execute(
            "UPDATE callbacks SET status='rejected' WHERE status='pending' AND delegation_id IN "
            f"(SELECT delegation_id FROM delegations WHERE {stale})",
            (now, now - _WORKER_LEASE_SECONDS),
        )
        connection.execute(f"DELETE FROM delegations WHERE {stale}", (now, now - _WORKER_LEASE_SECONDS))
        connection.execute("DELETE FROM workers WHERE heartbeat <= ?", (now - _WORKER_LEASE_SECONDS,))
        connection.execute(
            "UPDATE callbacks SET status='rejected' WHERE status='pending' AND created_at < ?",
            (now - 300,),
        )
        connection.execute(
            "DELETE FROM delegations WHERE delegation_id IN "
            "(SELECT delegation_id FROM callbacks WHERE created_at < ?)",
            (now - 300,),
        )
        connection.execute("DELETE FROM callbacks WHERE created_at < ?", (now - 300,))
