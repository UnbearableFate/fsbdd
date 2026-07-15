from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time
from pathlib import Path
from typing import Any


class StructuredLogger:
    def __init__(
        self,
        path: Path,
        *,
        role: str,
        run_id: str,
        fsync_every: int = 1,
    ) -> None:
        if not isinstance(path, Path):
            raise TypeError("structured log path must be a Path")
        if not isinstance(role, str) or not role:
            raise ValueError("structured log role must be a non-empty string")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("structured log run_id must be a non-empty string")
        if (
            not isinstance(fsync_every, int)
            or isinstance(fsync_every, bool)
            or fsync_every <= 0
        ):
            raise ValueError("fsync_every must be a positive integer")
        self.path = path
        self.role = role
        self.run_id = run_id
        self.fsync_every = fsync_every
        self._emits = 0
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("a", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        reserved = {
            "schema_version",
            "timestamp_utc",
            "monotonic_ns",
            "pid",
            "run_id",
            "role",
            "event",
        }
        collisions = reserved & fields.keys()
        if collisions:
            raise ValueError(
                f"structured log fields use reserved names: {sorted(collisions)}"
            )
        if not isinstance(event, str) or not event:
            raise ValueError("structured log event must be a non-empty string")
        record = {
            "schema_version": 1,
            "timestamp_utc": dt.datetime.now(dt.UTC)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z"),
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "run_id": self.run_id,
            "role": self.role,
            "event": event,
        }
        record.update(fields)
        with self._lock:
            self._stream.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            self._emits += 1
            if self._emits % self.fsync_every == 0:
                self._stream.flush()
                os.fsync(self._stream.fileno())
        return record

    def sync(self) -> None:
        with self._lock:
            self._stream.flush()
            os.fsync(self._stream.fileno())

    def close(self) -> None:
        with self._lock:
            if self._stream.closed:
                return
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
