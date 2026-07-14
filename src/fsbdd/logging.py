from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any


class StructuredLogger:
    def __init__(self, path: Path, *, role: str, run_id: str) -> None:
        self.path = path
        self.role = role
        self.run_id = run_id
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {
            "schema_version": 1,
            "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "run_id": self.run_id,
            "role": self.role,
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return record
