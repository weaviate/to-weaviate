import json
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROGRESS_INTERVAL_SECONDS = 5.0


class Events:
    def __init__(self, path: Path, clock: Callable[[], float] = time.monotonic) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._clock = clock
        self._lock = threading.Lock()
        self._last_progress: dict[str, float] = {}

    def close(self) -> None:
        """No-op: each line is written through a short-lived file handle."""

    def emit(self, level: str, event: str, **fields: Any) -> None:
        with self._lock:
            self._write(level, event, fields)

    def progress(self, ref: str, copied: int, total: int) -> None:
        with self._lock:
            now = self._clock()
            last = self._last_progress.get(ref)
            if last is not None and now - last < PROGRESS_INTERVAL_SECONDS:
                return
            self._last_progress[ref] = now
            self._write(
                "info", "progress", {"ref": ref, "copied": copied, "total": total}
            )

    def _write(self, level: str, event: str, fields: dict[str, Any]) -> None:
        line = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
            **fields,
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, default=str) + "\n")
