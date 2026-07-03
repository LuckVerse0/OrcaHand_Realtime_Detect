from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_LOG_FLUSH_EVERY = 30
DEFAULT_LOG_FLUSH_INTERVAL_S = 1.0


class SessionLogger:
    def __init__(
        self,
        log_dir: str | Path,
        stem: str,
        *,
        flush_every: int = DEFAULT_LOG_FLUSH_EVERY,
        flush_interval_s: float = DEFAULT_LOG_FLUSH_INTERVAL_S,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"{stem}.csv"
        self.jsonl_path = self.log_dir / f"{stem}.jsonl"
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._jsonl_file = self.jsonl_path.open("w", encoding="utf-8")
        self._flush_every = max(1, int(flush_every))
        self._flush_interval_s = max(0.0, float(flush_interval_s))
        self._pending_rows = 0
        self._last_flush = time.monotonic()
        self._fieldnames = [
            "timestamp",
            "state",
            "accepted",
            "reasons",
            "joints",
            "motor_positions",
        ]
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
        self._writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        normalized = {
            key: self._serialize(row.get(key, "")) for key in self._fieldnames
        }
        self._writer.writerow(normalized)
        self._jsonl_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self._pending_rows += 1
        if self._should_flush():
            self.flush()

    def flush(self) -> None:
        self._csv_file.flush()
        self._jsonl_file.flush()
        self._pending_rows = 0
        self._last_flush = time.monotonic()

    def close(self) -> None:
        self.flush()
        self._csv_file.close()
        self._jsonl_file.close()

    def _should_flush(self) -> bool:
        return self._pending_rows >= self._flush_every or (
            time.monotonic() - self._last_flush
        ) >= self._flush_interval_s

    @staticmethod
    def _serialize(value: Any) -> str:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value)
