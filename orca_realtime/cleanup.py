from __future__ import annotations

from pathlib import Path


GENERATED_LOG_PATTERNS = ("orcahand_*.csv", "orcahand_*.jsonl")


def cleanup_generated_logs(log_dir: str | Path = "logs") -> list[Path]:
    directory = Path(log_dir).resolve()
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory")

    removed: list[Path] = []
    for pattern in GENERATED_LOG_PATTERNS:
        for path in directory.glob(pattern):
            if path.is_file():
                path.unlink()
                removed.append(path)
    return sorted(removed)
