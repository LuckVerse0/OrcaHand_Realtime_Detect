from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import yaml


VISUAL_CALIBRATION_VERSION = 1
PROFILE_SUFFIX = ".yaml"


@dataclass(frozen=True)
class VisualCalibrationProfile:
    name: str
    path: Path
    created_at: str


def sanitize_profile_name(name: str) -> str:
    cleaned = str(name).strip()
    if cleaned.endswith((".yaml", ".yml")):
        cleaned = Path(cleaned).stem
    if not cleaned:
        raise ValueError("profile name cannot be empty")
    if any(separator in cleaned for separator in ("/", "\\")) or ".." in cleaned:
        raise ValueError("profile name must not contain path separators")
    if re.match(r"^[A-Za-z]:", cleaned):
        raise ValueError("profile name must not be an absolute path")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", cleaned)
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("profile name has no usable characters")
    return cleaned


def profile_path(profile_dir: str | Path, name: str) -> Path:
    directory = Path(profile_dir).resolve()
    sanitized = sanitize_profile_name(name)
    path = (directory / f"{sanitized}{PROFILE_SUFFIX}").resolve()
    if path.parent != directory:
        raise ValueError("profile path escaped the visual calibration directory")
    return path


def save_visual_calibration(
    profile_dir: str | Path,
    name: str,
    kinematics,
) -> Path:
    path = profile_path(profile_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "version": VISUAL_CALIBRATION_VERSION,
        "name": path.stem,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kinematics": kinematics.export_visual_calibration(),
    }
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(profile, file, sort_keys=True, allow_unicode=False)
    return path


def load_visual_calibration(path_or_dir: str | Path, name: str | None = None) -> dict[str, Any]:
    path = (
        profile_path(path_or_dir, name)
        if name is not None
        else Path(path_or_dir).resolve()
    )
    with path.open("r", encoding="utf-8") as file:
        profile = yaml.safe_load(file) or {}
    if int(profile.get("version", 0)) != VISUAL_CALIBRATION_VERSION:
        raise ValueError(f"unsupported visual calibration version in {path}")
    if "kinematics" not in profile:
        raise ValueError(f"visual calibration profile {path} has no kinematics block")
    return profile


def apply_visual_calibration(profile: dict[str, Any], kinematics) -> None:
    kinematics.import_visual_calibration(profile["kinematics"])


def list_visual_calibrations(profile_dir: str | Path) -> list[VisualCalibrationProfile]:
    directory = Path(profile_dir)
    if not directory.exists():
        return []

    profiles: list[VisualCalibrationProfile] = []
    for path in sorted(directory.glob(f"*{PROFILE_SUFFIX}")):
        if not path.is_file():
            continue
        try:
            profile = load_visual_calibration(path)
        except Exception:
            continue
        profiles.append(
            VisualCalibrationProfile(
                name=str(profile.get("name", path.stem)),
                path=path,
                created_at=str(profile.get("created_at", "")),
            )
        )
    return sorted(profiles, key=lambda profile: profile.name)
