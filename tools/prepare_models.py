from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "rtmlib" / "hub" / "checkpoints"
MODELS = (
    {
        "name": "rtmdet_nano_8xb32-300e_hand-267f9c8f",
        "url": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip"
        ),
    },
    {
        "name": "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320",
        "url": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"
        ),
    },
)


def download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        with destination.open("wb") as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded / total * 100
                    print(f"\r{destination.name}: {percent:5.1f}%", end="")
        print()


def extract_end2end(zip_path: Path, output_path: Path) -> None:
    temp_dir = zip_path.with_suffix("")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(temp_dir)

        matches = list(temp_dir.rglob("*end2end.onnx"))
        if not matches:
            raise RuntimeError(f"No end2end.onnx found in {zip_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(matches[0]), output_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        output_path = MODEL_DIR / f"{model['name']}.onnx"
        if output_path.exists():
            print(f"Exists: {output_path}")
            continue

        zip_path = MODEL_DIR / f"{model['name']}.zip"
        if zip_path.exists():
            zip_path.unlink()

        print(f"Downloading {model['name']} ...")
        download_file(model["url"], zip_path)
        extract_end2end(zip_path, output_path)
        zip_path.unlink(missing_ok=True)
        print(f"Ready: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
