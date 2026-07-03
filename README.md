# Real Time Detect With Orca

Realtime MediaPipe hand tracking for OrcaHand, with a preview-first GUI and a
dry-run default.

## Setup

The main app runs from the repository root environment. It imports
`vendor/orca_core` directly from source, so you only need one virtual
environment to run `realtime_orcahand.py`.

From a fresh clone:

```powershell
cd "\Real Time Detect With Orca"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

Primary GUI:

```powershell
.\.venv\Scripts\python.exe realtime_orcahand.py --preview-only
```

For live hardware output:

```powershell
.\.venv\Scripts\python.exe realtime_orcahand.py --live
```

Useful checks:

```powershell
.\.venv\Scripts\python.exe realtime_orcahand.py --check
.\.venv\Scripts\python.exe -m pytest -q
```

## Editor Setup

The workspace settings are tuned for VS Code to prefer the root `.venv` and to
resolve imports from both the repo root and `vendor\orca_core`.

If VS Code asks you to pick an interpreter, select:

```text
${workspaceFolder}\.venv\Scripts\python.exe
```

## Vendor Core

`vendor/orca_core` is vendored in this repo for convenience. You do **not**
need to activate its own virtual environment to run the main GUI.

If you want to work on `orca_core` itself, use its own package workflow:

```powershell
cd vendor\orca_core
uv sync --group dev
uv run pytest
```

## Current Runtime

- Tracker: MediaPipe Tasks HandLandmarker
- Delegate on Windows: TensorFlow Lite XNNPACK CPU
- Hands detected per frame: 2 by default, then filtered to the configured hand
- Camera target: 640x480 at 30 FPS with a one-frame capture buffer
- Camera reading: background latest-frame cache so slow `cap.read()` calls do
  not block the Tk frame loop
- Inference target: downscaled to 480px width by default
- Display target: capped at 640x480 for Tk image conversion

These defaults keep the CPU path near a 30 FPS budget while preserving a clear
preview. The app still shows the camera preview at the display size; only the
MediaPipe input is downscaled for inference.

## Notes

- `inference_ms` is the MediaPipe tracking time for one frame, not total frame
  time. Total frame time also includes camera read, drawing, Tk conversion, and
  scheduling.
- Keeping `DEFAULT_MAX_DETECTED_HANDS = 2` is intentional. It lets the program
  see both hands and then choose the configured hand, instead of letting
  MediaPipe pick the wrong single hand.
- The older `realtime_orcahand_control.py` and `tools\mediapipe_hand.py` paths
  are kept for CLI/demo use and now share the same realtime camera and inference
  defaults.

## Experimental Tools

The `tools\rtmpose_hand_cuda.py` and `tools\vitpose_hand_cuda.py` scripts are
legacy GPU experiments. They are useful for model comparison, but they are not
the primary OrcaHand GUI path.
