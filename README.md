# Real Time Detect With Orca

Realtime MediaPipe hand tracking for OrcaHand, with a preview-first GUI and a
dry-run default. The current primary entrypoint is the single-file GUI:

```powershell
.\.venv\Scripts\python.exe realtime_orcahand_single_file.py --preview-only
```

Use `--live` only when the OrcaHand hardware is connected and the calibration
files under `config\` are valid.

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

## Run

```powershell
cd "D:\programming\Real Time Detect With Orca"
.\.venv\Scripts\python.exe realtime_orcahand_single_file.py --preview-only
```

For live hardware output:

```powershell
.\.venv\Scripts\python.exe realtime_orcahand_single_file.py --live
```

Useful checks:

```powershell
.\.venv\Scripts\python.exe realtime_orcahand_single_file.py --check
.\.venv\Scripts\python.exe -m pytest -q
```

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
