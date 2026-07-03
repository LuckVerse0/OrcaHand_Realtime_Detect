# 1080p60 Video Import Playback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a connected-hardware-only video import flow that preprocesses an input video into a 1080p60 skeleton overlay video and replays it while synchronizing right-hand motion to OrcaHand.

**Architecture:** Keep camera recognition and imported-video playback as separate modes inside `OrcaRealtimeGui`. The preprocessing worker reads source frames, detects the preferred/right hand, writes a letterboxed 1920x1080/60fps output video, and records per-frame safe joint commands. Playback reads the processed video on the Tk event loop and sends cached joint commands by frame index.

**Tech Stack:** Python, Tkinter, OpenCV, MediaPipe hand landmarker, existing `HandKinematics`, `SafetyController`, and `OrcaController`.

---

### Task 1: Video Processing Data Model

**Files:**
- Modify: `realtime_orcahand.py`
- Test: `tests/test_realtime_orcahand_video_import.py`

- [ ] Add `ProcessedVideoFrame` and `ProcessedVideo` dataclasses with frame index, timestamp, optional joints, output path, fps, width, and height.
- [ ] Add tests that construct these dataclasses and verify timestamps are stable for 60fps.

### Task 2: 1080p Letterbox Helpers

**Files:**
- Modify: `realtime_orcahand.py`
- Test: `tests/test_realtime_orcahand_video_import.py`

- [ ] Add `_letterbox_to_size(frame, target_width=1920, target_height=1080)` that preserves aspect ratio and pads with black.
- [ ] Add tests for 16:9 and non-16:9 frames to verify output shape and padding.

### Task 3: Offline Video Processor

**Files:**
- Modify: `realtime_orcahand.py`
- Test: `tests/test_realtime_orcahand_video_import.py`

- [ ] Add `process_video_to_1080p60(...)` that opens input video, creates the MediaPipe landmarker, processes every frame, draws the skeleton on the source frame, letterboxes to 1080p, writes output video at 60fps, and records safe joints per frame.
- [ ] Keep inference bounded by existing `detect_frame()` behavior instead of passing 4K frames directly to the model.
- [ ] Treat no-hand and rejected-safety frames as frames with `joints=None`.

### Task 4: GUI Import and Gating

**Files:**
- Modify: `realtime_orcahand.py`
- Test: `tests/test_realtime_orcahand_video_import.py`

- [ ] Add `Import Video` and `Stop Video` buttons to the right panel.
- [ ] Enable import only when hardware is allowed, connected, not processing, and not playing.
- [ ] Use `filedialog.askopenfilename` for video selection.
- [ ] Run preprocessing in a background thread and report completion via the existing queue.

### Task 5: Processed Video Playback and Hardware Sync

**Files:**
- Modify: `realtime_orcahand.py`
- Test: `tests/test_realtime_orcahand_video_import.py`

- [ ] Add playback state fields: processed video, playback capture, frame index, playback start time, and playing flag.
- [ ] Display processed frames in the existing left video label.
- [ ] On each playback frame, send cached joints for that frame when connected and safe.
- [ ] Stop playback on end, stop button, emergency stop, close, or runtime error.

### Task 6: Verification

**Files:**
- Modify: none

- [ ] Run `python -m py_compile realtime_orcahand.py`.
- [ ] Run focused tests:
  `python -m pytest tests/test_realtime_orcahand_video_import.py tests/test_realtime_orcahand_missing_calibration.py tests/test_realtime_orcahand_calibration_stop.py -q`.
- [ ] Run a smoke import command that imports `realtime_orcahand` and constructs core video dataclasses.
