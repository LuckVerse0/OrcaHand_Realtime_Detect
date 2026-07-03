from pathlib import Path

from orca_realtime.logging_utils import SessionLogger
from orca_realtime.orca_controller import OrcaController


def test_session_logger_writes_csv_and_jsonl_rows(tmp_path):
    logger = SessionLogger(tmp_path, "session")

    logger.write(
        {
            "timestamp": 1.23,
            "state": "live",
            "accepted": True,
            "reasons": [],
            "joints": {"index_mcp": 12.0},
            "motor_positions": {15: 1.5},
        }
    )
    logger.close()

    csv_text = (tmp_path / "session.csv").read_text(encoding="utf-8")
    jsonl_text = (tmp_path / "session.jsonl").read_text(encoding="utf-8")

    assert "index_mcp" in csv_text
    assert '"state": "live"' in jsonl_text


def test_session_logger_batches_flushes_and_flushes_on_close(tmp_path):
    logger = SessionLogger(
        tmp_path,
        "session",
        flush_every=2,
        flush_interval_s=999.0,
    )
    row = {
        "timestamp": 1.23,
        "state": "live",
        "accepted": True,
        "reasons": [],
        "joints": {"index_mcp": 12.0},
        "motor_positions": {15: 1.5},
    }

    logger.write(row)

    assert "index_mcp" not in (tmp_path / "session.csv").read_text(encoding="utf-8")
    assert "index_mcp" not in (tmp_path / "session.jsonl").read_text(encoding="utf-8")

    logger.write(row)

    assert "index_mcp" in (tmp_path / "session.csv").read_text(encoding="utf-8")
    assert "index_mcp" in (tmp_path / "session.jsonl").read_text(encoding="utf-8")

    logger.write(row)
    logger.close()

    assert (tmp_path / "session.csv").read_text(encoding="utf-8").count("index_mcp") == 3


def test_orca_controller_dry_run_never_requires_orca_core(tmp_path):
    controller = OrcaController(
        config_path=Path("config") / "config.yaml",
        live=False,
        orca_core_root=tmp_path / "missing",
    )

    controller.connect()
    controller.send({"index_mcp": 10.0})
    controller.emergency_stop()

    assert controller.connected is False


def test_orca_controller_uses_official_lifecycle_and_send_timing(tmp_path, monkeypatch):
    events = []

    class FakeHand:
        def __init__(self, config_path):
            events.append(("init", config_path))

        def connect(self):
            events.append(("connect",))
            return True, "connected"

        def init_joints(self, *, force_calibrate=False, move_to_neutral=True):
            events.append(("init_joints", force_calibrate, move_to_neutral))

        def set_joint_positions(self, joints, *, num_steps=1, step_size=1e-2):
            events.append(("set_joint_positions", joints, num_steps, step_size))

        def stop_task(self):
            events.append(("stop_task",))

        def disconnect(self):
            events.append(("disconnect",))
            return True, "disconnected"

    controller = OrcaController(
        config_path=tmp_path / "config.yaml",
        live=True,
        force_calibrate=True,
        move_to_neutral=False,
        send_num_steps=4,
        send_step_size=0.02,
    )
    monkeypatch.setattr(controller, "_load_orca_hand", lambda: FakeHand)

    controller.connect()
    controller.send({"index_mcp": 12.0})
    controller.disconnect()

    assert events == [
        ("init", str(tmp_path / "config.yaml")),
        ("connect",),
        ("init_joints", True, False),
        ("set_joint_positions", {"index_mcp": 12.0}, 4, 0.02),
        ("stop_task",),
        ("disconnect",),
    ]
    assert controller.connected is False


def test_orca_controller_emergency_stop_stops_tasks_before_disabling_torque(tmp_path, monkeypatch):
    events = []

    class FakeHand:
        def __init__(self, config_path):
            pass

        def connect(self):
            return True, "connected"

        def init_joints(self, *, force_calibrate=False, move_to_neutral=True):
            pass

        def stop_task(self):
            events.append("stop_task")

        def disable_torque(self):
            events.append("disable_torque")

    controller = OrcaController(tmp_path / "config.yaml", live=True)
    monkeypatch.setattr(controller, "_load_orca_hand", lambda: FakeHand)

    controller.connect()
    controller.emergency_stop()

    assert events == ["stop_task", "disable_torque"]
    assert controller.connected is False
    assert controller.hand is None


def test_orca_controller_disconnect_clears_hand_even_when_disconnect_fails(tmp_path):
    class FakeHand:
        def stop_task(self):
            pass

        def disconnect(self):
            raise RuntimeError("disconnect failed")

    controller = OrcaController(tmp_path / "config.yaml", live=True)
    controller.hand = FakeHand()
    controller.connected = True

    try:
        controller.disconnect()
    except RuntimeError as exc:
        assert str(exc) == "disconnect failed"

    assert controller.connected is False
    assert controller.hand is None


def test_orca_controller_mock_backend_does_not_force_calibration_by_default(tmp_path, monkeypatch):
    init_calls = []

    class FakeHand:
        def __init__(self, config_path):
            pass

        def connect(self):
            return True, "connected"

        def init_joints(self, *, force_calibrate=False, move_to_neutral=True):
            init_calls.append((force_calibrate, move_to_neutral))

    controller = OrcaController(tmp_path / "config.yaml", live=True, mock=True)
    monkeypatch.setattr(controller, "_load_orca_hand", lambda: FakeHand)

    controller.connect()

    assert init_calls == [(False, True)]
