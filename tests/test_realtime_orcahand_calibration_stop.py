from __future__ import annotations

import threading
import time

import realtime_orcahand as rt


def load_realtime_module():
    return rt


def test_orca_controller_calibration_task_can_be_stopped(monkeypatch):
    module = load_realtime_module()
    hand_instances = []

    class FakeHand:
        def __init__(self, config_path):
            self.config_path = config_path
            self.entered = threading.Event()
            self.exited = threading.Event()
            self._task_stop_event = threading.Event()
            self._task_thread = None
            hand_instances.append(self)

        def connect(self):
            return True, "connected"

        def calibrate(self, *, force_wrist=False, joints=None):
            raise AssertionError("calibration must run as a stoppable task")

        def _start_task(self, task_fn, *args, **kwargs):
            self._task_stop_event.clear()
            self._task_thread = threading.Thread(
                target=task_fn,
                args=args,
                kwargs=kwargs,
                daemon=True,
            )
            self._task_thread.start()

        def _calibrate(self, *, force_wrist=False, joints=None):
            self.calibration_args = (force_wrist, joints)
            self.entered.set()
            while not self._task_stop_event.is_set():
                time.sleep(0.01)
            self.exited.set()
            return None

    controller = module.OrcaController("config/config.yaml", live=True)
    monkeypatch.setattr(controller, "_load_orca_hand", lambda: FakeHand)
    worker_errors = []

    def run_calibration():
        try:
            controller.calibrate(force_wrist=True, joints=["index_mcp"])
        except module.CalibrationStopped as exc:
            worker_errors.append(exc)

    worker = threading.Thread(
        target=run_calibration,
        daemon=True,
    )

    worker.start()
    hand = hand_instances[0]
    assert hand.entered.wait(1.0)

    controller.stop_task()
    worker.join(1.0)

    assert worker.is_alive() is False
    assert hand.exited.is_set()
    assert hand.calibration_args == (True, ["index_mcp"])
    assert len(worker_errors) == 1


def test_gui_stop_calibration_button_tracks_calibration_busy():
    module = load_realtime_module()

    class FakeVar:
        def set(self, _value):
            pass

    class FakeButton:
        def __init__(self):
            self.state = None

        def configure(self, *, state):
            self.state = state

    class FakeController:
        connected = False

    class FakeGui:
        running = False
        hardware_allowed = True
        _background_busy = False
        _calibration_busy = False
        state = module.RuntimeStateMachine()
        controller = FakeController()
        state_var = FakeVar()
        hardware_var = FakeVar()
        start_button = FakeButton()
        stop_button = FakeButton()
        connect_button = FakeButton()
        map_button = FakeButton()
        stop_map_button = FakeButton()
        output_button = FakeButton()
        stop_output_button = FakeButton()
        calibrate_button = FakeButton()
        stop_calibrate_button = FakeButton()

    gui = FakeGui()

    module.OrcaRealtimeGui._update_button_states(gui)
    assert gui.stop_calibrate_button.state == "disabled"

    gui._calibration_busy = True
    module.OrcaRealtimeGui._update_button_states(gui)

    assert gui.stop_calibrate_button.state == "normal"
