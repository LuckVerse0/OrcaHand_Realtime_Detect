from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


GUI_FILE = Path("realtime_orcahand_gui.py")
pytestmark = pytest.mark.skipif(
    not GUI_FILE.exists(),
    reason="legacy realtime_orcahand_gui.py entrypoint is not present; single-file GUI is covered separately",
)


def load_gui_module():
    spec = importlib.util.spec_from_file_location("realtime_orcahand_gui", GUI_FILE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gui_module_imports_without_starting_tk_root():
    module = load_gui_module()

    assert hasattr(module, "OrcaRealtimeGui")
    assert hasattr(module, "build_arg_parser")


def test_gui_parser_defaults_save_visual_profiles_in_profiles_folder():
    module = load_gui_module()
    args = module.build_arg_parser().parse_args([])

    assert args.config_dir == Path("config")
    assert args.visual_calibration_dir == Path("profiles") / "visual"
    assert args.log_dir == Path("logs")
    assert args.orca_core_root == module.PROJECT_ROOT / "vendor" / "orca_core"
    assert args.live is True
    assert module.build_arg_parser().parse_args(["--preview-only"]).live is False
    assert module.build_arg_parser().parse_args(["--no-live"]).live is False


def test_gui_right_control_panel_is_scrollable():
    source = GUI_FILE.read_text(encoding="utf-8")

    gui_source = source.split("class OrcaRealtimeGui:", 1)[1].split(
        "def run_gui",
        1,
    )[0]

    assert "tk.Canvas" in gui_source
    assert "ttk.Scrollbar" in gui_source
    assert "yscrollcommand" in gui_source
    assert "_on_panel_mousewheel" in gui_source


def test_gui_keeps_start_mapping_clickable_for_calibration_guidance():
    module = load_gui_module()

    class FakeVar:
        def __init__(self):
            self.value = None

        def set(self, value):
            self.value = value

    class FakeButton:
        def __init__(self):
            self.state = None

        def configure(self, *, state):
            self.state = state

    class FakeController:
        connected = False

    class FakeGui:
        running = False
        hardware_allowed = False
        _background_busy = False
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

        def _mapping_calibration_ready(self):
            return False

    gui = FakeGui()

    module.OrcaRealtimeGui._update_button_states(gui)

    assert gui.map_button.state == "normal"


def test_gui_start_mapping_enables_hardware_output_when_connected():
    module = load_gui_module()

    class FakeController:
        connected = True

    class FakeResettable:
        def __init__(self):
            self.reset_count = 0

        def reset_to_safe_neutral(self):
            self.reset_count += 1

        def reset(self):
            self.reset_count += 1

    class FakeGui:
        running = True
        hardware_allowed = True
        controller = FakeController()
        state = module.RuntimeStateMachine()
        safety = FakeResettable()
        joint_smoother = FakeResettable()

        def __init__(self):
            self.status_messages = []
            self.update_count = 0

        def _mapping_calibration_ready(self):
            return True

        def _set_status(self, message):
            self.status_messages.append(message)

        def _update_button_states(self):
            self.update_count += 1

    gui = FakeGui()

    module.OrcaRealtimeGui.start_mapping(gui)

    assert gui.state.state == module.RuntimeState.LIVE
    assert gui.safety.reset_count == 1
    assert gui.joint_smoother.reset_count == 1
    assert gui.update_count == 1
    assert "Hardware output enabled" in gui.status_messages[-1]
