import numpy as np

import realtime_orcahand as rt


ExponentialSmoother = rt.ExponentialSmoother
JointSmoother = rt.JointSmoother
RuntimeState = rt.RuntimeState
RuntimeStateMachine = rt.RuntimeStateMachine


def test_exponential_smoother_blends_arrays_and_resets():
    smoother = ExponentialSmoother(alpha=0.25)

    first = smoother.update(np.array([0.0, 0.0]))
    second = smoother.update(np.array([4.0, 8.0]))
    smoother.reset()
    third = smoother.update(np.array([10.0, 20.0]))

    np.testing.assert_allclose(first, np.array([0.0, 0.0]))
    np.testing.assert_allclose(second, np.array([1.0, 2.0]))
    np.testing.assert_allclose(third, np.array([10.0, 20.0]))


def test_joint_smoother_uses_abd_alpha_for_abduction_joints():
    smoother = JointSmoother(default_alpha=0.5, abd_alpha=0.25)

    smoother.update({"index_mcp": 0.0, "index_abd": 0.0})
    result = smoother.update({"index_mcp": 10.0, "index_abd": 8.0})

    assert result["index_mcp"] == 5.0
    assert result["index_abd"] == 2.0


def test_state_machine_gates_live_and_fault_reset():
    machine = RuntimeStateMachine()

    assert machine.state == RuntimeState.PREVIEW
    machine.start_mapping()
    assert machine.state == RuntimeState.ARMED
    machine.enable_live()
    assert machine.state == RuntimeState.LIVE
    machine.tracking_lost("lost hand")
    assert machine.state == RuntimeState.TRACKING_LOST
    assert machine.reason == "lost hand"
    machine.recover_tracking()
    assert machine.state == RuntimeState.LIVE
    machine.fault("emergency")
    assert machine.state == RuntimeState.FAULT
    machine.reset_fault()
    assert machine.state == RuntimeState.PREVIEW
