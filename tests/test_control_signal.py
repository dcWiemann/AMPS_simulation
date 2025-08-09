import pytest
from amps_simulation.core.control_signal import ControlSignal

def test_control_signal_creation_and_call():
    ControlSignal.clear_registry()

    def step_func(t, threshold=1.0):
        return 1 if t >= threshold else 0

    cs = ControlSignal(name="step", control_function=step_func)
    assert cs.name == "step"
    assert cs(0.5) == 0
    assert cs(1.0) == 1
    assert cs(2.0) == 1

def test_control_signal_uniqueness():
    ControlSignal.clear_registry()

    cs1 = ControlSignal(name="unique", control_function=lambda t: t)
    with pytest.raises(ValueError):
        ControlSignal(name="unique", control_function=lambda t: t + 1)

def test_get_control_signal():
    ControlSignal.clear_registry()

    cs = ControlSignal(name="findme", control_function=lambda t: t)
    found = ControlSignal.get_control_signal("findme")
    assert found is cs
    assert found(5) == 5

def test_clear_registry():
    ControlSignal.clear_registry()
    cs = ControlSignal(name="to_clear", control_function=lambda t: t)
    assert ControlSignal.get_control_signal("to_clear") is cs
    ControlSignal.clear_registry()
    assert ControlSignal.get_control_signal("to_clear") is None