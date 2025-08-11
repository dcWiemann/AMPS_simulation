import pytest
from amps_simulation.core.control_port import ControlPort

def test_control_port_unique_name():
    ControlPort.clear_registry()
    port1 = ControlPort(name="portA", variable=5)
    assert port1.name == "portA"
    assert port1.variable == 5
    # Should raise ValueError for duplicate name
    with pytest.raises(ValueError):
        ControlPort(name="portA", variable=10)