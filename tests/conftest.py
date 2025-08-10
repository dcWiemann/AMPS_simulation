import pytest
from amps_simulation.core.control_port import ControlPort

@pytest.fixture(autouse=True)
def clear_control_port_registry():
    ControlPort.clear_registry()
    yield
    ControlPort.clear_registry()