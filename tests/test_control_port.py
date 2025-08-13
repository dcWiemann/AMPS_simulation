import pytest
from amps_simulation.core.control_port import ControlPort

def test_control_port_unique_name():
    ControlPort.clear_registry()
    port1 = ControlPort(name="portA", variable=5, port_type="source")
    assert port1.name == "portA"
    assert port1.variable == 5
    assert port1.port_type == "source"
    # Should raise ValueError for duplicate name
    with pytest.raises(ValueError):
        ControlPort(name="portA", variable=10, port_type="switch")

def test_control_port_types():
    ControlPort.clear_registry()
    
    # Test different port types
    source_port = ControlPort(name="source_test", variable="V1", port_type="source")
    switch_port = ControlPort(name="switch_test", variable="S1", port_type="switch")  
    other_port = ControlPort(name="other_test", variable="X1", port_type="other")
    
    assert source_port.port_type == "source"
    assert switch_port.port_type == "switch"
    assert other_port.port_type == "other"

def test_control_port_registry_operations():
    ControlPort.clear_registry()
    
    port = ControlPort(name="registry_test", variable="V1", port_type="source")
    
    # Test get_control_port
    retrieved = ControlPort.get_control_port("registry_test")
    assert retrieved is port
    
    # Test get_control_port with non-existent name
    assert ControlPort.get_control_port("nonexistent") is None
    
    # Test clear_registry
    ControlPort.clear_registry()
    assert ControlPort.get_control_port("registry_test") is None