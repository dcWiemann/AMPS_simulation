import pytest
from amps_simulation.core.components import Resistor, Component, ElecJunction, PowerSwitch

def test_component_variable_names():
    """Test that component variable names are correctly generated."""
    # Create a test component
    resistor = Resistor(comp_id="R1", resistance=100.0)
    
    # Test current and voltage variable names
    assert resistor.current_var == "i_R1"
    assert resistor.voltage_var == "v_R1"
    
    # Test with a different component ID
    resistor2 = Resistor(comp_id="R2", resistance=200.0)
    assert resistor2.current_var == "i_R2"
    assert resistor2.voltage_var == "v_R2"

def test_component_registry():
    """Test the component registry functionality."""
    # Clear the registry before testing
    Component.clear_registry()
    
    # Create components
    r1 = Resistor(comp_id="R1", resistance=100.0)
    r2 = Resistor(comp_id="R2", resistance=200.0)
    
    # Test component retrieval
    assert Component.get_component("R1") == r1
    assert Component.get_component("R2") == r2
    assert Component.get_component("R3") is None

def test_unique_component_ids():
    """Test that duplicate component IDs are not allowed."""
    # Clear the registry before testing
    Component.clear_registry()
    
    # Create first component
    Resistor(comp_id="R1", resistance=100.0)
    
    # Attempt to create second component with same ID
    with pytest.raises(ValueError, match="Component ID 'R1' is already in use"):
        Resistor(comp_id="R1", resistance=200.0)

def test_component_id_starts_with_letter():
    """Test that component IDs must start with a letter."""
    # Clear the registry before testing
    Component.clear_registry()
    
    # Valid component IDs
    Resistor(comp_id="R1", resistance=100.0)
    Resistor(comp_id="L1", resistance=200.0)
    Resistor(comp_id="C1", resistance=300.0)
    
    # Invalid component IDs
    with pytest.raises(ValueError, match=r"String should match pattern '\^\[A-Za-z\]\..*'"):
        Resistor(comp_id="1R", resistance=400.0)
    
    with pytest.raises(ValueError, match=r"String should match pattern '\^\[A-Za-z\]\..*'"):
        Resistor(comp_id="_R1", resistance=500.0)

def test_elec_junction_voltage_var():
    """Test that ElecJunction voltage variable names are correctly generated."""
    # Create a test junction
    junction = ElecJunction(junction_id=1)
    
    # Test voltage variable name
    assert junction.voltage_var == "V_1"
    
    # Test with a different junction ID
    junction2 = ElecJunction(junction_id=2)
    assert junction2.voltage_var == "V_2"

def test_unique_junction_ids():
    """Test that duplicate junction IDs are not allowed."""
    # Clear the registry before testing
    ElecJunction.clear_registry()
    
    # Create first junction
    ElecJunction(junction_id=1)
    
    # Attempt to create second junction with same ID
    with pytest.raises(ValueError, match="Junction ID '1' is already in use"):
        ElecJunction(junction_id=1)

def test_power_switch_control_signal():
    """Test the control signal calculation for PowerSwitch."""
    # Create a PowerSwitch instance
    switch = PowerSwitch(comp_id="SW1", switch_time=1.0, is_on=True)
    
    # Test when the switch is on
    assert switch.control_signal(0.2) == 0  # Should be 0 when off, before switch_time
    
    # Test when the switch is off
    switch.is_on = False
    assert switch.control_signal(1) == 1  # Should be 1 when on, after switch_time