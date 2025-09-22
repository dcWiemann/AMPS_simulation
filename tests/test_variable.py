import pytest
from sympy import symbols
from amps_simulation.core.variable import Variable
from amps_simulation.core.components import Resistor, Component
from amps_simulation.core.control_port import ControlPort


def test_variable_creation():
    """Test basic Variable creation with required attributes."""
    # Clear registries
    Component.clear_registry()
    ControlPort.clear_registry()

    # Create a component
    resistor = Resistor(comp_id="R1", resistance=100.0)

    # Create a symbolic variable
    symbolic_var = symbols("i_R1")

    # Create Variable instance
    variable = Variable(component=resistor, symbolic=symbolic_var)

    # Test attributes
    assert variable.component == resistor
    assert variable.symbolic == symbolic_var
    assert variable.control_port is None


def test_variable_with_control_port():
    """Test Variable creation with optional control_port attribute."""
    # Clear registries
    Component.clear_registry()
    ControlPort.clear_registry()

    # Create a component
    resistor = Resistor(comp_id="R1", resistance=100.0)

    # Create a symbolic variable
    symbolic_var = symbols("v_R1")

    # Create a control port
    control_port = ControlPort(name="voltage_control", variable=symbolic_var, port_type="source")

    # Create Variable instance with control port
    variable = Variable(component=resistor, symbolic=symbolic_var, control_port=control_port)

    # Test attributes
    assert variable.component == resistor
    assert variable.symbolic == symbolic_var
    assert variable.control_port == control_port
    assert variable.control_port.name == "voltage_control"
    assert variable.control_port.port_type == "source"


def test_variable_component_relationship():
    """Test that Variable properly maintains component relationship."""
    # Clear registries
    Component.clear_registry()
    ControlPort.clear_registry()

    # Create multiple components
    resistor1 = Resistor(comp_id="R1", resistance=100.0)
    resistor2 = Resistor(comp_id="R2", resistance=200.0)

    # Create variables for each component
    var1 = Variable(component=resistor1, symbolic=symbols("i_R1"))
    var2 = Variable(component=resistor2, symbolic=symbols("i_R2"))

    # Test that variables maintain correct component references
    assert var1.component.comp_id == "R1"
    assert var1.component.resistance == 100.0
    assert var2.component.comp_id == "R2"
    assert var2.component.resistance == 200.0

    # Test that components are different
    assert var1.component != var2.component


def test_variable_symbolic_types():
    """Test Variable with different types of symbolic variables."""
    # Clear registries
    Component.clear_registry()
    ControlPort.clear_registry()

    resistor = Resistor(comp_id="R1", resistance=100.0)

    # Test with current symbol
    current_var = Variable(component=resistor, symbolic=symbols("i_R1"))
    assert str(current_var.symbolic) == "i_R1"

    # Test with voltage symbol
    voltage_var = Variable(component=resistor, symbolic=symbols("v_R1"))
    assert str(voltage_var.symbolic) == "v_R1"

    # Test with custom symbol
    custom_var = Variable(component=resistor, symbolic=symbols("x"))
    assert str(custom_var.symbolic) == "x"


def test_variable_pydantic_validation():
    """Test that Variable properly validates required fields."""
    # Clear registries
    Component.clear_registry()
    ControlPort.clear_registry()

    resistor = Resistor(comp_id="R1", resistance=100.0)
    symbolic_var = symbols("i_R1")

    # Test that component is required
    with pytest.raises(ValueError):
        Variable(symbolic=symbolic_var)

    # Test that symbolic is required
    with pytest.raises(ValueError):
        Variable(component=resistor)

    # Test that both required fields work
    variable = Variable(component=resistor, symbolic=symbolic_var)
    assert variable.component == resistor
    assert variable.symbolic == symbolic_var