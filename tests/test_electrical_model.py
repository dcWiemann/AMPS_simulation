import pytest
import json
import os
import sympy as sp
from amps_simulation.core.electrical_model import ElectricalModel
from amps_simulation.core.parsing import build_electrical_nodes, build_circuit_components
from amps_simulation.core.simulation import Simulation

# Test data configuration
TEST_DATA_DIR = "test_data"
TEST_FILES = [
    "test_rlc.json",      # RLC circuit
    "test_2v.json",       # Circuit with 2 voltage sources
]

def load_test_file(filename):
    """Load a test circuit file and return its contents."""
    file_path = os.path.join(TEST_DATA_DIR, filename)
    with open(file_path, 'r') as f:
        return json.load(f)

# Mock data for basic tests
@pytest.fixture
def mock_rlc_circuit():
    """Create a simple RLC circuit for testing."""
    electrical_nodes = {
        0: {("V1", "1"), ("R1", "0")},
        1: {("R1", "1"), ("C1", "0"), ("L1", "0")},
        2: {("C1", "1"), ("L1", "1"), ("V1", "0")}
    }
    
    circuit_components = {
        "V1": {
            "type": "voltage-source",
            "value": None,  # Symbolic model doesn't need values
            "terminals": {"0": 2, "1": 0}
        },
        "R1": {
            "type": "resistor",
            "value": None,
            "terminals": {"0": 0, "1": 1}
        },
        "C1": {
            "type": "capacitor",
            "value": None,
            "terminals": {"0": 1, "1": 2}
        },
        "L1": {
            "type": "inductor",
            "value": None,
            "terminals": {"0": 1, "1": 2}
        }
    }
    
    # Create variables using Simulation class
    simulation = Simulation(electrical_nodes, circuit_components)
    voltage_vars, current_vars, state_vars, state_derivatives, input_vars, ground_node = simulation.assign_variables()
    
    return {
        "electrical_nodes": electrical_nodes,
        "circuit_components": circuit_components,
        "voltage_vars": voltage_vars,
        "current_vars": current_vars,
        "state_vars": state_vars,
        "state_derivatives": state_derivatives,
        "input_vars": input_vars,
        "ground_node": ground_node
    }

def test_electrical_model_init(mock_rlc_circuit):
    """Test ElectricalModel initialization."""
    model = ElectricalModel(**mock_rlc_circuit)
    
    assert model.electrical_nodes == mock_rlc_circuit["electrical_nodes"]
    assert model.circuit_components == mock_rlc_circuit["circuit_components"]
    assert model.voltage_vars == mock_rlc_circuit["voltage_vars"]
    assert model.current_vars == mock_rlc_circuit["current_vars"]
    assert model.state_vars == mock_rlc_circuit["state_vars"]
    assert model.input_vars == mock_rlc_circuit["input_vars"]
    assert model.ground_node == mock_rlc_circuit["ground_node"]

def test_electrical_model_build(mock_rlc_circuit):
    """Test building the electrical model."""
    model = ElectricalModel(**mock_rlc_circuit)
    A, B, solved_helpers, differential_equations = model.build_model()
    
    # Basic validation
    assert isinstance(A, sp.Matrix), "A should be a SymPy Matrix"
    assert isinstance(B, sp.Matrix), "B should be a SymPy Matrix"
    assert isinstance(solved_helpers, dict), "solved_helpers should be a dictionary"
    assert isinstance(differential_equations, dict), "differential_equations should be a dictionary"
    
    # For RLC circuit:
    assert A.shape == (2, 2), "A should be 2x2 for RLC circuit"
    assert B.shape == (2, 1), "B should be 2x1 for RLC circuit"

@pytest.mark.parametrize("test_file", TEST_FILES)
def test_electrical_model_with_real_circuits(test_file):
    """Test ElectricalModel with real circuit data."""
    # Load and parse circuit data
    circuit_data = load_test_file(test_file)
    components = circuit_data["nodes"]
    connections = circuit_data["edges"]
    
    electrical_nodes = build_electrical_nodes(components, connections)
    circuit_components = build_circuit_components(components, electrical_nodes)
    
    # Create variables using Simulation class
    simulation = Simulation(electrical_nodes, circuit_components)
    voltage_vars, current_vars, state_vars, state_derivatives, input_vars, ground_node = simulation.assign_variables()
    
    # Create and test ElectricalModel
    model = ElectricalModel(
        electrical_nodes=electrical_nodes,
        circuit_components=circuit_components,
        voltage_vars=voltage_vars,
        current_vars=current_vars,
        state_vars=state_vars,
        state_derivatives=state_derivatives,
        input_vars=input_vars,
        ground_node=ground_node
    )
    
    A, B, solved_helpers, differential_equations = model.build_model()
    
    # Validate outputs
    assert isinstance(A, sp.Matrix), f"A should be a SymPy Matrix for {test_file}"
    assert isinstance(B, sp.Matrix), f"B should be a SymPy Matrix for {test_file}"
    assert A.shape[0] == len(state_vars), f"A dimensions should match state_vars count for {test_file}"
    assert B.shape[1] == len(input_vars), f"B columns should match input_vars count for {test_file}"

def test_electrical_model_specific_rlc_circuit():
    """Test specific properties of the RLC circuit model."""
    # Load RLC circuit
    circuit_data = load_test_file("test_rlc.json")
    components = circuit_data["nodes"]
    connections = circuit_data["edges"]
    
    electrical_nodes = build_electrical_nodes(components, connections)
    circuit_components = build_circuit_components(components, electrical_nodes)
    
    # Create variables using Simulation class
    simulation = Simulation(electrical_nodes, circuit_components)
    voltage_vars, current_vars, state_vars, state_derivatives, input_vars, ground_node = simulation.assign_variables()
    
    # Create model
    model = ElectricalModel(
        electrical_nodes=electrical_nodes,
        circuit_components=circuit_components,
        voltage_vars=voltage_vars,
        current_vars=current_vars,
        state_vars=state_vars,
        state_derivatives=state_derivatives,
        input_vars=input_vars,
        ground_node=ground_node
    )
    
    A, B, solved_helpers, differential_equations = model.build_model()
    
    # Specific RLC circuit tests
    assert len(state_vars) == 2, "RLC circuit should have two state variables"
    assert len(input_vars) == 1, "RLC circuit should have one input variable"
    assert A.shape == (2, 2), "A should be 2x2 for RLC circuit"
    assert B.shape == (2, 1), "B should be 2x1 for RLC circuit" 