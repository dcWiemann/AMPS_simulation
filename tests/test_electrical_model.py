import pytest
import json
import os
import sympy as sp
from amps_simulation.core.electrical_model import ElectricalModel
from amps_simulation.core.parser import ParserJson
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

@pytest.fixture
def mock_rlc_circuit():
    """Create a simple RLC circuit for testing."""
    # Load and parse the circuit
    circuit_data = load_test_file("test_rlc.json")
    parser = ParserJson()
    electrical_nodes, circuit_components = parser.parse(circuit_data)
    
    # Create simulation instance to get variables
    simulation = Simulation(electrical_nodes, circuit_components)
    simulation.initialize()
    
    return {
        "electrical_nodes": electrical_nodes,
        "circuit_components": circuit_components,
        "voltage_vars": simulation.voltage_vars,
        "current_vars": simulation.current_vars,
        "state_vars": simulation.state_vars,
        "state_derivatives": simulation.state_derivatives,
        "input_vars": simulation.input_vars,
        "ground_node": simulation.ground_node
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
    solved_helpers, differential_equations = model.build_model()
    
    # Basic validation
    assert isinstance(solved_helpers, dict), "solved_helpers should be a dictionary"
    assert isinstance(differential_equations, dict), "differential_equations should be a dictionary"
    
    # For RLC circuit:
    assert len(differential_equations) == 2, "Should have 2 state variables for RLC circuit"

@pytest.mark.parametrize("test_file", TEST_FILES)
def test_electrical_model_with_real_circuits(test_file):
    """Test ElectricalModel with real circuit data."""
    # Load and parse circuit data
    circuit_data = load_test_file(test_file)
    parser = ParserJson()
    electrical_nodes, circuit_components = parser.parse(circuit_data)
    
    # Create simulation instance to get variables
    simulation = Simulation(electrical_nodes, circuit_components)
    simulation.initialize()
    
    # Create and test ElectricalModel
    model = ElectricalModel(
        electrical_nodes=electrical_nodes,
        circuit_components=circuit_components,
        voltage_vars=simulation.voltage_vars,
        current_vars=simulation.current_vars,
        state_vars=simulation.state_vars,
        state_derivatives=simulation.state_derivatives,
        input_vars=simulation.input_vars,
        ground_node=simulation.ground_node
    )
    
    solved_helpers, differential_equations = model.build_model()
    
    # Validate outputs
    assert isinstance(solved_helpers, dict), f"solved_helpers should be a dictionary for {test_file}"
    assert isinstance(differential_equations, dict), f"differential_equations should be a dictionary for {test_file}"

def test_electrical_model_specific_rlc_circuit():
    """Test specific properties of the RLC circuit model."""
    # Load and parse RLC circuit
    circuit_data = load_test_file("test_rlc.json")
    parser = ParserJson()
    electrical_nodes, circuit_components = parser.parse(circuit_data)
    
    # Create simulation instance to get variables
    simulation = Simulation(electrical_nodes, circuit_components)
    simulation.initialize()
    
    # Create model
    model = ElectricalModel(
        electrical_nodes=electrical_nodes,
        circuit_components=circuit_components,
        voltage_vars=simulation.voltage_vars,
        current_vars=simulation.current_vars,
        state_vars=simulation.state_vars,
        state_derivatives=simulation.state_derivatives,
        input_vars=simulation.input_vars,
        ground_node=simulation.ground_node
    )
    
    solved_helpers, differential_equations = model.build_model()
    
    # Specific RLC circuit tests
    assert len(model.state_vars) == 2, "RLC circuit should have two state variables"
    assert len(model.input_vars) == 1, "RLC circuit should have one input variable"
    assert len(differential_equations) == 2, "Should have 2 state variables for RLC circuit" 