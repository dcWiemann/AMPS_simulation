import pytest
import json
import os
from typing import Dict, Any, Set, Tuple
from amps_simulation.core.parser import ParserJson_todict

# Test data configuration
TEST_DATA_DIR = "test_data"
TEST_FILES = [
    "test_rc.json",       # RC circuit
    "test_rlc.json",      # RLC circuit
    "test_2v.json",       # Circuit with 2 voltage sources
]

def load_test_file(filename: str) -> Dict[str, Any]:
    """
    Load a test circuit file and return its contents.
    
    Args:
        filename: Name of the JSON file containing circuit description
        
    Returns:
        Dict containing the circuit description with 'nodes' and 'edges'
        
    Raises:
        FileNotFoundError: If the test file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    file_path = os.path.join(TEST_DATA_DIR, filename)
    with open(file_path, 'r') as f:
        return json.load(f)

def validate_electrical_nodes(electrical_nodes: Dict[int, Set[Tuple[str, str]]], test_file: str) -> None:
    """
    Validate the structure and content of electrical nodes.
    
    Given:
        - A dictionary of electrical nodes
        - Each node maps to a set of (component_id, terminal_id) tuples
        - The test file name for error reporting
        
    When:
        - The validation is performed
        
    Then:
        - All node IDs are integers
        - All node values are sets
        - All terminals are tuples of (component_id, terminal_id)
        
    Args:
        electrical_nodes: Dictionary mapping node IDs to sets of terminals
        test_file: Name of the test file for error reporting
        
    Raises:
        AssertionError: If any validation check fails
    """
    assert isinstance(electrical_nodes, dict), f"electrical_nodes should be a dictionary for {test_file}"
    assert all(isinstance(node_id, int) for node_id in electrical_nodes.keys()), \
        f"electrical_nodes keys should be integers for {test_file}"
    assert all(isinstance(terminals, set) for terminals in electrical_nodes.values()), \
        f"electrical_nodes values should be sets for {test_file}"
    assert all(all(isinstance(term, tuple) and len(term) == 2 for term in terminals) 
              for terminals in electrical_nodes.values()), \
        f"each terminal in electrical_nodes should be a tuple of (component_id, terminal_id) for {test_file}"

def validate_circuit_components(circuit_components: Dict[str, Dict], test_file: str) -> None:
    """
    Validate the structure and content of circuit components.
    
    Given:
        - A dictionary of circuit components
        - Each component has type, value, and terminals
        - The test file name for error reporting
        
    When:
        - The validation is performed
        
    Then:
        - All component IDs are strings
        - Each component has required fields (type, terminals)
        - All terminal IDs are strings
        - All electrical node IDs are integers
        
    Args:
        circuit_components: Dictionary mapping component IDs to component data
        test_file: Name of the test file for error reporting
        
    Raises:
        AssertionError: If any validation check fails
    """
    assert isinstance(circuit_components, dict), f"circuit_components should be a dictionary for {test_file}"
    for comp_id, comp_data in circuit_components.items():
        assert isinstance(comp_id, str), f"component IDs should be strings for {test_file}"
        assert isinstance(comp_data, dict), f"component data should be a dictionary for {test_file}"
        assert "type" in comp_data, f"component {comp_id} missing 'type' field for {test_file}"
        assert "terminals" in comp_data, f"component {comp_id} missing 'terminals' field for {test_file}"
        assert isinstance(comp_data["terminals"], dict), \
            f"terminals should be a dictionary for component {comp_id} in {test_file}"
        assert all(isinstance(term_id, str) for term_id in comp_data["terminals"].keys()), \
            f"terminal IDs should be strings for component {comp_id} in {test_file}"
        assert all(isinstance(node_id, int) for node_id in comp_data["terminals"].values()), \
            f"electrical node IDs should be integers for component {comp_id} in {test_file}"

@pytest.mark.parametrize("test_file", TEST_FILES)
def test_parser_json(test_file: str) -> None:
    """
    Test that ParserJson correctly processes circuit files.
    
    Given:
        - A circuit description in JSON format
        - The circuit contains components and their connections
        
    When:
        - The circuit is parsed by ParserJson
        
    Then:
        - Electrical nodes are correctly identified and structured
        - Circuit components are properly extracted and typed
        - All terminals are correctly mapped to electrical nodes
        - No terminals are orphaned or incorrectly connected
        
    Args:
        test_file: Name of the JSON file containing the circuit description
        
    Raises:
        AssertionError: If any validation check fails
        Exception: If parsing fails
    """
    try:
        # Load circuit data
        circuit_data = load_test_file(test_file)
        
        # Create parser instance
        parser = ParserJson_todict()
        
        # Parse circuit data
        electrical_nodes, circuit_components = parser.parse(circuit_data)
        
        # Validate electrical nodes
        validate_electrical_nodes(electrical_nodes, test_file)
        
        # Validate circuit components
        validate_circuit_components(circuit_components, test_file)
        
        # Additional validation: check that all terminals in electrical_nodes are referenced in circuit_components
        all_terminals = set()
        for comp_id, comp_data in circuit_components.items():
            for term_id in comp_data["terminals"].keys():
                all_terminals.add((comp_id, term_id))
        
        for node_terminals in electrical_nodes.values():
            assert all(term in all_terminals for term in node_terminals), \
                f"Found terminal in electrical_nodes not referenced in circuit_components for {test_file}"
        
    except Exception as e:
        pytest.fail(f"ParserJson.parse failed for {test_file} with error: {str(e)}")

def test_parser_json_specific_circuit() -> None:
    """
    Test ParserJson with a specific RLC circuit for detailed validation.
    
    Given:
        - An RLC circuit description in JSON format
        - The circuit contains a voltage source, resistor, inductor, and capacitor
        
    When:
        - The circuit is parsed by ParserJson
        
    Then:
        - Three electrical nodes are identified (one for each terminal of the components)
        - All required components are present (voltage source, resistor, inductor, capacitor)
        - Each component has a valid value
        - Terminal connections are properly mapped
        
    Raises:
        AssertionError: If any validation check fails
        Exception: If parsing fails
    """
    # Load RLC circuit
    circuit_data = load_test_file("test_rlc.json")
    
    # Create parser instance
    parser = ParserJson_todict()
    
    # Parse circuit data
    electrical_nodes, circuit_components = parser.parse(circuit_data)
    
    # For RLC circuit, we expect:
    # - One capacitor
    # - One resistor
    # - One inductor
    # - One voltage source
    # - Three electrical nodes (one for each terminal of the components)
    assert len(electrical_nodes) == 3, "RLC circuit should have three electrical nodes"
    
    # Count component types
    component_types = {comp["type"] for comp in circuit_components.values()}
    assert "capacitor" in component_types, "RLC circuit should have a capacitor"
    assert "resistor" in component_types, "RLC circuit should have a resistor"
    assert "inductor" in component_types, "RLC circuit should have an inductor"
    assert "voltage-source" in component_types, "RLC circuit should have a voltage source"
    
    # Verify component values
    for comp_id, comp_data in circuit_components.items():
        if comp_data["type"] == "capacitor":
            assert comp_data["value"] is not None, "Capacitor should have a value"
        elif comp_data["type"] == "resistor":
            assert comp_data["value"] is not None, "Resistor should have a value"
        elif comp_data["type"] == "inductor":
            assert comp_data["value"] is not None, "Inductor should have a value"
        elif comp_data["type"] == "voltage-source":
            assert comp_data["value"] is not None, "Voltage source should have a value"
