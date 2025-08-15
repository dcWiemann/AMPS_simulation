import pytest
import json
import networkx as nx
import sympy
from amps_simulation.core.electrical_graph import ElectricalGraph
from amps_simulation.core.parser import ParserJson
from amps_simulation.core.components import PowerSwitch


def create_simple_test_circuit():
    """Create a simple test circuit with 3 nodes and 2 resistors."""
    circuit_json = {
        "nodes": [
            {
                "id": "R1",
                "data": {
                    "componentType": "resistor",
                    "value": 10.0
                }
            },
            {
                "id": "R2",
                "data": {
                    "componentType": "resistor", 
                    "value": 20.0
                }
            },
            {
                "id": "GND",
                "data": {
                    "componentType": "ground"
                }
            }
        ],
        "edges": [
            {
                "source": "R1",
                "target": "R2",
                "sourceHandle": "1",
                "targetHandle": "0"
            },
            {
                "source": "R2",
                "target": "GND",
                "sourceHandle": "1",
                "targetHandle": "0"
            }
        ]
    }
    
    parser = ParserJson()
    graph, control_graph = parser.parse(circuit_json)
    return graph


def test_electrical_graph_initialization():
    """Test basic initialization of ElectricalGraph."""
    graph = create_simple_test_circuit()
    electrical_graph = ElectricalGraph(graph)
    
    # Check initial state
    assert electrical_graph.graph is graph
    assert electrical_graph.initialized is False
    assert electrical_graph.incidence_matrix is None
    assert electrical_graph.junction_voltage_var_list is None
    assert electrical_graph.component_current_var_list is None
    assert electrical_graph.component_voltage_var_list is None
    assert electrical_graph.switch_list is None


def test_electrical_graph_initialize():
    """Test initialization method of ElectricalGraph."""
    graph = create_simple_test_circuit()
    electrical_graph = ElectricalGraph(graph)
    
    # Initialize
    electrical_graph.initialize()
    
    # Check that initialization completed
    assert electrical_graph.initialized is True
    assert electrical_graph.incidence_matrix is not None
    assert electrical_graph.junction_voltage_var_list is not None
    assert electrical_graph.component_current_var_list is not None
    assert electrical_graph.component_voltage_var_list is not None
    assert electrical_graph.switch_list is not None


def test_compute_incidence_matrix():
    """Test computation of incidence matrix."""
    graph = create_simple_test_circuit()
    electrical_graph = ElectricalGraph(graph)
    
    incidence_matrix = electrical_graph.compute_incidence_matrix()
    
    # Check matrix dimensions (3 nodes including ground, 2 components)
    assert incidence_matrix.shape == (3, 2)
    
    # Check that matrix is symbolic
    assert hasattr(incidence_matrix, 'subs')  # SymPy Matrix property
    
    # Check matrix values (should be -1, 0, or 1)
    import numpy as np
    matrix_array = np.array(incidence_matrix.tolist(), dtype=float)
    unique_values = set(matrix_array.flatten())
    assert unique_values.issubset({-1.0, 0.0, 1.0})


def test_variable_lists():
    """Test variable lists computation."""
    graph = create_simple_test_circuit()
    electrical_graph = ElectricalGraph(graph)
    
    junction_voltage_vars, component_current_vars, component_voltage_vars = electrical_graph.variable_lists()
    
    # Check that lists have correct lengths
    assert len(junction_voltage_vars) == 3  # 3 nodes
    assert len(component_current_vars) == 2  # 2 components
    assert len(component_voltage_vars) == 2  # 2 components
    
    # Check that variables are symbolic
    assert all(isinstance(var, (sympy.Basic, int)) for var in junction_voltage_vars)
    assert all(isinstance(var, sympy.Basic) for var in component_current_vars)
    assert all(isinstance(var, sympy.Basic) for var in component_voltage_vars)


def test_find_switches_empty():
    """Test finding switches in circuit with no switches."""
    graph = create_simple_test_circuit()
    electrical_graph = ElectricalGraph(graph)
    
    switches = electrical_graph.find_switches()
    
    # Should find no switches
    assert len(switches) == 0
    assert switches == []


def test_find_switches_with_switch():
    """Test finding switches in circuit with switches."""
    # Load circuit with switches
    parser = ParserJson()
    with open('test_data/DaeModel_meters.json', 'r') as f:
        circuit_json = json.load(f)
    graph, _ = parser.parse(circuit_json)
    
    electrical_graph = ElectricalGraph(graph)
    switches = electrical_graph.find_switches()
    
    # Should find switches
    assert len(switches) > 0
    assert all(isinstance(switch, PowerSwitch) for switch in switches)


def test_initialize_sets_all_properties():
    """Test that initialize() properly sets all properties."""
    graph = create_simple_test_circuit()
    electrical_graph = ElectricalGraph(graph)
    
    # Initialize
    electrical_graph.initialize()
    
    # Verify all properties are set
    assert electrical_graph.incidence_matrix is not None
    assert electrical_graph.junction_voltage_var_list is not None
    assert electrical_graph.component_current_var_list is not None 
    assert electrical_graph.component_voltage_var_list is not None
    assert electrical_graph.switch_list is not None
    assert electrical_graph.initialized is True
    
    # Verify properties have expected types
    assert hasattr(electrical_graph.incidence_matrix, 'shape')  # Matrix-like
    assert isinstance(electrical_graph.junction_voltage_var_list, list)
    assert isinstance(electrical_graph.component_current_var_list, list)
    assert isinstance(electrical_graph.component_voltage_var_list, list)
    assert isinstance(electrical_graph.switch_list, list)


def test_multiple_initializations():
    """Test that multiple calls to initialize() don't cause issues."""
    graph = create_simple_test_circuit()
    electrical_graph = ElectricalGraph(graph)
    
    # Initialize multiple times
    electrical_graph.initialize()
    first_incidence_matrix = electrical_graph.incidence_matrix
    
    electrical_graph.initialize()
    second_incidence_matrix = electrical_graph.incidence_matrix
    
    # Should get consistent results
    assert electrical_graph.initialized is True
    # Matrices should be equivalent (though potentially different objects)
    assert first_incidence_matrix.shape == second_incidence_matrix.shape


def test_electrical_graph_integration():
    """Test ElectricalGraph integration with more complex circuit."""
    # Load a more complex circuit
    parser = ParserJson()
    with open('test_data/DaeModel_circuit_var_solution.json', 'r') as f:
        circuit_json = json.load(f)
    graph, _ = parser.parse(circuit_json)
    
    electrical_graph = ElectricalGraph(graph)
    electrical_graph.initialize()
    
    # Verify initialization worked
    assert electrical_graph.initialized is True
    
    # Check dimensions make sense
    n_nodes = len(list(graph.nodes()))
    n_edges = len(list(graph.edges()))
    
    assert electrical_graph.incidence_matrix.shape[0] == n_nodes
    assert electrical_graph.incidence_matrix.shape[1] == n_edges
    assert len(electrical_graph.junction_voltage_var_list) == n_nodes
    assert len(electrical_graph.component_current_var_list) == n_edges
    assert len(electrical_graph.component_voltage_var_list) == n_edges