import pytest
import json
import networkx as nx
import sympy
from amps_simulation.core.electrical_model import ElectricalModel
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


def test_electrical_model_initialization():
    """Test basic initialization of ElectricalModel."""
    graph = create_simple_test_circuit()
    electrical_model = ElectricalModel(graph)
    
    # Check initial state
    assert electrical_model.graph is graph
    assert electrical_model.initialized is False
    assert electrical_model.incidence_matrix is None
    assert electrical_model.junction_voltage_var_list is None
    assert electrical_model.component_current_var_list is None
    assert electrical_model.component_voltage_var_list is None
    assert electrical_model.switch_list is None


def test_electrical_model_initialize():
    """Test initialization method of ElectricalModel."""
    graph = create_simple_test_circuit()
    electrical_model = ElectricalModel(graph)
    
    # Initialize
    electrical_model.initialize()
    
    # Check that initialization completed
    assert electrical_model.initialized is True
    assert electrical_model.incidence_matrix is not None
    assert electrical_model.junction_voltage_var_list is not None
    assert electrical_model.component_current_var_list is not None
    assert electrical_model.component_voltage_var_list is not None
    assert electrical_model.switch_list is not None


def test_compute_incidence_matrix():
    """Test computation of incidence matrix."""
    graph = create_simple_test_circuit()
    electrical_model = ElectricalModel(graph)
    
    incidence_matrix = electrical_model.compute_incidence_matrix()
    
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
    electrical_model = ElectricalModel(graph)
    
    junction_voltage_vars, component_current_vars, component_voltage_vars = electrical_model.variable_lists()
    
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
    electrical_model = ElectricalModel(graph)
    
    switches = electrical_model.find_switches()
    
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
    
    electrical_model = ElectricalModel(graph)
    switches = electrical_model.find_switches()
    
    # Should find switches
    assert len(switches) > 0
    assert all(isinstance(switch, PowerSwitch) for switch in switches)


def test_initialize_sets_all_properties():
    """Test that initialize() properly sets all properties."""
    graph = create_simple_test_circuit()
    electrical_model = ElectricalModel(graph)
    
    # Initialize
    electrical_model.initialize()
    
    # Verify all properties are set
    assert electrical_model.incidence_matrix is not None
    assert electrical_model.junction_voltage_var_list is not None
    assert electrical_model.component_current_var_list is not None
    assert electrical_model.component_voltage_var_list is not None
    assert electrical_model.switch_list is not None
    assert electrical_model.state_vars is not None
    assert electrical_model.input_vars is not None
    assert electrical_model.output_vars is not None
    assert electrical_model.initialized is True
    
    # Verify properties have expected types
    assert hasattr(electrical_model.incidence_matrix, 'shape')  # Matrix-like
    assert isinstance(electrical_model.junction_voltage_var_list, list)
    assert isinstance(electrical_model.component_current_var_list, list)
    assert isinstance(electrical_model.component_voltage_var_list, list)
    assert isinstance(electrical_model.switch_list, list)
    assert isinstance(electrical_model.state_vars, list)
    assert isinstance(electrical_model.input_vars, list)
    assert isinstance(electrical_model.output_vars, list)


def test_multiple_initializations():
    """Test that multiple calls to initialize() don't cause issues."""
    graph = create_simple_test_circuit()
    electrical_model = ElectricalModel(graph)
    
    # Initialize multiple times
    electrical_model.initialize()
    first_incidence_matrix = electrical_model.incidence_matrix
    
    electrical_model.initialize()
    second_incidence_matrix = electrical_model.incidence_matrix
    
    # Should get consistent results
    assert electrical_model.initialized is True
    # Matrices should be equivalent (though potentially different objects)
    assert first_incidence_matrix.shape == second_incidence_matrix.shape


def test_electrical_model_integration():
    """Test ElectricalModel integration with more complex circuit."""
    # Load a more complex circuit
    parser = ParserJson()
    with open('test_data/DaeModel_circuit_var_solution.json', 'r') as f:
        circuit_json = json.load(f)
    graph, _ = parser.parse(circuit_json)
    
    electrical_model = ElectricalModel(graph)
    electrical_model.initialize()
    
    # Verify initialization worked
    assert electrical_model.initialized is True
    
    # Check dimensions make sense
    n_nodes = len(list(graph.nodes()))
    n_edges = len(list(graph.edges()))
    
    assert electrical_model.incidence_matrix.shape[0] == n_nodes
    assert electrical_model.incidence_matrix.shape[1] == n_edges
    assert len(electrical_model.junction_voltage_var_list) == n_nodes
    assert len(electrical_model.component_current_var_list) == n_edges
    assert len(electrical_model.component_voltage_var_list) == n_edges


# Tests for new programmatic API

def test_electrical_model_empty_constructor():
    """Test ElectricalModel can be created without graph parameter."""
    from amps_simulation.core.components import Component, ElecJunction
    Component.clear_registry()
    ElecJunction.clear_registry()

    em = ElectricalModel()
    assert em.graph is not None
    assert isinstance(em.graph, nx.MultiDiGraph)
    assert len(em.graph.nodes()) == 0
    assert len(em.graph.edges()) == 0


def test_add_node_basic():
    """Test adding nodes to empty ElectricalModel."""
    from amps_simulation.core.components import Component, ElecJunction
    Component.clear_registry()
    ElecJunction.clear_registry()

    em = ElectricalModel()

    # Add regular node
    em.add_node(1)
    assert 1 in em.graph.nodes()
    assert em.graph.nodes[1]['junction'].junction_id == 1
    assert em.graph.nodes[1]['junction'].is_ground == False

    # Add ground node
    em.add_node(0, is_ground=True)
    assert 0 in em.graph.nodes()
    assert em.graph.nodes[0]['junction'].junction_id == 0
    assert em.graph.nodes[0]['junction'].is_ground == True


def test_add_component_list_format():
    """Test adding components with list format terminals."""
    from amps_simulation.core.components import Component, Resistor, ElecJunction
    Component.clear_registry()
    ElecJunction.clear_registry()

    em = ElectricalModel()
    R1 = Resistor('R1', 10)

    # Add component with list format
    em.add_component(R1, [1, 0])

    # Check nodes were auto-created
    assert 1 in em.graph.nodes()
    assert 0 in em.graph.nodes()

    # Check edge was created
    assert em.graph.has_edge(1, 0)
    edge_data = em.graph.get_edge_data(1, 0)
    assert len(edge_data) == 1  # One edge between these nodes
    assert edge_data[0]['component'] == R1


def test_add_component_kwargs_format():
    """Test adding components with named terminal arguments."""
    from amps_simulation.core.components import Component, Capacitor, ElecJunction
    Component.clear_registry()
    ElecJunction.clear_registry()

    em = ElectricalModel()
    C1 = Capacitor('C1', 1e-4)

    # Add component with named terminals
    em.add_component(C1, p=3, n=0)

    # Check nodes were auto-created
    assert 3 in em.graph.nodes()
    assert 0 in em.graph.nodes()

    # Check edge was created
    assert em.graph.has_edge(3, 0)
    edge_data = em.graph.get_edge_data(3, 0)
    assert edge_data[0]['component'] == C1


def test_component_positional_constructors():
    """Test component constructors with positional arguments."""
    from amps_simulation.core.components import Component, Resistor, Capacitor, Inductor, VoltageSource, Diode
    Component.clear_registry()

    # Test Resistor
    R1 = Resistor('R1', 10)
    assert R1.comp_id == 'R1'
    assert R1.resistance == 10

    # Test Capacitor
    C1 = Capacitor('C1', 1e-4)
    assert C1.comp_id == 'C1'
    assert C1.capacitance == 1e-4

    # Test Inductor
    L1 = Inductor('L1', 1e-2)
    assert L1.comp_id == 'L1'
    assert L1.inductance == 1e-2

    # Test VoltageSource
    V1 = VoltageSource('V1', 12)
    assert V1.comp_id == 'V1'
    assert V1.voltage == 12

    # Test Diode
    D1 = Diode('D1')
    assert D1.comp_id == 'D1'
    assert D1.get_comp_eq(False) == D1.current_var  # Default behavior when off


def test_programmatic_api_example():
    """Test complete example matching API documentation."""
    from amps_simulation.core.components import Component, Resistor, Capacitor, VoltageSource, Inductor, Diode, ElecJunction
    Component.clear_registry()
    ElecJunction.clear_registry()

    # Create model and components
    em = ElectricalModel()

    R1 = Resistor('R1', 10)
    C1 = Capacitor('C1', 1e-4)
    V1 = VoltageSource('V1', 12)
    L1 = Inductor('L1', 1e-2)
    D1 = Diode('D1')

    # Build circuit
    em.add_node(0, is_ground=True)
    em.add_component(V1, [1, 0])
    em.add_component(L1, [2, 3])
    em.add_component(R1, [3, 0])
    em.add_component(C1, p=3, n=0)
    em.add_component(D1, [0, 2])

    # Verify structure
    assert len(em.graph.nodes()) == 4  # nodes 0, 1, 2, 3
    assert len(em.graph.edges()) == 5  # 5 components

    # Check ground node
    assert em.graph.nodes[0]['junction'].is_ground == True

    # Check components are connected correctly
    assert em.graph.has_edge(1, 0)  # V1
    assert em.graph.has_edge(2, 3)  # L1
    assert em.graph.has_edge(3, 0)  # R1
    assert em.graph.has_edge(3, 0)  # C1 (parallel to R1)
    assert em.graph.has_edge(0, 2)  # D1


def test_backward_compatibility():
    """Test that existing graph-based constructor still works."""
    from amps_simulation.core.components import Component, ElecJunction
    Component.clear_registry()
    ElecJunction.clear_registry()

    # Create graph manually (old way)
    graph = create_simple_test_circuit()
    em = ElectricalModel(graph)

    # Should still work
    assert em.graph is not None
    assert len(em.graph.nodes()) > 0
    assert len(em.graph.edges()) > 0
