import pytest
from amps_simulation.core.dae_system import DaeSystem, ElectricalDaeSystem
from amps_simulation.core.electrical_model import ElectricalModel
from amps_simulation.core.parser import ParserJson
from amps_simulation.core.components import Resistor, ElecJunction, PowerSwitch
from typing import Dict
import networkx as nx
import numpy as np
import sympy
from sympy import Derivative
from sympy.abc import t
import json


class SimpleDaeSystem(DaeSystem):
    """A simple concrete implementation of DaeSystem for testing purposes."""
    
    def evaluate(self, t: float, states: Dict[str, float], inputs: Dict[str, float]) -> None:
        """Implement a simple DAE system: dx/dt = -x + u, y = x."""
        x = states.get('x', 0.0)
        u = inputs.get('u', 0.0)
        
        self.derivatives['x'] = -x + u
        self.output_eqs['y'] = x


def test_dae_system_initialization():
    """Test that a DAE system initializes with empty dictionaries."""
    G = nx.Graph()  # Create an empty graph for the simple system
    model = SimpleDaeSystem(G)
    assert model.derivatives == {}
    assert model.output_eqs == {}


def test_dae_system_getters():
    """Test the getter methods for derivatives and outputs."""
    G = nx.Graph()  # Create an empty graph for the simple system
    model = SimpleDaeSystem(G)
    states = {'x': 1.0}
    inputs = {'u': 2.0}
    
    model.evaluate(t=0.0, states=states, inputs=inputs)
    
    derivatives = model.get_derivatives()
    output_eqs = model.get_outputs()
    
    assert derivatives['x'] == 1.0
    assert output_eqs['y'] == 1.0


def create_test_circuit():
    """Create a simple test circuit with 3 nodes and 2 resistors using ParserJson."""
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


def load_test_circuit(filename):
    """Helper function to load test circuit from JSON file."""
    with open(f"test_data/{filename}", "r") as f:
        return json.load(f)


def test_electrical_dae_system_initialization():
    """Test initialization of ElectricalDaeSystem."""
    G = create_test_circuit()
    electrical_model = ElectricalModel(G)
    model = ElectricalDaeSystem(electrical_model)
    assert isinstance(model.graph, nx.Graph)
    assert model.derivatives == {}
    assert model.output_eqs == {}


def test_compute_incidence_matrix():
    """Test computation of incidence matrix."""
    G = create_test_circuit()
    electrical_model = ElectricalModel(G)
    
    incidence_matrix = electrical_model.compute_incidence_matrix()
    
    # Check matrix dimensions (3 nodes including ground, 2 components)
    assert incidence_matrix.shape == (3, 2)


def test_compute_kcl_equations():
    """Test computation of KCL equations."""
    G = create_test_circuit()
    electrical_model = ElectricalModel(G)
    model = ElectricalDaeSystem(electrical_model)
    
    kcl_equations = model.compute_kcl_equations()
    
    # Check that we get 2 equations (one for each non-ground node)
    assert len(kcl_equations) == 2
    
    # Check that equations are symbolic expressions
    assert all(isinstance(eq, sympy.Basic) for eq in kcl_equations)


def test_compute_kvl_equations():
    """Test computation of KVL equations."""
    G = create_test_circuit()
    electrical_model = ElectricalModel(G)
    model = ElectricalDaeSystem(electrical_model)
    
    kvl_equations = model.compute_kvl_equations()
    
    # Check that we get 2 equations (one for each component)
    assert len(kvl_equations) == 2
    
    # Check that equations are symbolic expressions
    assert all(isinstance(eq, sympy.Basic) for eq in kvl_equations)


def test_compute_static_component_equations():
    """Test computation of resistance equations."""
    parser = ParserJson()
    with open('test_data/DaeModel_meters.json', 'r') as f:
        circuit_json = json.load(f)
    G, _ = parser.parse(circuit_json)
    electrical_model = ElectricalModel(G)
    model = ElectricalDaeSystem(electrical_model)
    
    static_eqs = model.compute_static_component_equations()
    
    # Check that we get 2 equations (one for each resistor)
    assert len(static_eqs) == 5
    
    # Check that equations are symbolic expressions
    assert all(isinstance(eq, sympy.Basic) for eq in static_eqs)



def test_compute_switch_equations():
    """Test computation of switch equations."""
    parser = ParserJson()
    with open('test_data/DaeModel_meters.json', 'r') as f:
        circuit_json = json.load(f)
    G, _ = parser.parse(circuit_json)
    electrical_model = ElectricalModel(G)
    model = ElectricalDaeSystem(electrical_model)
    
    switch_eqs = model.compute_switch_equations()
    print("switch_eqs off:", switch_eqs)
    assert switch_eqs == [sympy.symbols('i_S2')]

    for edge in G.edges(data=True):
        if isinstance(edge[2]['component'], PowerSwitch):
            if edge[2].get('sim_info'):
                edge[2]['sim_info'].value = True

    switch_eqs = model.compute_switch_equations()
    print("switch_eqs on: ", switch_eqs)
    assert switch_eqs == [sympy.symbols('v_S2')]
    # Check that equations are symbolic expressions
    assert all(isinstance(eq, sympy.Basic) for eq in switch_eqs)


def test_compute_circuit_equations():
    """Test solving of circuit variables using DaeModel_circuit_var_solution.json."""
    # Load and analyze the circuit from DaeModel_circuit_var_solution.json
    parser = ParserJson()
    with open('test_data/DaeModel_circuit_var_solution.json', 'r') as f:
        circuit_json = json.load(f)
    G, _ = parser.parse(circuit_json)
    for edge in G.edges(data=True):
        if isinstance(edge[2]['component'], PowerSwitch):
            if edge[2].get('sim_info'):
                edge[2]['sim_info'].value = False

    electrical_model = ElectricalModel(G)
    model = ElectricalDaeSystem(electrical_model)

    circuit_eqs = model.compute_circuit_equations()
    print("circuit_eqs: ", circuit_eqs)

    # Check that the solution is a dictionary
    assert isinstance(circuit_eqs, dict)
    
    # Define resistance value
    R = 10
    V_1 = sympy.symbols('V_1')
    v_V1 = sympy.Function('v_V1')(t)
    i_R1 = sympy.symbols('i_R1')
    i_C1 = sympy.symbols('i_C1')
    v_L1 = sympy.symbols('v_L1')
    v_C1 = sympy.Function('v_C1')(t)
    i_L1 = sympy.Function('i_L1')(t)

    # Check that the solution contains expected symbolic relationships
    expected_relationships = {
        V_1: v_V1,
        i_R1: -(1/R) * v_C1 + (1/R) * v_V1,
        i_C1: -i_L1 - (1/R) * v_C1 + (1/R) * v_V1,
        v_L1: v_C1 
    }
    for var, expr in expected_relationships.items():
        actual_expr = circuit_eqs.get(var)
        # Use sympy.simplify to check if the difference is zero
        assert sympy.simplify(actual_expr - expr) == 0, f"Mismatch for {var}: expected {expr}, got {actual_expr}"


def test_print_dae_model_components():
    """Test function to print out key components of the DAE model for inspection."""
    # Load and analyze the circuit from DaeModel_meters.json
    print("\n=== Circuit Analysis from DaeModel_meters.json ===")
    parser = ParserJson()
    with open('test_data/DaeModel_meters.json', 'r') as f:
        circuit_json = json.load(f)
    G, _ = parser.parse(circuit_json)
    electrical_model = ElectricalModel(G)
    model = ElectricalDaeSystem(electrical_model)
    model.initialize()
    for _, _, edge_data in model.graph.edges(data=True):
        if isinstance(edge_data.get('component'), PowerSwitch) and edge_data.get('sim_info'):
            edge_data['sim_info'].value = True
    model.update_switch_states()
    
    # print attributes of the model
    print("\nstate_vars: ", model.state_vars)
    print("\noutput_vars: ", model.output_vars)
    print("\ninput_vars: ", model.input_vars)
    print("\njunction_voltage_var_list: ", model.junction_voltage_var_list)
    print("\ncomponent_current_var_list: ", model.component_current_var_list)
    print("\ncomponent_voltage_var_list: ", model.component_voltage_var_list)
    print("\nincidence_matrix: ", model.incidence_matrix)
    print("\nkcl_eqs: ", model.kcl_eqs)
    print("\nkvl_eqs: ", model.kvl_eqs)
    print("\nstatic_eqs: ", model.static_eqs)
    print("\nswitch_eqs: ", model.switch_eqs)
    print("\ncircuit_eqs: ", model.circuit_eqs)
    print("\nderivatives: ", model.derivatives)
    print("\noutput_eqs: ", model.output_eqs)



def test_compute_state_space_model():
    """Test computation of state-space matrices for a simple RLC circuit."""
    circuit_data = load_test_circuit("test_rlc.json")
    parser = ParserJson()
    graph, _ = parser.parse(circuit_data)

    electrical_model = ElectricalModel(graph)
    model = ElectricalDaeSystem(electrical_model)
    model.initialize()

    derivatives = model.derivatives
    output_eqs = model.output_eqs

    sorted_derivatives = model._sort_derivatives_by_state_vars(derivatives)
    sorted_output_eqs = model._sort_output_eqs_by_output_vars(output_eqs)

    A, B, C, D = model.compute_state_space_model(sorted_derivatives, sorted_output_eqs)

    n_states = len(model.state_vars)
    n_inputs = len(model.input_vars)
    n_outputs = len(model.output_vars)

    assert A.shape == (n_states, n_states)
    assert B.shape == (n_states, n_inputs)
    assert C.shape == (n_outputs, n_states)
    assert D.shape == (n_outputs, n_inputs)

    assert all(isinstance(expr, sympy.Basic) for expr in A)
    assert all(isinstance(expr, sympy.Basic) for expr in B)
    assert all(isinstance(expr, sympy.Basic) for expr in C)
    assert all(isinstance(expr, sympy.Basic) for expr in D)


def test_update_ode_returns_numeric_matrices_and_caches():
    """Test update_ode returns numeric matrices and caches results."""
    circuit_data = load_test_circuit("test_rlc.json")
    parser = ParserJson()
    graph, _ = parser.parse(circuit_data)

    electrical_model = ElectricalModel(graph)
    model = ElectricalDaeSystem(electrical_model)
    model.initialize()

    x = np.zeros(len(model.state_vars))
    u = np.zeros(len(model.input_vars))

    A1, B1, C1, D1 = model.update_ode([], x, u)

    assert isinstance(A1, np.ndarray)
    assert isinstance(B1, np.ndarray)
    assert isinstance(C1, np.ndarray)
    assert isinstance(D1, np.ndarray)

    n_states = len(model.state_vars)
    n_inputs = len(model.input_vars)
    n_outputs = len(model.output_vars)
    assert A1.shape == (n_states, n_states)
    assert B1.shape == (n_states, n_inputs)
    assert C1.shape == (n_outputs, n_states)
    assert D1.shape == (n_outputs, n_inputs)

    assert len(model.switchmap) == 1
    A2, B2, C2, D2 = model.update_ode([], x, u)
    assert len(model.switchmap) == 1

    assert np.allclose(A1, A2)
    assert np.allclose(B1, B2)
    assert np.allclose(C1, C2)
    assert np.allclose(D1, D2)

def test_kcl_equations_exclude_ground():
    """Test that KCL equations exclude the ground node equation."""
    # Load and analyze the circuit from DaeModel_kcl_minimal.json
    parser = ParserJson()
    with open('test_data/DaeModel_kcl_minimal.json', 'r') as f:
        circuit_json = json.load(f)
    G, _ = parser.parse(circuit_json)
    electrical_model = ElectricalModel(G)
    model = ElectricalDaeSystem(electrical_model)
    
    # Get KCL equations
    kcl_equations = model.compute_kcl_equations()
    
    # Verify that we only have one equation (for the non-ground node)
    assert len(kcl_equations) == 1, "Should have only one KCL equation (excluding ground)"
    
    # Verify that the equation contains the resistor current
    assert any(var in kcl_equations[0].free_symbols for var in [sympy.symbols('i_R8'), sympy.symbols('i_R9')]), "KCL equation should contain the resistor current"


def test_compute_derivatives():
    """Test computation of derivatives of state variables using DaeModel_circuit_var_solution.json."""
    # Load and analyze the circuit from DaeModel_circuit_var_solution.json
    parser = ParserJson()
    with open('test_data/DaeModel_circuit_var_solution.json', 'r') as f:
        circuit_json = json.load(f)
    G, _ = parser.parse(circuit_json)
    electrical_model = ElectricalModel(G)
    model = ElectricalDaeSystem(electrical_model)
    model.initialize()
    
    derivatives = model.compute_derivatives()
    print("derivatives: ", derivatives)
    
    # Check that derivatives are computed
    assert len(derivatives) == 2
    
    # Check that derivatives are symbolic expressions
    assert all(isinstance(derivative, sympy.Basic) for derivative in derivatives)
    
    # Define the component values
    R1 = 10.0
    L1 = 0.001
    C1 = 0.002   
    # Use the correct symbolic variables with time dependency
    i_L1 = sympy.Function('i_L1')(t)
    v_C1 = sympy.Function('v_C1')(t)
    v_V1 = sympy.Function('v_V1')(t)

    expected_derivatives = [
        (v_C1.diff(t), (-i_L1*R1 - v_C1 + v_V1)/(C1*R1)),
        (i_L1.diff(t), v_C1/L1)
    ]
    print("expected_derivatives: ", expected_derivatives)

    for expected in expected_derivatives:
        # Extract the left and right sides of the equality
        for derivative in derivatives:
            if isinstance(derivative, sympy.Eq):
                lhs, rhs = derivative.lhs, derivative.rhs
                # Check if the left-hand side matches the expected derivative
                if sympy.simplify(lhs - expected[0]) == 0:
                    # Check if the right-hand side matches the expected expression
                    assert sympy.simplify(rhs - expected[1]) == 0
                    break
        else:
            # If no match was found, the test should fail
            assert False, f"No matching derivative found for {expected}"
