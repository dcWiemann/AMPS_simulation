import pytest
from amps_simulation.core.dae_model import DaeModel, ElectricalDaeModel
from amps_simulation.core.parser_networkx import ParserJson
from amps_simulation.core.components import Resistor, ElecJunction, PowerSwitch
from typing import Dict
import networkx as nx
import numpy as np
import sympy
from sympy import Derivative
from sympy.abc import t
import json


class SimpleDaeModel(DaeModel):
    """A simple concrete implementation of DaeModel for testing purposes."""
    
    def evaluate(self, t: float, states: Dict[str, float], inputs: Dict[str, float]) -> None:
        """Implement a simple DAE model: dx/dt = -x + u, y = x."""
        x = states.get('x', 0.0)
        u = inputs.get('u', 0.0)
        
        self.derivatives['x'] = -x + u
        self.outputs['y'] = x


def test_dae_model_initialization():
    """Test that a DAE model initializes with empty dictionaries."""
    G = nx.Graph()  # Create an empty graph for the simple model
    model = SimpleDaeModel(G)
    assert model.derivatives == {}
    assert model.outputs == {}


def test_dae_model_getters():
    """Test the getter methods for derivatives and outputs."""
    G = nx.Graph()  # Create an empty graph for the simple model
    model = SimpleDaeModel(G)
    states = {'x': 1.0}
    inputs = {'u': 2.0}
    
    model.evaluate(t=0.0, states=states, inputs=inputs)
    
    derivatives = model.get_derivatives()
    outputs = model.get_outputs()
    
    assert derivatives['x'] == 1.0
    assert outputs['y'] == 1.0


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
    return parser.parse(circuit_json)


def test_electrical_dae_model_initialization():
    """Test initialization of ElectricalDaeModel."""
    G = create_test_circuit()
    model = ElectricalDaeModel(G)
    assert isinstance(model.graph, nx.Graph)
    assert model.derivatives == {}
    assert model.outputs == {}


def test_compute_incidence_matrix():
    """Test computation of incidence matrix."""
    G = create_test_circuit()
    model = ElectricalDaeModel(G)
    
    incidence_matrix = model.compute_incidence_matrix()
    
    # Check matrix dimensions (3 nodes including ground, 2 components)
    assert incidence_matrix.shape == (3, 2)


def test_compute_kcl_equations():
    """Test computation of KCL equations."""
    G = create_test_circuit()
    model = ElectricalDaeModel(G)
    
    kcl_equations = model.compute_kcl_equations()
    
    # Check that we get 2 equations (one for each non-ground node)
    assert len(kcl_equations) == 2
    
    # Check that equations are symbolic expressions
    assert all(isinstance(eq, sympy.Basic) for eq in kcl_equations)


def test_compute_kvl_equations():
    """Test computation of KVL equations."""
    G = create_test_circuit()
    model = ElectricalDaeModel(G)
    
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
    G = parser.parse(circuit_json)
    model = ElectricalDaeModel(G)
    
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
    G = parser.parse(circuit_json)
    model = ElectricalDaeModel(G)
    
    switch_eqs = model.compute_switch_equations()
    print("switch_eqs off:", switch_eqs)
    assert switch_eqs == [sympy.symbols('i_S2')]

    for edge in G.edges(data=True):
        if isinstance(edge[2]['component'], PowerSwitch):
            edge[2]['component'].is_on = True

    switch_eqs = model.compute_switch_equations()
    print("switch_eqs on: ", switch_eqs)
    assert switch_eqs == [sympy.symbols('v_S2')]
    # Check that equations are symbolic expressions
    assert all(isinstance(eq, sympy.Basic) for eq in switch_eqs)


def test_compute_circuit_vars():
    """Test solving of circuit variables using DaeModel_circuit_var_solution.json."""
    # Load and analyze the circuit from DaeModel_circuit_var_solution.json
    parser = ParserJson()
    with open('test_data/DaeModel_circuit_var_solution.json', 'r') as f:
        circuit_json = json.load(f)
    G = parser.parse(circuit_json)
    for edge in G.edges(data=True):
        if isinstance(edge[2]['component'], PowerSwitch):
            edge[2]['component'].is_on = False

    model = ElectricalDaeModel(G)

    circuit_vars = model.compute_circuit_vars()
    print("circuit_vars: ", circuit_vars)

    # Check that the solution is a dictionary
    assert isinstance(circuit_vars, dict)
    
    # Define resistance value
    R = 10
    V_1 = sympy.symbols('V_1')
    v_V1 = sympy.symbols('v_V1')
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
        actual_expr = circuit_vars.get(var)
        # Use sympy.simplify to check if the difference is zero
        assert sympy.simplify(actual_expr - expr) == 0, f"Mismatch for {var}: expected {expr}, got {actual_expr}"


def test_print_dae_model_components():
    """Test function to print out key components of the DAE model for inspection."""
    # Load and analyze the circuit from DaeModel_meters.json
    print("\n=== Circuit Analysis from DaeModel_meters.json ===")
    parser = ParserJson()
    with open('test_data/DaeModel_meters.json', 'r') as f:
        circuit_json = json.load(f)
    G = parser.parse(circuit_json)
    model = ElectricalDaeModel(G)
    model.initialize()
    
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
    print("\ncircuit_vars: ", model.circuit_vars)
    print("\nderivatives: ", model.derivatives)
    print("\noutputs: ", model.outputs)


def test_kcl_equations_exclude_ground():
    """Test that KCL equations exclude the ground node equation."""
    # Load and analyze the circuit from DaeModel_kcl_minimal.json
    parser = ParserJson()
    with open('test_data/DaeModel_kcl_minimal.json', 'r') as f:
        circuit_json = json.load(f)
    G = parser.parse(circuit_json)
    model = ElectricalDaeModel(G)
    
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
    G = parser.parse(circuit_json)
    model = ElectricalDaeModel(G)
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
    v_V1 = sympy.symbols('v_V1')

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
