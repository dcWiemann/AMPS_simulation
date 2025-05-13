import pytest
from amps_simulation.core.dae_model import DaeModel, ElectricalDaeModel
from amps_simulation.core.parser_networkx import ParserJson
from amps_simulation.core.components import Resistor, ElecJunction
from typing import Dict
import networkx as nx
import numpy as np
import sympy
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
    
    incidence_matrix, junction_vars, comp_current_vars, comp_voltage_vars = model.compute_incidence_matrix()
    
    # Check matrix dimensions (3 nodes including ground, 2 components)
    assert incidence_matrix.shape == (3, 2)
    
    # Check node variables (should be voltage variables for all nodes)
    assert len(junction_vars) == 3
    assert all(isinstance(var, sympy.Basic) for var in junction_vars)
    
    # Check component variables (should be current variables for resistors)
    assert len(comp_current_vars) == 2
    assert all(isinstance(var, sympy.Basic) for var in comp_current_vars)
    
    # Check component voltage variables
    assert len(comp_voltage_vars) == 2
    assert all(isinstance(var, sympy.Basic) for var in comp_voltage_vars)


def test_compute_kcl_equations():
    """Test computation of KCL equations."""
    G = create_test_circuit()
    model = ElectricalDaeModel(G)
    
    kcl_equations = model.compute_kcl_equations()
    
    # Check that we get 3 equations (one for each node)
    assert len(kcl_equations) == 3
    
    # Check that equations are symbolic expressions
    assert all(isinstance(eq, str) for eq in kcl_equations)


def test_compute_kvl_equations():
    """Test computation of KVL equations."""
    G = create_test_circuit()
    model = ElectricalDaeModel(G)
    
    kvl_equations = model.compute_kvl_equations()
    
    # Check that we get 2 equations (one for each component)
    assert len(kvl_equations) == 2
    
    # Check that equations are symbolic expressions
    assert all(isinstance(eq, str) for eq in kvl_equations)


def test_compute_resistance_equations():
    """Test computation of resistance equations."""
    G = create_test_circuit()
    model = ElectricalDaeModel(G)
    
    R_eqs = model.compute_resistance_equations()
    
    # Check that we get 2 equations (one for each resistor)
    assert len(R_eqs) == 2
    
    # Check that equations are strings
    assert all(isinstance(eq, str) for eq in R_eqs)
    
    # Check that equations follow the form v = i*R
    for (source, target, data), eq in zip(G.edges(data=True), R_eqs):
        component = data['component']
        assert isinstance(component.resistance, float)
        assert component.resistance in [10.0, 20.0]  # Values from our test circuit 


def test_print_dae_model_components():
    """Test function to print out key components of the DAE model for inspection."""
    # Load and analyze the circuit from DaeModel.json
    print("\n=== Circuit Analysis from DaeModel.json ===")
    parser = ParserJson()
    with open('test_data/DaeModel.json', 'r') as f:
        circuit_json = json.load(f)
    G = parser.parse(circuit_json)
    model = ElectricalDaeModel(G)
    
    # Get and print incidence matrix and variables
    inc_matrix, junction_vars, comp_current_vars, comp_voltage_vars = model.compute_incidence_matrix()
    print("\nIncidence Matrix:")
    print(inc_matrix)
    print("\nJunction Variables (Voltages):")
    for i, var in enumerate(junction_vars):
        print(f"V{i}: {var}")
    
    print("\nComponent Current Variables:")
    for i, var in enumerate(comp_current_vars):
        print(f"I{i}: {var}")
    
    print("\nComponent Voltage Variables:")
    for i, var in enumerate(comp_voltage_vars):
        print(f"V{i}: {var}")
    
    # Get and print equations
    kcl_eqs = model.compute_kcl_equations()
    kvl_eqs = model.compute_kvl_equations()
    r_eqs = model.compute_resistance_equations()
    
    print("\nKCL Equations:")
    for i, eq in enumerate(kcl_eqs):
        print(f"KCL {i}: {eq}")
    
    print("\nKVL Equations:")
    for i, eq in enumerate(kvl_eqs):
        print(f"KVL {i}: {eq}")
    
    print("\nResistance Equations:")
    for i, eq in enumerate(r_eqs):
        print(f"R {i}: {eq}")


def test_kcl_equations_exclude_ground():
    """Test that KCL equations exclude the ground node equation."""
    # Create a simple circuit with a ground node
    graph = nx.Graph()
    
    # Create junctions
    ground_junction = ElecJunction(junction_id=1, is_ground=True)
    node_junction = ElecJunction(junction_id=2)
    
    # Add nodes
    graph.add_node("1", junction=ground_junction)
    graph.add_node("2", junction=node_junction)
    
    # Add a resistor between nodes
    resistor1 = Resistor(comp_id="R1", value=1000)
    resistor2 = Resistor(comp_id="R2", value=2000)
    graph.add_edge("1", "2", component=resistor1)
    graph.add_edge("2", "1", component=resistor2)
    
    # Create DAE model
    model = ElectricalDaeModel(graph)
    
    # Get KCL equations
    kcl_equations = model.compute_kcl_equations()
    
    # Verify that we only have one equation (for the non-ground node)
    assert len(kcl_equations) == 1, "Should have only one KCL equation (excluding ground)"
    
    # Verify that the equation contains the resistor current
    assert "I_R1" in kcl_equations[0], "KCL equation should contain the resistor current" 