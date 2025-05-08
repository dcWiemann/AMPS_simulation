import pytest
import networkx as nx
import sympy as sp
import json
from amps_simulation.core.engine import Engine
from amps_simulation.core.components import Capacitor, Inductor, VoltageSource, CurrentSource, PowerSwitch
from amps_simulation.core.parser_networkx import ParserJson

def load_test_circuit(filename):
    """Helper function to load test circuit from JSON file."""
    with open(f"test_data/{filename}", 'r') as f:
        return json.load(f)

def test_initialize_rlc():
    """
    Test initialization of RLC circuit.
    """
    # Load and parse RLC circuit
    circuit_data = load_test_circuit("test_rlc.json")
    parser = ParserJson()
    graph = parser.parse(circuit_data)
    
    # Create engine instance and initialize
    engine = Engine(graph)
    engine.initialize()
    
    # Test components list
    assert len(engine.components_list) > 0
    
    # Test state variables
    v_C = sp.Symbol("v_C")
    i_L = sp.Symbol("i_L")
    assert v_C in engine.state_vars
    assert i_L in engine.state_vars
    
    # Test input variables
    v_V = sp.Symbol("v_V")  # voltage source V
    assert v_V in engine.input_vars
    assert engine.input_vars[v_V] == "V"
    
    # Test power switches (should be empty for RLC)
    assert len(engine.power_switches) == 0

def test_initialize_resistive():
    """
    Test initialization of resistive circuit with switch.
    """
    # Load and parse resistive circuit
    circuit_data = load_test_circuit("engine_resistive.json")
    parser = ParserJson()
    graph = parser.parse(circuit_data)
    
    # Create engine instance and initialize
    engine = Engine(graph)
    engine.initialize()
    
    # Test components list
    assert len(engine.components_list) > 0
    
    # Test state variables (should be empty for resistive circuit)
    assert len(engine.state_vars) == 0
    assert len(engine.state_derivatives) == 0
    
    # Test input variables
    v_V1 = sp.Symbol("v_V1")  # voltage source V1
    assert v_V1 in engine.input_vars
    assert engine.input_vars[v_V1] == "V1"
    
    # Test power switches
    assert len(engine.power_switches) == 1
    assert "S1" in engine.power_switches

