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

def test_switch_control_signals():
    """
    Test the switch control signals functionality with multiple switches.
    Tests a circuit with three switches:
    - S1: switch time = -1.1s (always ON)
    - S2: switch time = 20.0s
    - S3: switch time = 2.3333s
    """
    # Load and parse circuit with multiple switches
    circuit_data = load_test_circuit("engine_switch_control.json")
    parser = ParserJson()
    graph = parser.parse(circuit_data)
    
    # Create engine instance and initialize
    engine = Engine(graph)
    engine.initialize()
    
    # Define switch times
    switch_times = {
        "S1": -1.1,  # Switch S1 is always ON
        "S2": 20.0,  # Switch S2 turns ON at t=20.0
        "S3": 2.3333  # Switch S3 turns ON at t=2.3333
    }
    
    # Get the switch control signals function
    switch_control_signals = engine._get_switch_control_signals()
    
    # Test switch states at different times
    test_times = {
        0.0: {"S1": 1, "S2": 0, "S3": 0},  # t=0 (S1 ON, S2 OFF, S3 OFF)
        1.0: {"S1": 1, "S2": 0, "S3": 0},  # t=1 (S1 ON, S2 OFF, S3 OFF)
        2.3333: {"S1": 1, "S2": 0, "S3": 1},  # t=2.3333 (S1 ON, S2 OFF, S3 ON)
        2.3334: {"S1": 1, "S2": 0, "S3": 1},  # t=2.3334 (S1 ON, S2 OFF, S3 ON)
        20.0: {"S1": 1, "S2": 1, "S3": 1},  # t=20.0 (S1 ON, S2 ON, S3 ON)
        21.0: {"S1": 1, "S2": 1, "S3": 1}  # t=21.0 (S1 ON, S2 ON, S3 ON)
    }
    
    for t, expected_states in test_times.items():
        states = switch_control_signals(t)
        assert len(states) == len(engine.power_switches)
        for i, switch_id in enumerate(engine.power_switches):
            assert states[i] == expected_states[switch_id], f"Switch {switch_id} at t={t}"

def test_switch_events():
    """
    Test the creation of switch events for solve_ivp.
    """
    # Load and parse circuit with multiple switches
    circuit_data = load_test_circuit("engine_switch_control.json")
    parser = ParserJson()
    graph = parser.parse(circuit_data)
    
    # Create engine instance and initialize
    engine = Engine(graph)
    engine.initialize()
    
    # Get the switch events
    switch_events = engine._get_switch_events()
    
    # Define expected switch times
    expected_switch_times = {
        "S1": -1.1,
        "S2": 20.0,
        "S3": 2.3333
    }
    
    # Test that the number of events matches the number of switches
    assert len(switch_events) == len(engine.power_switches)
    
    # Test each event function
    for i, switch_id in enumerate(engine.power_switches):
        event = switch_events[i]
        switch_time = expected_switch_times[switch_id]
        # Event should return 0 at the switch time
        assert event(switch_time, None) == 0
        # Event should not return 0 before the switch time
        assert event(switch_time - 0.1, None) != 0
        # Event should not return 0 after the switch time
        assert event(switch_time + 0.1, None) != 0

