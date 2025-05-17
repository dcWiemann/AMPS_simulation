import pytest
import networkx as nx
import sympy as sp
from sympy.abc import t
import json
from amps_simulation.core.engine import Engine
from amps_simulation.core.components import Capacitor, Inductor, VoltageSource, CurrentSource, PowerSwitch, Component
from amps_simulation.core.parser_networkx import ParserJson
from amps_simulation.core.dae_model import ElectricalDaeModel

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
    v_C = sp.Function("v_C")(t)
    i_L = sp.Function("i_L")(t)
    assert v_C in engine.state_vars
    assert i_L in engine.state_vars
    
    # Test input variables
    v_V = sp.Function("v_V")(t)  # voltage source V
    assert v_V in engine.input_vars
    
    # Test power switches (should be empty for RLC)
    assert len(engine.switch_list) == 0

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
    
    # Test input variables
    v_V1 = sp.Function("v_V1")(t)  # voltage source V1
    assert v_V1 in engine.input_vars
    
    # Test power switches
    assert len(engine.switch_list) == 1
    assert any(switch.comp_id == "S1" for switch in engine.switch_list)

# def test_switch_control_signals():
#     """
#     Test the switch control signals functionality with multiple switches.
#     Tests a circuit with three switches:
#     - S1: switch time = -1.1s (always ON)
#     - S2: switch time = 20.0s
#     - S3: switch time = 2.3333s
#     """
#     # Load and parse circuit with multiple switches
#     circuit_data = load_test_circuit("engine_switch_control.json")
#     parser = ParserJson()
#     graph = parser.parse(circuit_data)
    
#     # Create engine instance and initialize
#     engine = Engine(graph)
#     engine.initialize()
    
#     # Define switch times
#     switch_times = {
#         "S1": -1.1,  # Switch S1 is always ON
#         "S2": 20.0,  # Switch S2 turns ON at t=20.0
#         "S3": 2.3333  # Switch S3 turns ON at t=2.3333
#     }
    
#     # Get the switch control signals function
#     switch_control_signals = engine._get_switch_control_signals()
    
#     # Test switch states at different times
#     test_times = {
#         0.0: {"S1": 1, "S2": 0, "S3": 0},  # t=0 (S1 ON, S2 OFF, S3 OFF)
#         1.0: {"S1": 1, "S2": 0, "S3": 0},  # t=1 (S1 ON, S2 OFF, S3 OFF)
#         2.3333: {"S1": 1, "S2": 0, "S3": 1},  # t=2.3333 (S1 ON, S2 OFF, S3 ON)
#         2.3334: {"S1": 1, "S2": 0, "S3": 1},  # t=2.3334 (S1 ON, S2 OFF, S3 ON)
#         20.0: {"S1": 1, "S2": 1, "S3": 1},  # t=20.0 (S1 ON, S2 ON, S3 ON)
#         21.0: {"S1": 1, "S2": 1, "S3": 1}  # t=21.0 (S1 ON, S2 ON, S3 ON)
#     }
    
#     for t, expected_states in test_times.items():
#         states = switch_control_signals(t)
#         assert len(states) == len(engine.switch_list)
#         for i, switch_id in enumerate(engine.switch_list):
#             assert states[i] == expected_states[switch_id], f"Switch {switch_id} at t={t}"

def test_switch_events():
    """
    Test the creation of switch events for solve_ivp.
    """
    # Load and parse circuit with multiple switches
    circuit_data = load_test_circuit("engine_switch_control.json")
    parser = ParserJson()
    graph = parser.parse(circuit_data)
    
    Component.clear_registry()
    # Create engine instance and initialize
    engine = Engine(graph)
    engine.switch_list = [PowerSwitch(comp_id="S1", switch_time=-1.1, is_on=True),
                          PowerSwitch(comp_id="S2", switch_time=20.0, is_on=False),
                          PowerSwitch(comp_id="S3", switch_time=2.3333, is_on=False)]
    
    # Get the switch events
    switch_events = engine._get_switch_events()
    
    # Define expected switch times
    expected_switch_times = {
        "S1": -1.1,
        "S2": 20.0,
        "S3": 2.3333
    }
    
    # Test that the number of events matches the number of switches
    assert len(switch_events) == len(engine.switch_list)
    
    # Test each event function
    for i, switch in enumerate(engine.switch_list):
        event = switch_events[i]
        switch_time = expected_switch_times[switch.comp_id]
        # Event should return 0 at the switch time
        assert event(switch_time, None) == 0
        # Event should not return 0 before the switch time
        assert event(switch_time - 0.1, None) != 0
        # Event should not return 0 after the switch time
        assert event(switch_time + 0.1, None) != 0

def test_compute_state_space_model():
    """
    Test the computation of state space model matrices for an RLC circuit.
    The circuit consists of:
    - A voltage source V
    - A resistor R
    - An inductor L
    - A capacitor C
    """
    # Load and parse RLC circuit
    circuit_data = load_test_circuit("test_rlc.json")
    parser = ParserJson()
    graph = parser.parse(circuit_data)
    
    # Create engine instance and initialize
    engine = Engine(graph)
    engine.initialize()
    
    # Create model and get equations
    model = ElectricalDaeModel(graph)
    model.initialize()
    derivatives = model.get_derivatives()
    output_eqs = model.output_eqs
    
    # Sort equations to match state and output variables
    sorted_derivatives = engine._sort_derivatives_by_state_vars(derivatives)
    sorted_output_eqs = engine._sort_output_eqs_by_output_vars(output_eqs)
    
    # Compute state space model
    A, B, C, D = engine.compute_state_space_model(sorted_derivatives, sorted_output_eqs)
    
    # Test matrix dimensions
    n_states = len(engine.state_vars)
    n_inputs = len(engine.input_vars)
    n_outputs = len(engine.output_vars)
    
    assert A.shape == (n_states, n_states)
    assert B.shape == (n_states, n_inputs)
    assert C.shape == (n_outputs, n_states)
    assert D.shape == (n_outputs, n_inputs)
    
    # Test that matrices contain symbolic expressions
    assert all(isinstance(expr, sp.Expr) for expr in A)
    assert all(isinstance(expr, sp.Expr) for expr in B)
    assert all(isinstance(expr, sp.Expr) for expr in C)
    assert all(isinstance(expr, sp.Expr) for expr in D)

