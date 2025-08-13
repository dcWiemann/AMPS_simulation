import pytest
import networkx as nx
import sympy as sp
from sympy.abc import t
import json
import numpy as np
from amps_simulation.core.engine import Engine
from amps_simulation.core.components import Capacitor, Inductor, VoltageSource, CurrentSource, PowerSwitch, Component
from amps_simulation.core.parser import ParserJson
from amps_simulation.core.dae_model import ElectricalDaeModel
from amps_simulation.core.control_orchestrator import ControlGraph

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
    graph, control_graph = parser.parse(circuit_data)
    
    # Create engine instance and initialize
    engine = Engine(graph, control_graph)
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
    graph, control_graph = parser.parse(circuit_data)
    
    # Create engine instance and initialize
    engine = Engine(graph, control_graph)
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

def test_run_simulation_basic():
    """Test the new run_simulation method with a simple circuit."""
    # Load and parse RC circuit
    circuit_data = load_test_circuit("test_rc.json")
    parser = ParserJson()
    graph, control_graph = parser.parse(circuit_data)
    
    # Create engine instance and initialize
    engine = Engine(graph, control_graph)
    engine.initialize()
    
    # Run simulation
    result = engine.run_simulation(t_span=(0, 0.1), method='RK45')
    
    # Test result structure
    assert result['success'] == True
    assert 't' in result
    assert 'y' in result
    assert 'out' in result
    assert 'switchmap_size' in result
    
    # Test that time array is reasonable
    assert len(result['t']) > 1
    assert result['t'][0] == 0.0
    assert result['t'][-1] >= 0.1
    
    # Test that state array has correct shape
    n_states = len(engine.state_vars)
    assert result['y'].shape[0] == n_states
    assert result['y'].shape[1] == len(result['t'])

def test_run_simulation_with_switch():
    """Test the new run_simulation method with a switching circuit."""
    # Load and parse switching circuit with meters
    circuit_data = load_test_circuit("engine_switch_meters.json")
    parser = ParserJson()
    graph, control_graph = parser.parse(circuit_data)
    
    # Create engine instance and initialize
    engine = Engine(graph, control_graph)
    engine.initialize()
    
    # Run simulation that spans the switch event
    result = engine.run_simulation(t_span=(0, 1.5), method='RK45')
    
    # Test result structure
    assert result['success'] == True
    assert 't' in result
    assert 'y' in result
    assert 'out' in result
    assert 'switchmap_size' in result
    assert 't_events' in result
    
    # Test switch-specific features
    assert result['switchmap_size'] == 2  # Should cache models for switch open/closed
    assert len([e for e in result['t_events'] if len(e) > 0]) == 1  # One switch event
    
    # Test that outputs exist (circuit has meters)
    assert result['out'] is not None
    assert result['out'].shape[0] == len(engine.output_vars)  # Number of meter outputs
    assert result['out'].shape[1] == len(result['t'])  # Same number of time points

def test_run_simulation_with_outputs():
    """Test that run_simulation correctly generates outputs for circuits with meters."""
    # Use the switching circuit which has both state variables and meter outputs
    circuit_data = load_test_circuit("engine_switch_meters.json")
    parser = ParserJson()
    graph, control_graph = parser.parse(circuit_data)
    
    Component.clear_registry()
    engine = Engine(graph, control_graph)
    engine.initialize()
    
    # Run simulation
    result = engine.run_simulation(t_span=(0, 0.5), method='RK45')
    
    # Test outputs
    assert result['success'] == True
    assert result['out'] is not None
    assert len(engine.output_vars) > 0  # Should have meter outputs
    assert result['out'].shape[0] == len(engine.output_vars)
    assert result['out'].shape[1] == len(result['t'])

def test_run_simulation_no_states():
    """Test the run_simulation method with a circuit that has no state variables."""
    # Load and parse the no-states circuit (current source + resistor + voltmeter)
    circuit_data = load_test_circuit("engine_nostates.json")
    parser = ParserJson()
    graph, control_graph = parser.parse(circuit_data)
    
    Component.clear_registry()
    # Create engine instance and initialize
    engine = Engine(graph, control_graph)
    engine.initialize()
    
    # Verify this is indeed a no-states circuit
    assert len(engine.state_vars) == 0, "Circuit should have no state variables"
    assert len(engine.input_vars) > 0, "Circuit should have input variables (current source)"
    assert len(engine.output_vars) > 0, "Circuit should have output variables (voltmeter)"
    
    # Run simulation - this should work even with no states
    result = engine.run_simulation(t_span=(0, 0.1), method='RK45')
    
    # Test result structure
    assert result['success'] == True, "Simulation should succeed even with no states"
    assert 't' in result
    assert 'y' in result
    assert 'out' in result
    assert 'switchmap_size' in result
    
    # For no-states circuit, y should be empty but out should have values
    assert result['y'].shape[0] == 0, "Should have no state variables"
    assert result['out'] is not None, "Should have outputs"
    assert result['out'].shape[0] == len(engine.output_vars), "Output size should match number of meters"
    
    # Outputs should be constant (steady-state algebraic solution)
    # For I=4.4A through R=0.5Ω, voltage magnitude should be |V|=|IR|=2.2V
    # The DAE solver found v_VM2 = -0.5*i_I1(t) = -2.2V due to direction convention
    expected_voltage = -2.2  # From DAE solution: v_VM2 = -0.5*i_I1(t) = -0.5*4.4 = -2.2V
    output_values = result['out'][0]  # Voltmeter reading
    assert np.allclose(output_values, expected_voltage, rtol=1e-6), f"Expected {expected_voltage}V, got {output_values}"

def test_switch_events():
    """
    Test the creation of switch events for solve_ivp.
    """
    # Load and parse circuit with switch
    circuit_data = load_test_circuit("engine_switch_meters.json")
    parser = ParserJson()
    graph, control_graph = parser.parse(circuit_data)
    
    Component.clear_registry()
    # Create engine instance and initialize
    engine = Engine(graph, control_graph)
    engine.initialize()
    
    # Test that switch events were created
    if engine.switch_list and len(engine.switch_list) > 0:
        switch_events = engine.switch_events
        assert switch_events is not None
        assert len(switch_events) == len(engine.switch_list)
        
        # Test each event function
        for i, switch in enumerate(engine.switch_list):
            event = switch_events[i]
            switch_time = switch.switch_time
            
            # Event should return 0 at the switch time
            assert abs(event(switch_time, None)) < 1e-10
            # Event should not return 0 before the switch time
            assert event(switch_time - 0.1, None) != 0
            # Event should not return 0 after the switch time  
            assert event(switch_time + 0.1, None) != 0
            
            # Test event properties
            assert hasattr(event, 'terminal')
            assert hasattr(event, 'direction')
            assert event.terminal == False  # Should allow continuation
            assert event.direction == 1     # Positive crossings only

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
    graph, control_graph = parser.parse(circuit_data)
    
    # Create engine instance and initialize
    engine = Engine(graph, control_graph)
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

def test_engine_control_orchestrator_integration():
    """Test that Engine integrates with ControlOrchestrator for sources with values."""
    Component.clear_registry()
    
    # Create test circuit with voltage source having a value
    circuit_data = {
        "nodes": [
            {"id": "V1", "data": {"componentType": "voltage-source", "value": 10.0}},
            {"id": "R1", "data": {"componentType": "resistor", "value": 100}},
            {"id": "C1", "data": {"componentType": "capacitor", "value": 1e-6}},
            {"id": "GND", "data": {"componentType": "ground"}}
        ],
        "edges": [
            {"source": "V1", "target": "R1", "sourceHandle": "0", "targetHandle": "0"},
            {"source": "R1", "target": "C1", "sourceHandle": "1", "targetHandle": "0"},
            {"source": "C1", "target": "GND", "sourceHandle": "1", "targetHandle": "0"},
            {"source": "GND", "target": "V1", "sourceHandle": "0", "targetHandle": "1"}
        ]
    }

    parser = ParserJson()
    graph, control_graph = parser.parse(circuit_data)

    # Create engine with control graph
    engine = Engine(graph, control_graph)
    engine.initialize()

    # Verify ControlOrchestrator was created
    assert hasattr(engine, 'control_orchestrator')
    assert engine.control_orchestrator.control_graph is control_graph

    # Verify control input function was created for source ports
    assert hasattr(engine, 'control_input_function')
    
    # Test the input function
    input_values = engine.control_input_function(0.0)
    assert len(input_values) == 1  # One source
    assert input_values[0] == 10.0  # Voltage source value

def test_engine_no_control_sources():
    """Test Engine behavior when circuit has no sources with values."""
    Component.clear_registry()
    
    # Create test circuit without sources with values
    circuit_data = {
        "nodes": [
            {"id": "V1", "data": {"componentType": "voltage-source"}},  # No value field
            {"id": "R1", "data": {"componentType": "resistor", "value": 100}},
            {"id": "GND", "data": {"componentType": "ground"}}
        ],
        "edges": [
            {"source": "V1", "target": "R1", "sourceHandle": "0", "targetHandle": "0"},
            {"source": "R1", "target": "GND", "sourceHandle": "1", "targetHandle": "0"},
            {"source": "GND", "target": "V1", "sourceHandle": "0", "targetHandle": "1"}
        ]
    }

    parser = ParserJson()
    graph, control_graph = parser.parse(circuit_data)

    # Create engine with empty control graph
    engine = Engine(graph, control_graph)
    engine.initialize()

    # Should still have ControlOrchestrator
    assert hasattr(engine, 'control_orchestrator')
    
    # Should not have control input function since no source ports
    assert not hasattr(engine, 'control_input_function')

def test_engine_multiple_source_control():
    """Test Engine with multiple sources having values."""
    Component.clear_registry()
    
    circuit_data = {
        "nodes": [
            {"id": "V1", "data": {"componentType": "voltage-source", "value": 5.0}},
            {"id": "I1", "data": {"componentType": "current-source", "value": 0.1}},
            {"id": "R1", "data": {"componentType": "resistor", "value": 50}},
            {"id": "R2", "data": {"componentType": "resistor", "value": 100}},
            {"id": "GND", "data": {"componentType": "ground"}}
        ],
        "edges": [
            {"source": "V1", "target": "R1", "sourceHandle": "0", "targetHandle": "0"},
            {"source": "R1", "target": "I1", "sourceHandle": "1", "targetHandle": "1"},
            {"source": "I1", "target": "R2", "sourceHandle": "0", "targetHandle": "0"},
            {"source": "R2", "target": "GND", "sourceHandle": "1", "targetHandle": "0"},
            {"source": "GND", "target": "V1", "sourceHandle": "0", "targetHandle": "1"}
        ]
    }

    parser = ParserJson()
    graph, control_graph = parser.parse(circuit_data)

    engine = Engine(graph, control_graph)
    engine.initialize()

    # Should have control input function for multiple sources
    assert hasattr(engine, 'control_input_function')
    
    # Test input function returns values for both sources
    input_values = engine.control_input_function(0.0)
    assert len(input_values) == 2  # Two sources
    # Values should match the order engine determines for input_vars
    assert 5.0 in input_values  # Voltage source value
    assert 0.1 in input_values  # Current source value

