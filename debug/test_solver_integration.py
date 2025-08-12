#!/usr/bin/env python3
"""
Test script for the new solver integration in Engine class.
Tests the run_simulation method with solve_ivp on an RLC circuit.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from amps_simulation.core.parser_networkx import ParserJson
from amps_simulation.core.engine import Engine
from amps_simulation.core.components import Component
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def load_test_circuit(filename):
    """Helper function to load test circuit from JSON file."""
    with open(f"test_data/{filename}", 'r') as f:
        return json.load(f)

def test_rlc_simulation():
    """Test the new solver integration with RLC circuit."""
    print("Testing RLC circuit with new solver integration...")
    
    # Clear component registry
    Component.clear_registry()
    
    # Load and parse RLC circuit
    circuit_data = load_test_circuit("test_rlc.json")
    parser = ParserJson()
    graph, control_graph = parser.parse(circuit_data)
    
    # Create engine instance and initialize
    engine = Engine(graph, control_graph)
    engine.initialize()
    
    print(f"State variables: {engine.state_vars}")
    print(f"Input variables: {engine.input_vars}")
    print(f"Output variables: {engine.output_vars}")
    print(f"Switch list: {engine.switch_list}")
    
    # Set up simulation parameters
    t_span = (0.0, 10.0)  # 10 second simulation
    initial_conditions = np.zeros(len(engine.state_vars))  # Start from zero
    
    print(f"Running simulation from t={t_span[0]} to t={t_span[1]}")
    print(f"Initial conditions: {initial_conditions}")
    
    # Run simulation using the new solve_ivp integration
    try:
        result = engine.run_simulation(
            t_span=t_span,
            initial_conditions=initial_conditions,
            method='RK45',
            dense_output=True,
            max_step=0.1
        )
        
        print(f"Simulation completed successfully!")
        print(f"Time points: {len(result['t'])}")
        print(f"Final time: {result['t'][-1]}")
        print(f"Final state: {result['y'][:, -1] if result['y'].size > 0 else 'No states'}")
        print(f"Switch models used: {result['switch_models_used']}")
        
        # Plot results if we have state variables
        if result['y'].size > 0 and len(result['t']) > 1:
            plt.figure(figsize=(12, 8))
            
            # Plot state variables
            for i, state_var in enumerate(engine.state_vars):
                plt.subplot(2, 2, i + 1)
                plt.plot(result['t'], result['y'][i, :])
                plt.title(f'State Variable: {state_var}')
                plt.xlabel('Time (s)')
                plt.ylabel('Value')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('rlc_simulation_results.png', dpi=150, bbox_inches='tight')
            print("Results plotted and saved to 'rlc_simulation_results.png'")
        else:
            print("No state variables to plot")
            
        return True
        
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rlc_simulation()
    if success:
        print("SUCCESS: Solver integration test PASSED")
    else:
        print("FAILED: Solver integration test FAILED")