#!/usr/bin/env python3
"""
Test script for solver integration with switching circuits.
Tests the run_simulation method with solve_ivp on a circuit containing switches.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from amps_simulation.core.parser import ParserJson
from amps_simulation.core.engine import Engine
from amps_simulation.core.components import Component
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

def load_test_circuit(filename):
    """Helper function to load test circuit from JSON file."""
    with open(f"test_data/{filename}", 'r') as f:
        return json.load(f)

def test_switch_simulation():
    """Test the new solver integration with a switching circuit."""
    print("Testing switching circuit with new solver integration...")
    
    # Clear component registry
    Component.clear_registry()
    
    # Load and parse switching circuit
    circuit_data = load_test_circuit("engine_RC_switch.json")
    parser = ParserJson()
    graph, control_graph = parser.parse(circuit_data)
    
    # Create engine instance and initialize
    engine = Engine(graph, control_graph)
    engine.initialize()
    
    print(f"State variables: {engine.state_vars}")
    print(f"Input variables: {engine.input_vars}")
    print(f"Output variables: {engine.output_vars}")
    print(f"Switch list: {[switch.comp_id for switch in engine.switch_list]}")
    print(f"Switch times: {[switch.switch_time for switch in engine.switch_list]}")
    
    # Set up simulation parameters
    t_span = (0.0, 5.0)  # 5 second simulation to see switch events
    initial_conditions = np.zeros(len(engine.state_vars))  # Start from zero
    
    print(f"Running simulation from t={t_span[0]} to t={t_span[1]}")
    print(f"Initial conditions: {initial_conditions}")
    
    # Run simulation using the new solve_ivp integration
    try:
        result = engine.run_simulation(
            t_span=t_span,
            initial_conditions=initial_conditions,
            method='RK45',
            max_step=0.01  # Smaller step to catch switch events
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
                
                # Mark switch times
                for switch in engine.switch_list:
                    if switch.switch_time <= result['t'][-1]:
                        plt.axvline(x=switch.switch_time, color='red', linestyle='--', 
                                  label=f'Switch {switch.comp_id} at t={switch.switch_time}')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig('switch_simulation_results.png', dpi=150, bbox_inches='tight')
            print("Results plotted and saved to 'switch_simulation_results.png'")
        else:
            print("No state variables to plot")
            
        return True
        
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_switch_simulation()
    if success:
        print("SUCCESS: Switch solver integration test PASSED")
    else:
        print("FAILED: Switch solver integration test FAILED")