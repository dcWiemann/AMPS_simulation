#!/usr/bin/env python3
"""
Debug the circuit test issues with full_var circuits.
"""

import json
import numpy as np
from amps_simulation.core.parser import ParserJson
from amps_simulation.core.engine import Engine
from amps_simulation.core.components import Component

def test_circuit(filename):
    """Test a specific circuit file."""
    print(f"\n=== Testing {filename} ===")
    
    # Clear component registry
    Component.clear_registry()
    
    try:
        # Load circuit
        with open(f"test_data/{filename}", 'r') as f:
            circuit_data = json.load(f)
        
        # Parse circuit
        parser = ParserJson()
        graph, control_graph = parser.parse(circuit_data)
        
        # Create and initialize engine
        engine = Engine(graph, control_graph)
        engine.initialize()
        
        print(f"State vars: {engine.state_vars}")
        print(f"Input vars: {engine.input_vars}")
        print(f"Output vars: {engine.output_vars}")
        
        # Try a quick simulation
        t_span = (0.0, 1.0)
        initial_conditions = np.zeros(len(engine.state_vars))
        
        result = engine.run_simulation(
            t_span=t_span,
            initial_conditions=initial_conditions,
            method='RK45',
            max_step=0.1
        )
        
        print(f"Simulation successful! Final time: {result['t'][-1]:.3f}s")
        print(f"Final states: {result['y'][:, -1] if result['y'].size > 0 else 'No states'}")
        
        if 'outputs' in result and result['outputs'].size > 0:
            print(f"Final outputs: {result['outputs'][:, -1]}")
            
            # Check for expected values
            final_outputs = result['outputs'][:, -1]
            for i, output_var in enumerate(engine.output_vars):
                value = abs(final_outputs[i])
                print(f"  {output_var}: {value:.4f}")
                
                # Check if it's a voltage (should be ~5V) or current (should be ~2.5A)
                if 'VM' in output_var or 'v_' in output_var:
                    if abs(value - 5.0) <= 0.01:
                        print(f"    ✓ Voltage within spec: 5.0 ± 0.01V")
                    else:
                        print(f"    ! Voltage outside spec: {value:.4f}V (expected 5.0 ± 0.01V)")
                elif 'AM' in output_var or 'i_' in output_var:
                    if abs(value - 2.5) <= 0.01:
                        print(f"    ✓ Current within spec: 2.5 ± 0.01A")
                    else:
                        print(f"    ! Current outside spec: {value:.4f}A (expected 2.5 ± 0.01A)")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    circuits = ['full_var0.json', 'full_var1.json', 'full_var2.json', 'full_var3.json']
    
    results = {}
    for circuit in circuits:
        results[circuit] = test_circuit(circuit)
    
    print(f"\n=== SUMMARY ===")
    for circuit, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{circuit}: {status}")
        
    all_passed = all(results.values())
    print(f"\nOverall: {'ALL CIRCUITS WORKING' if all_passed else 'SOME CIRCUITS FAILED'}")