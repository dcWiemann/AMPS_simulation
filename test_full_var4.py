#!/usr/bin/env python3
"""
Quick test of full_var4.json to check if it has the same constraint issues.
"""

import json
import logging
from amps_simulation.core.parser import ParserJson
from amps_simulation.core.engine import Engine
from amps_simulation.core.electrical_model import ElectricalModel
from amps_simulation.core.components import Component

# Configure logging to see debug info
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_full_var0():
    """Test full_var0.json circuit."""
    print("Testing full_var0.json...")
    
    # Clear component registry
    Component.clear_registry()
    
    # Load circuit
    with open("test_data/full_var0.json", 'r') as f:
        circuit_data = json.load(f)
    
    print(f"Components: {[comp['id'] + ' (' + comp['data']['componentType'] + ')' for comp in circuit_data['nodes']]}")
    print(f"Connections: {len(circuit_data['edges'])}")
    
    # Parse circuit
    parser = ParserJson()
    try:
        graph, control_graph = parser.parse(circuit_data)
        print(f"[+] Parsing successful - {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    except Exception as e:
        print(f"[-] Parsing failed: {e}")
        return False

    # Create electrical model and engine
    try:
        electrical_model = ElectricalModel(graph)
        engine = Engine(electrical_model, control_graph)
        print("[+] Engine creation successful")
    except Exception as e:
        print(f"[-] Engine creation failed: {e}")
        return False
    
    # Initialize engine
    try:
        engine.initialize()
        print("[+] Engine initialization successful!")
        print(f"  State vars: {engine.state_vars}")
        print(f"  Input vars: {engine.input_vars}")
        print(f"  Output vars: {engine.output_vars}")
        return True
    except Exception as e:
        print(f"[-] Engine initialization failed: {e}")
        return False

if __name__ == "__main__":
    success = test_full_var0()
    if success:
        print("\nSUCCESS: full_var0.json works!")
    else:
        print("\nFAILED: full_var0.json has the same constraint issues")