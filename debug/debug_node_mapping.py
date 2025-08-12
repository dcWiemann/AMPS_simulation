#!/usr/bin/env python3
"""
Debug node mapping in the parser to understand why R1 and C2 aren't connecting to ground.
"""

import json
from amps_simulation.core.parser_networkx import ParserJson
from amps_simulation.core.components import Component
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

def debug_node_mapping():
    """Debug the node mapping process."""
    print("=== Debugging Node Mapping Process ===\n")
    
    # Clear component registry
    Component.clear_registry()
    
    # Load circuit data
    with open("test_data/engine_RC_switch.json", 'r') as f:
        circuit_data = json.load(f)
    
    # Create parser and inspect the node mapping process
    parser = ParserJson()
    
    # Extract components and connections
    components = circuit_data["nodes"]
    connections = circuit_data["edges"]
    
    print("1. CONNECTIONS (EDGES) FROM JSON:")
    print("=================================")
    for i, conn in enumerate(connections):
        print(f"  Connection {i}:")
        print(f"    {conn['source']} (handle {conn.get('sourceHandle')}) -> {conn['target']} (handle {conn.get('targetHandle')})")
    
    print("\n2. COMPONENTS FROM JSON:")
    print("========================")
    for comp in components:
        comp_type = comp["data"].get("componentType")
        print(f"  {comp['id']}: {comp_type}")
    
    # Create components first
    parser.components_list = parser._create_circuit_components(components)
    
    print("\n3. CREATED COMPONENT OBJECTS:")
    print("=============================")
    for comp in parser.components_list:
        print(f"  {comp.comp_id}: {type(comp).__name__}")
    
    # Now debug the node mapping process step by step
    print("\n4. NODE MAPPING PROCESS:")
    print("========================")
    
    # Call the method and capture the result
    node_mapping, next_node_number, ground_node = parser._identify_electrical_nodes(connections)
    
    print(f"Final node mapping: {node_mapping}")
    print(f"Next node number: {next_node_number}")
    print(f"Ground node: {ground_node}")
    
    print("\n5. EXPECTED vs ACTUAL MAPPING:")
    print("==============================")
    expected_mapping = {
        ('V1', '0'): 'Node connected to R2',
        ('V1', '1'): 'Ground node (should be same as GND1)',
        ('R2', '0'): 'Node connected to V1',
        ('R2', '1'): 'Node connected to S1 and C2',
        ('S1', '0'): 'Node connected to R2 and C2',
        ('S1', '1'): 'Node connected to R1',
        ('C2', '0'): 'Node connected to R2 and S1',
        ('C2', '1'): 'Ground node (should be same as V1 and GND1)',
        ('R1', '0'): 'Node connected to S1',
        ('R1', '1'): 'Ground node (should be same as V1, C2, and GND1)',
        ('GND1', '0'): 'Ground node (should be same as V1, C2, and R1)',
    }
    
    print("Expected connections:")
    for key, description in expected_mapping.items():
        actual_node = node_mapping.get(key, 'NOT FOUND')
        print(f"  {key}: {description} -> Actual: Node {actual_node}")
    
    print("\n6. GROUND NODE ANALYSIS:")
    print("========================")
    ground_terminals = []
    for key, node in node_mapping.items():
        if node == ground_node:
            ground_terminals.append(key)
    print(f"Terminals connected to ground node {ground_node}: {ground_terminals}")
    
    print("\nThe issue: If R1 and C2 terminals that should connect to ground")
    print("are not in the ground_terminals list, then the parser failed to")
    print("properly identify the electrical connections.")

if __name__ == "__main__":
    debug_node_mapping()