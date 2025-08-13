#!/usr/bin/env python3
"""
Debug script to examine the electrical graph construction for the RC switch circuit.
"""

import json
import networkx as nx
from amps_simulation.core.parser import ParserJson
from amps_simulation.core.components import Component
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

def load_test_circuit(filename):
    """Helper function to load test circuit from JSON file."""
    with open(f"test_data/{filename}", 'r') as f:
        return json.load(f)

def debug_graph_construction():
    """Debug the graph construction for the RC switch circuit."""
    print("=== Debugging RC Switch Circuit Graph Construction ===\n")
    
    # Clear component registry
    Component.clear_registry()
    
    # Load and parse circuit
    circuit_data = load_test_circuit("engine_RC_switch.json")
    parser = ParserJson()
    graph, control_graph = parser.parse(circuit_data)
    
    print("1. NODES (Junctions):")
    print("====================")
    for node_id in graph.nodes():
        node_data = graph.nodes[node_id]
        junction = node_data.get('junction')
        if junction:
            print(f"  Node {node_id}: voltage_var={junction.voltage_var}, is_ground={junction.is_ground}")
        else:
            print(f"  Node {node_id}: No junction data")
    
    print("\n2. EDGES (Components):")
    print("======================")
    for i, (source, target, edge_data) in enumerate(graph.edges(data=True)):
        component = edge_data.get('component')
        if component:
            print(f"  Edge {i}: {source} -> {target}")
            print(f"    Component: {component.comp_id} ({type(component).__name__})")
            print(f"    Value: {getattr(component, 'resistance', None) or getattr(component, 'capacitance', None) or getattr(component, 'voltage', None) or getattr(component, 'switch_time', None)}")
            if hasattr(component, 'voltage_var'):
                print(f"    Voltage var: {component.voltage_var}")
            if hasattr(component, 'current_var'):
                print(f"    Current var: {component.current_var}")
            if hasattr(component, 'switch_time'):
                print(f"    Switch time: {component.switch_time}")
        else:
            print(f"  Edge {i}: {source} -> {target} - NO COMPONENT")
        print()
    
    print("3. EXPECTED CIRCUIT TOPOLOGY:")
    print("=============================")
    print("When Switch OFF (S1 open):")
    print("  V1(+) -> R2 -> C2(+) -> (open switch) -> R1 -> V1(-)")
    print("  Expected: Capacitor charges through R2 only")
    print()
    print("When Switch ON (S1 closed):")
    print("  V1(+) -> R2 -> C2(+) -> S1(closed) -> R1 -> V1(-)")  
    print("  Expected: Capacitor charges through R2, discharges through R1+R2")
    print()
    
    print("4. ADJACENCY ANALYSIS:")
    print("======================")
    print("Graph adjacency (showing actual connections):")
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        print(f"  Node {node} connected to: {neighbors}")
    
    print("\n5. COMPONENT ANALYSIS:")
    print("======================")
    components = []
    for _, _, edge_data in graph.edges(data=True):
        component = edge_data.get('component')
        if component:
            components.append(component)
    
    print("Components found:")
    for comp in components:
        print(f"  {comp.comp_id}: {type(comp).__name__}")
        if hasattr(comp, 'switch_time'):
            print(f"    Switch time: {comp.switch_time}")
        if hasattr(comp, 'resistance'):
            print(f"    Resistance: {comp.resistance}")
        if hasattr(comp, 'capacitance'):
            print(f"    Capacitance: {comp.capacitance}")
        if hasattr(comp, 'voltage'):
            print(f"    Voltage: {comp.voltage}")

if __name__ == "__main__":
    debug_graph_construction()