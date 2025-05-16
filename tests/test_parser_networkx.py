import pytest
import json
import networkx as nx
from amps_simulation.core.parser_networkx import ParserJson
from amps_simulation.core.components import (
    Resistor, Capacitor, Inductor, PowerSwitch,
    VoltageSource, Ground, Component, CurrentSource, Diode, Ammeter, Voltmeter
)

def load_test_file(filename: str) -> dict:
    """Load a test circuit file from the test_data directory."""
    with open(f"test_data/{filename}", "r") as f:
        return json.load(f)

def print_graph(graph, title="Graph Structure"):
    """Helper function to print graph structure."""
    print(f"\n=== {title} ===")
    print("\nNodes:", graph.nodes)
        # Print detailed node information
    print("\nDetailed Node Information:")
    for node in graph.nodes:
        print(f"Node {node}: {graph.nodes[node]}")
        
    print("\nEdges:", graph.edges)
    print("\nDetailed Edge Information:")
    for u, v, d in graph.edges(data=True):
        comp = d["component"]
        print(f"\n  {u} -> {v}:")
        print(f"    Component Type: {comp.__class__.__name__}")
        print(f"    Component ID: {comp.comp_id}")
        # Print component-specific attributes
        if hasattr(comp, "resistance"):
            print(f"    Resistance: {comp.resistance}")
        if hasattr(comp, "capacitance"):
            print(f"    Capacitance: {comp.capacitance}")
        if hasattr(comp, "inductance"):
            print(f"    Inductance: {comp.inductance}")
        if hasattr(comp, "voltage"):
            print(f"    Voltage: {comp.voltage}")
        if hasattr(comp, "current"):
            print(f"    Current: {comp.current}")

def test_parser_json_creates_correct_graph():
    """Test parsing a complex circuit from JSON file into an electrical graph."""
    # Clear the component registry to avoid duplicate comp_id issues
    Component.clear_registry()

    # Load the test circuit
    with open("test_data/parser_nodes.json", "r") as f:
        circuit_json = json.load(f)
    
    # Create parser and parse circuit
    parser = ParserJson()
    graph = parser.parse(circuit_json)
    print_graph(graph, "Complex Circuit")
    
    # Print node mapping for debugging
    print("\nComponent Connections:")
    for conn in circuit_json["edges"]:
        print(f"  {conn['source']}:{conn['sourceHandle']} -> {conn['target']}:{conn['targetHandle']}")
    
    # Check that we have the correct number of nodes (electrical junctions)
    # We expect 6 electrical junctions:
    # 1. V2 top/S2 left
    # 2. S2 right/R7 left
    # 3. R7 right/C4 top/L2 left
    # 4. V2 bottom/C4 bottom/GND1/GND2 (merged ground node)
    # 5. L2 right
    # 6. R8/C5 connection point
    assert len(graph.nodes) == 6
    
    # Check node attributes
    for node in graph.nodes:
        assert "junction" in graph.nodes[node]  # Check for junction attribute
        # Verify nodes are numbered sequentially
        assert node.isdigit()
        assert 1 <= int(node) <= 6
    
    # Check that we have the correct number of edges (components)
    # We expect 7 edges (one for each two-terminal component)
    assert len(graph.edges) == 7
    
    # Check that all nodes are connected (no isolated nodes)
    assert nx.is_connected(graph.to_undirected())
    
    # Check component types in edges
    # Find edges with specific components by checking edge attributes
    voltage_source_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                           if isinstance(d["component"], VoltageSource)]
    assert len(voltage_source_edges) == 1  # One voltage source
    
    resistor_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                     if isinstance(d["component"], Resistor)]
    assert len(resistor_edges) == 2  # Two resistors (R7, R8)
    
    capacitor_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                      if isinstance(d["component"], Capacitor)]
    assert len(capacitor_edges) == 2  # Two capacitors (C4, C5)
    
    inductor_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                     if isinstance(d["component"], Inductor)]
    assert len(inductor_edges) == 1  # One inductor
    
    switch_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                   if isinstance(d["component"], PowerSwitch)]
    assert len(switch_edges) == 1  # One switch


def test_parser_networkx_all_components() -> None:
    """
    Test ParserJson from parser_networkx.py with a circuit containing all supported components.
    
    Given:
        - A circuit description in JSON format containing all supported components
        - The circuit contains voltage source, resistor, inductor, capacitor, ground, and switch
        
    When:
        - The circuit is parsed by ParserJson
        
    Then:
        - All components are correctly created with their proper types and values
        - The graph structure is properly initialized
        
    Raises:
        AssertionError: If any validation check fails
        Exception: If parsing fails
    """
    # Clear the component registry to avoid duplicate comp_id issues
    Component.clear_registry()

    # Load test circuit
    circuit_data = load_test_file("parser_all_components.json")
    
    # Create parser instance
    parser = ParserJson()
    
    # Parse circuit data
    graph = parser.parse(circuit_data)
    print_graph(graph, "All Components Test")
    
    # Get the created components
    components = graph.edges(data=True)

    # Verify graph is initialized
    assert isinstance(graph, nx.MultiDiGraph), "Parser should return a directed graph"
    
    # Verify we have the expected number of components
    assert len(components) == 9, "Should have 9 components (Vin, Iin, S, D, R, C, L, Am, Vm)"
    
    type_list = ['VoltageSource', 'CurrentSource', 'PowerSwitch', 'Diode', 'Resistor', 'Capacitor', 'Inductor', 'Ammeter', 'Voltmeter']
    
    # Verify that the components are of the correct type
    for comp in components:
        assert isinstance(comp[2]['component'], Component), "Component should be of type Component"
    
    # Create a mapping from type names to actual component classes
    component_type_map = {
        'VoltageSource': VoltageSource,
        'CurrentSource': CurrentSource,
        'PowerSwitch': PowerSwitch,
        'Diode': Diode,
        'Resistor': Resistor,
        'Capacitor': Capacitor,
        'Inductor': Inductor,
        'Ground': Ground,
        'Ammeter': Ammeter,
        'Voltmeter': Voltmeter
    }

    # Verify that each type appears exactly once
    for t in type_list:
        count = 0  # Initialize count for each type
        for comp in components:
            if isinstance(comp[2]['component'], component_type_map[t]):
                count += 1
        assert count == 1, f"Component type {t} should appear exactly once"

# def test_parser_networkx_has_ground_node() -> None:
#     """Test that the parser creates a ground node."""
#     # Clear the component registry to avoid duplicate comp_id issues
#     Component.clear_registry()

#     # Load test circuit
#     circuit_data = load_test_file("parser_no_ground_node.json")