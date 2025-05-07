import pytest
import json
import networkx as nx
from amps_simulation.core.parser_networkx import ParserJson
from amps_simulation.core.components import (
    Resistor, Capacitor, Inductor, PowerSwitch,
    VoltageSource, Ground
)

def load_test_file(filename: str) -> dict:
    """Load a test circuit file from the test_data directory."""
    with open(f"test_data/{filename}", "r") as f:
        return json.load(f)

def print_graph(graph, title="Graph Structure"):
    """Helper function to print graph structure."""
    print(f"\n=== {title} ===")
    print("\nNodes:", graph.nodes)
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
        assert graph.nodes[node]["type"] == "electrical_node"
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
    # Load test circuit
    circuit_data = load_test_file("parser_all_components.json")
    
    # Create parser instance
    parser = ParserJson()
    
    # Parse circuit data
    graph = parser.parse(circuit_data)
    print_graph(graph, "All Components Test")
    
    # Get the created components
    components = parser.circuit_components
    
    # Verify we have the expected number of components
    assert len(components) == 6, "Should have 6 components (V2, S2, R7, C4, L2, GND1)"
    
    # Verify each component type and value
    component_map = {
        "V2": ("voltage-source", 5),
        "S2": ("powerswitch", 0),
        "R7": ("resistor", 1),
        "C4": ("capacitor", 0.001),
        "L2": ("inductor", 0.001),
        "GND1": ("ground", 0)
    }
    
    for comp_id, (expected_type, expected_value) in component_map.items():
        # Find the component in the list
        component = next((c for c in components if c.comp_id == comp_id), None)
        assert component is not None, f"Component {comp_id} not found"
        
        # Verify component type
        assert component.__class__.__name__.lower() == expected_type.replace("-", ""), \
            f"Component {comp_id} has wrong type"
        
        # Verify component value
        if hasattr(component, "value"):
            assert component.value == expected_value, \
                f"Component {comp_id} has wrong value"
    
    # Verify graph is initialized
    assert isinstance(graph, nx.MultiDiGraph), "Parser should return a directed graph"
