import pytest
import json
import networkx as nx
from amps_simulation.core.parser_networkx import ParserJson
from amps_simulation.core.components import (
    Resistor, Capacitor, Inductor, PowerSwitch,
    VoltageSource, Ground
)

def print_graph(graph, title="Graph Structure"):
    """Helper function to print graph structure."""
    print(f"\n=== {title} ===")
    print("\nNodes:", graph.nodes)
    print("\nEdges:")
    for u, v, d in graph.edges(data=True):
        comp = d["component"]
        print(f"  {u} -> {v}: {comp.__class__.__name__} (id: {comp.comp_id})")

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
    # We expect 5 electrical junctions:
    # 1. V2 top/S2 left
    # 2. S2 right/R7 left
    # 3. R7 right/C4 top/L2 left
    # 4. V2 bottom/C4 bottom/GND1
    # 5. L2 right
    assert len(graph.nodes) == 5
    
    # Check node attributes
    for node in graph.nodes:
        assert graph.nodes[node]["type"] == "electrical_node"
        # Verify nodes are numbered sequentially
        assert node.isdigit()
        assert 1 <= int(node) <= 5
    
    # Check that we have the correct number of edges (components)
    # We expect 5 edges (one for each two-terminal component)
    assert len(graph.edges) == 5
    
    # Check that all nodes are connected (no isolated nodes)
    assert nx.is_connected(graph.to_undirected())
    
    # Check component types in edges
    # Find edges with specific components by checking edge attributes
    voltage_source_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                           if isinstance(d["component"], VoltageSource)]
    assert len(voltage_source_edges) == 1  # One voltage source
    
    resistor_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                     if isinstance(d["component"], Resistor)]
    assert len(resistor_edges) == 1  # One resistor
    
    capacitor_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                      if isinstance(d["component"], Capacitor)]
    assert len(capacitor_edges) == 1  # One capacitor
    
    inductor_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                     if isinstance(d["component"], Inductor)]
    assert len(inductor_edges) == 1  # One inductor
    
    switch_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                   if isinstance(d["component"], PowerSwitch)]
    assert len(switch_edges) == 1  # One switch 