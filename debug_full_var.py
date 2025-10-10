#!/usr/bin/env python3
"""
Debug script to analyze the full_var1.json circuit and understand why DAE solving fails.
"""

import json
import logging
from amps_simulation.core.parser import ParserJson
from amps_simulation.core.engine import Engine
from amps_simulation.core.electrical_model import ElectricalModel
from amps_simulation.core.components import Component

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

def debug_full_var1():
    """Debug the full_var1.json circuit."""
    print("=== Debugging full_var1.json Circuit ===\n")
    
    # Clear component registry
    Component.clear_registry()
    
    # Load circuit
    with open("test_data/full_var1.json", 'r') as f:
        circuit_data = json.load(f)
    
    print("1. COMPONENTS FROM JSON:")
    print("========================")
    for comp in circuit_data["nodes"]:
        comp_type = comp["data"].get("componentType")
        value = comp["data"].get("value")
        print(f"  {comp['id']}: {comp_type} = {value}")
    
    print("\n2. CONNECTIONS FROM JSON:")
    print("=========================")
    for i, conn in enumerate(circuit_data["edges"]):
        print(f"  Connection {i}: {conn['source']}[{conn.get('sourceHandle')}] -> {conn['target']}[{conn.get('targetHandle')}]")
    
    # Parse circuit
    parser = ParserJson()
    try:
        graph, control_graph = parser.parse(circuit_data)
        print(f"\n3. PARSING SUCCESS")
        print(f"   Graph nodes: {len(graph.nodes())}")
        print(f"   Graph edges: {len(graph.edges())}")
    except Exception as e:
        print(f"\n3. PARSING FAILED: {e}")
        return
    
    # Analyze graph structure
    print(f"\n4. GRAPH ANALYSIS:")
    print(f"==================")
    print(f"Nodes: {list(graph.nodes())}")
    print(f"Edges with components:")
    for i, (source, target, edge_data) in enumerate(graph.edges(data=True)):
        component = edge_data.get('component')
        if component:
            print(f"  {i}: {source} -> {target}: {component.comp_id} ({type(component).__name__})")
    
    # Try to create electrical model and engine
    try:
        electrical_model = ElectricalModel(graph)
        engine = Engine(electrical_model, control_graph)
        print(f"\n5. ENGINE CREATION SUCCESS")
    except Exception as e:
        print(f"\n5. ENGINE CREATION FAILED: {e}")
        return
    
    # Try to initialize engine (this is where the failure occurs)
    try:
        print(f"\n6. ATTEMPTING ENGINE INITIALIZATION...")
        engine.initialize()
        print(f"   SUCCESS!")
        print(f"   State vars: {engine.state_vars}")
        print(f"   Input vars: {engine.input_vars}")
        print(f"   Output vars: {engine.output_vars}")
    except Exception as e:
        print(f"\n6. ENGINE INITIALIZATION FAILED: {e}")
        print(f"   This is expected - the circuit equations cannot be solved")
        
        # Let's examine the electrical model to understand why
        if hasattr(engine, 'electrical_model'):
            model = engine.electrical_model
            try:
                # Get the equations manually to see what's happening
                print(f"\n7. ANALYZING ELECTRICAL MODEL:")
                print(f"===============================")
                
                input_vars = model.find_input_vars()
                output_vars = model.find_output_vars()
                state_vars = model.find_state_vars()
                
                print(f"   Input vars: {input_vars}")
                print(f"   Output vars: {output_vars}")
                print(f"   State vars: {state_vars}")
                
                kcl_eqs = model.compute_kcl_equations()
                kvl_eqs = model.compute_kvl_equations()
                static_eqs = model.compute_static_component_equations()
                switch_eqs = model.compute_switch_equations()
                
                print(f"   KCL equations ({len(kcl_eqs)}): {kcl_eqs}")
                print(f"   KVL equations ({len(kvl_eqs)}): {kvl_eqs}")
                print(f"   Static equations ({len(static_eqs)}): {static_eqs}")
                print(f"   Switch equations ({len(switch_eqs)}): {switch_eqs}")
                
                all_eqs = kcl_eqs + kvl_eqs + static_eqs + switch_eqs
                print(f"   Total equations: {len(all_eqs)}")
                
            except Exception as e2:
                print(f"   Error analyzing model: {e2}")
        
        return
    
    print(f"\n7. CIRCUIT ANALYSIS COMPLETE")

if __name__ == "__main__":
    debug_full_var1()