#!/usr/bin/env python3
"""
Debug script to test the _add_shunt_resistors_to_diodes method.

This script creates a simple circuit with diodes and tests adding shunt resistors.
"""

import sys
import os
import logging

# Add the parent directory to the path so we can import amps_simulation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from amps_simulation.core.electrical_model import ElectricalModel
from amps_simulation.core.dae_system import ElectricalDaeSystem
from amps_simulation.core.components import Resistor, Diode, VoltageSource
import matplotlib.pyplot as plt
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def visualize_circuit_comparison(original_model, shunt_model, title_prefix="Circuit"):
    """Visualize and compare original circuit with shunt resistor version."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Original circuit
    pos1 = nx.spring_layout(original_model.graph, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(original_model.graph, pos1, ax=ax1,
                          node_color='lightblue', node_size=800)

    # Draw edges with different colors for different components
    # Group edges by type to handle multiple edges between same nodes
    diode_edges = []
    resistor_edges = []
    voltage_source_edges = []
    other_edges = []

    for source, target, key, edge_data in original_model.graph.edges(data=True, keys=True):
        component = edge_data.get('component')
        edge_tuple = (source, target, key)
        if isinstance(component, Diode):
            diode_edges.append(edge_tuple)
        elif isinstance(component, Resistor):
            resistor_edges.append(edge_tuple)
        elif isinstance(component, VoltageSource):
            voltage_source_edges.append(edge_tuple)
        else:
            other_edges.append(edge_tuple)

    # Draw each edge type with different styles
    if diode_edges:
        nx.draw_networkx_edges(original_model.graph, pos1, diode_edges,
                             ax=ax1, edge_color='red', width=3, label='Diode')
    if resistor_edges:
        nx.draw_networkx_edges(original_model.graph, pos1, resistor_edges,
                             ax=ax1, edge_color='green', width=2, label='Resistor')
    if voltage_source_edges:
        nx.draw_networkx_edges(original_model.graph, pos1, voltage_source_edges,
                             ax=ax1, edge_color='blue', width=2, label='Voltage Source')
    if other_edges:
        nx.draw_networkx_edges(original_model.graph, pos1, other_edges,
                             ax=ax1, edge_color='black', width=1)

    # Add node labels
    nx.draw_networkx_labels(original_model.graph, pos1, ax=ax1)

    # Add edge labels with component info (for original model, there should be no parallel edges)
    edge_labels = {}
    for source, target, edge_data in original_model.graph.edges(data=True):
        component = edge_data.get('component')
        if component:
            comp_type = component.__class__.__name__
            comp_id = component.comp_id
            if hasattr(component, 'resistance'):
                edge_labels[(source, target)] = f"{comp_id}\n{comp_type}\n{component.resistance}Ω"
            elif hasattr(component, 'voltage'):
                edge_labels[(source, target)] = f"{comp_id}\n{comp_type}\n{component.voltage}V"
            else:
                edge_labels[(source, target)] = f"{comp_id}\n{comp_type}"

    nx.draw_networkx_edge_labels(original_model.graph, pos1, edge_labels, ax=ax1, font_size=8)

    ax1.set_title(f"{title_prefix} - Original")
    ax1.axis('off')

    # Shunt resistor circuit (use same layout for comparison)
    pos2 = nx.spring_layout(shunt_model.graph, pos=pos1, fixed=list(pos1.keys()), seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(shunt_model.graph, pos2, ax=ax2,
                          node_color='lightblue', node_size=800)

    # Draw edges with different colors for different components
    # Group edges by type to handle multiple edges between same nodes
    diode_edges2 = []
    regular_resistor_edges2 = []
    shunt_resistor_edges2 = []
    voltage_source_edges2 = []
    other_edges2 = []

    for source, target, key, edge_data in shunt_model.graph.edges(data=True, keys=True):
        component = edge_data.get('component')
        edge_tuple = (source, target, key)
        if isinstance(component, Diode):
            diode_edges2.append(edge_tuple)
        elif isinstance(component, Resistor):
            if '_shunt' in component.comp_id:
                shunt_resistor_edges2.append(edge_tuple)
            else:
                regular_resistor_edges2.append(edge_tuple)
        elif isinstance(component, VoltageSource):
            voltage_source_edges2.append(edge_tuple)
        else:
            other_edges2.append(edge_tuple)

    # Draw each edge type with different styles
    if diode_edges2:
        nx.draw_networkx_edges(shunt_model.graph, pos2, diode_edges2,
                             ax=ax2, edge_color='red', width=3, label='Diode')
    if regular_resistor_edges2:
        nx.draw_networkx_edges(shunt_model.graph, pos2, regular_resistor_edges2,
                             ax=ax2, edge_color='green', width=2, label='Resistor')
    if shunt_resistor_edges2:
        # Use connectionstyle to make parallel edges visible
        nx.draw_networkx_edges(shunt_model.graph, pos2, shunt_resistor_edges2,
                             ax=ax2, edge_color='orange', width=2, style='dashed',
                             connectionstyle="arc3,rad=0.1", label='Shunt Resistor')
    if voltage_source_edges2:
        nx.draw_networkx_edges(shunt_model.graph, pos2, voltage_source_edges2,
                             ax=ax2, edge_color='blue', width=2, label='Voltage Source')
    if other_edges2:
        nx.draw_networkx_edges(shunt_model.graph, pos2, other_edges2,
                             ax=ax2, edge_color='black', width=1)

    # Add node labels
    nx.draw_networkx_labels(shunt_model.graph, pos2, ax=ax2)

    # Add edge labels with component info (handle multiple edges between same nodes)
    edge_labels2 = {}

    # For shunt model, we need to handle multiple edges between same nodes
    edge_components = {}  # (source, target) -> [list of components]

    for source, target, edge_data in shunt_model.graph.edges(data=True):
        component = edge_data.get('component')
        if component:
            key = (source, target)
            if key not in edge_components:
                edge_components[key] = []
            edge_components[key].append(component)

    # Create labels that show all components between nodes
    for (source, target), components in edge_components.items():
        if len(components) == 1:
            # Single component - show normally
            component = components[0]
            comp_type = component.__class__.__name__
            comp_id = component.comp_id
            if hasattr(component, 'resistance'):
                if component.resistance >= 1e6:
                    edge_labels2[(source, target)] = f"{comp_id}\n{comp_type}\n{component.resistance:.0e}Ω"
                else:
                    edge_labels2[(source, target)] = f"{comp_id}\n{comp_type}\n{component.resistance}Ω"
            elif hasattr(component, 'voltage'):
                edge_labels2[(source, target)] = f"{comp_id}\n{comp_type}\n{component.voltage}V"
            else:
                edge_labels2[(source, target)] = f"{comp_id}\n{comp_type}"
        else:
            # Multiple components - show all
            labels = []
            for component in components:
                comp_type = component.__class__.__name__[:4]  # Shorten type names
                comp_id = component.comp_id
                if hasattr(component, 'resistance'):
                    if component.resistance >= 1e6:
                        labels.append(f"{comp_id}({component.resistance:.0e}Ω)")
                    else:
                        labels.append(f"{comp_id}({component.resistance}Ω)")
                elif hasattr(component, 'voltage'):
                    labels.append(f"{comp_id}({component.voltage}V)")
                else:
                    labels.append(f"{comp_id}")
            edge_labels2[(source, target)] = "\n".join(labels)

    nx.draw_networkx_edge_labels(shunt_model.graph, pos2, edge_labels2, ax=ax2, font_size=7)

    ax2.set_title(f"{title_prefix} - With Shunt Resistors")
    ax2.axis('off')

    # Add legends (create custom legends to avoid duplicates)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='Diode'),
        Line2D([0], [0], color='green', lw=2, label='Resistor'),
        Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Shunt Resistor'),
        Line2D([0], [0], color='blue', lw=2, label='Voltage Source')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    # Save the plot
    filename = f"debug/{title_prefix.lower().replace(' ', '_')}_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Circuit comparison saved as: {filename}")

    # Show the plot (comment out to avoid hanging in scripts)
    # plt.show()

    return fig

def create_simple_diode_circuit():
    """Create a simple circuit with diodes using ElectricalModel methods."""
    print("Creating simple diode circuit...")

    # Clear component and junction registries first to avoid conflicts
    from amps_simulation.core.components import Component, ElecJunction
    Component.clear_registry()
    ElecJunction.clear_registry()

    # Create empty electrical model
    model = ElectricalModel()

    # Add nodes (using integer IDs as required by ElecJunction)
    model.add_node(1, is_ground=False)  # n1
    model.add_node(2, is_ground=False)  # n2
    model.add_node(3, is_ground=False)  # n3
    model.add_node(0, is_ground=True)   # ground

    # Create components
    v_source = VoltageSource(comp_id="V1", voltage=10.0)
    r1 = Resistor(comp_id="R1", resistance=100.0)
    diode = Diode(comp_id="D1", is_on=False)
    r_load = Resistor(comp_id="R_load", resistance=1000.0)

    # Add components with terminal connections
    model.add_component(v_source, p=1, n=0)    # V1: n1 to gnd
    model.add_component(r1, p=1, n=2)          # R1: n1 to n2
    model.add_component(diode, p=2, n=3)       # D1: n2 to n3
    model.add_component(r_load, p=3, n=0)      # R_load: n3 to gnd

    return model

def create_multi_diode_circuit():
    """Create a bridge rectifier circuit with multiple diodes using ElectricalModel methods."""
    print("Creating multi-diode bridge rectifier...")

    # Clear component and junction registries
    from amps_simulation.core.components import Component, ElecJunction
    Component.clear_registry()
    ElecJunction.clear_registry()

    # Create empty electrical model
    model = ElectricalModel()

    # Add nodes (using integer IDs)
    model.add_node(1, is_ground=False)  # ac_pos
    model.add_node(2, is_ground=False)  # ac_neg
    model.add_node(3, is_ground=False)  # dc_pos
    model.add_node(0, is_ground=True)   # dc_neg (ground)

    # Create components
    v_ac = VoltageSource(comp_id="V_ac", voltage=15.0)
    diode_1 = Diode(comp_id="D1", is_on=False)
    diode_2 = Diode(comp_id="D2", is_on=False)
    diode_3 = Diode(comp_id="D3", is_on=False)
    diode_4 = Diode(comp_id="D4", is_on=False)
    r_load = Resistor(comp_id="R_load", resistance=1000.0)

    # Add components with terminal connections
    model.add_component(v_ac, p=1, n=2)        # V_ac: ac_pos to ac_neg
    model.add_component(diode_1, p=1, n=3)     # D1: ac_pos to dc_pos
    model.add_component(diode_2, p=2, n=3)     # D2: ac_neg to dc_pos
    model.add_component(diode_3, p=0, n=1)     # D3: ac_pos to dc_neg
    model.add_component(diode_4, p=0, n=2)     # D4: ac_neg to dc_neg
    model.add_component(r_load, p=3, n=0)      # R_load: dc_pos to dc_neg

    return model

def test_shunt_resistor_addition():
    """Test the _add_shunt_resistors_to_diodes method."""

    print("\n" + "="*60)
    print("TESTING SHUNT RESISTOR ADDITION")
    print("="*60)

    # Test with simple diode circuit
    print("\n1. Testing with simple diode circuit...")
    electrical_model1 = create_simple_diode_circuit()
    electrical_model1.initialize()
    dae_system1 = ElectricalDaeSystem(electrical_model1)

    print(f"Original circuit has {len(electrical_model1.diode_list)} diodes")
    print(f"Original circuit has {electrical_model1.graph.number_of_edges()} edges")

    for diode in electrical_model1.diode_list:
        print(f"  - Diode {diode.comp_id}")

    # Add shunt resistors
    R_shunt = 1e6  # 1 MΩ shunt resistors
    print(f"\nAdding shunt resistors with R = {R_shunt:.0e} Ω...")

    try:
        shunt_model1 = dae_system1._add_shunt_resistors_to_diodes(R_shunt)
        print(f"Shunt model has {shunt_model1.graph.number_of_edges()} edges")

        # List all components in shunt model
        print("\nComponents in shunt model:")
        diode_count = 0
        shunt_count = 0
        for source, target, edge_data in shunt_model1.graph.edges(data=True):
            component = edge_data.get('component')
            if component:
                comp_type = component.__class__.__name__
                comp_id = component.comp_id
                if isinstance(component, Diode):
                    diode_count += 1
                    print(f"  - {comp_type} {comp_id} (2->3)")
                elif isinstance(component, Resistor) and '_shunt' in component.comp_id:
                    shunt_count += 1
                    print(f"  - {comp_type} {comp_id}: R = {component.resistance} Ω (2->3) [PARALLEL TO DIODE]")
                elif hasattr(component, 'resistance'):
                    print(f"  - {comp_type} {comp_id}: R = {component.resistance} Ω ({source} -> {target})")
                else:
                    print(f"  - {comp_type} {comp_id} ({source} -> {target})")

        print(f"\nVerification: Found {diode_count} diodes and {shunt_count} shunt resistors in parallel")

        print("✓ Simple diode circuit test PASSED")

        # Visualize the comparison
        print("\nGenerating visualization for simple diode circuit...")
        try:
            visualize_circuit_comparison(electrical_model1, shunt_model1, "Simple Diode Circuit")
        except Exception as viz_error:
            print(f"Warning: Could not generate visualization: {viz_error}")

    except Exception as e:
        print(f"✗ Simple diode circuit test FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test with multi-diode circuit
    print("\n2. Testing with multi-diode bridge rectifier...")
    electrical_model2 = create_multi_diode_circuit()
    electrical_model2.initialize()
    dae_system2 = ElectricalDaeSystem(electrical_model2)

    print(f"Original bridge circuit has {len(electrical_model2.diode_list)} diodes")
    print(f"Original bridge circuit has {electrical_model2.graph.number_of_edges()} edges")

    for diode in electrical_model2.diode_list:
        print(f"  - Diode {diode.comp_id}")

    # Add shunt resistors
    R_shunt = 1e9  # 1 GΩ shunt resistors (very high for numerical stability)
    print(f"\nAdding shunt resistors with R = {R_shunt:.0e} Ω...")

    try:
        shunt_model2 = dae_system2._add_shunt_resistors_to_diodes(R_shunt)
        print(f"Shunt model has {shunt_model2.graph.number_of_edges()} edges")

        # Count diodes and shunt resistors
        diode_count = 0
        shunt_count = 0
        for source, target, edge_data in shunt_model2.graph.edges(data=True):
            component = edge_data.get('component')
            if component:
                if isinstance(component, Diode):
                    diode_count += 1
                elif isinstance(component, Resistor) and '_shunt' in component.comp_id:
                    shunt_count += 1

        print(f"Shunt model contains {diode_count} diodes and {shunt_count} shunt resistors")

        # Verify each diode has a corresponding shunt resistor
        expected_shunt_count = len(electrical_model2.diode_list)
        if shunt_count == expected_shunt_count:
            print("✓ Correct number of shunt resistors added")
        else:
            print(f"✗ Expected {expected_shunt_count} shunt resistors, got {shunt_count}")

        print("✓ Multi-diode circuit test PASSED")

        # Visualize the comparison
        print("\nGenerating visualization for bridge rectifier...")
        try:
            visualize_circuit_comparison(electrical_model2, shunt_model2, "Bridge Rectifier Circuit")
        except Exception as viz_error:
            print(f"Warning: Could not generate visualization: {viz_error}")

    except Exception as e:
        print(f"✗ Multi-diode circuit test FAILED: {e}")
        import traceback
        traceback.print_exc()

def test_edge_cases():
    """Test edge cases for shunt resistor addition."""

    print("\n" + "="*60)
    print("TESTING EDGE CASES")
    print("="*60)

    # Test with circuit that has no diodes
    print("\n1. Testing with circuit containing no diodes...")

    # Create a simple RC circuit (no diodes) using ElectricalModel methods
    from amps_simulation.core.components import Component, ElecJunction
    Component.clear_registry()
    ElecJunction.clear_registry()

    # Create empty electrical model
    electrical_model = ElectricalModel()

    # Add nodes
    electrical_model.add_node(1, is_ground=False)  # n1
    electrical_model.add_node(0, is_ground=True)   # ground

    # Create components
    v_source = VoltageSource(comp_id="V1", voltage=5.0)
    r1 = Resistor(comp_id="R1", resistance=1000.0)

    # Add components
    electrical_model.add_component(v_source, p=1, n=0)
    electrical_model.add_component(r1, p=1, n=0)
    electrical_model.initialize()
    dae_system = ElectricalDaeSystem(electrical_model)

    print(f"Circuit has {len(electrical_model.diode_list)} diodes")

    try:
        original_edge_count = electrical_model.graph.number_of_edges()
        shunt_model = dae_system._add_shunt_resistors_to_diodes(1e6)

        if shunt_model.graph.number_of_edges() == original_edge_count:
            print("✓ No diodes circuit test PASSED - no changes made")
        else:
            print("✗ No diodes circuit test FAILED - unexpected changes made")

    except Exception as e:
        print(f"✗ No diodes circuit test FAILED: {e}")

def main():
    """Main function to run all tests."""

    print("SHUNT RESISTOR DEBUG SCRIPT")
    print("="*60)
    print("Testing the _add_shunt_resistors_to_diodes method")

    try:
        test_shunt_resistor_addition()
        test_edge_cases()

        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)

    except Exception as e:
        print(f"CRITICAL ERROR in debug script: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()