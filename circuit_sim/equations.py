import sympy as sp

def extract_input_and_state_vars(circuit_components, voltage_vars, current_vars):
    """
    Extracts input and state variables, defining helper variables for clarity.

    Parameters:
    - circuit_components: { component_id: { "type": str, "value": float, "terminals": {terminal_id: electrical_node} } }
    - voltage_vars: { electrical_node_id: voltage_variable }
    - current_vars: { component_id: current_variable }
    
    Returns:
    - state_vars: Dictionary of helper state variables and their equations.
    - input_vars: Dictionary of helper input variables and their equations.
    """
    state_vars = {}
    input_vars = {}

    for comp_id, comp_data in circuit_components.items():
        comp_type = comp_data["type"]
        terminals = comp_data["terminals"]

        if comp_type == "capacitor":
            # Capacitor voltage state variable
            if len(terminals) == 2:
                node_a, node_b = terminals.values()
                v_a = voltage_vars.get(node_a, sp.Symbol(f"V_{node_a}"))
                v_b = voltage_vars.get(node_b, sp.Symbol(f"V_{node_b}"))

                helper_var = sp.Symbol(f"V_{comp_id}")  # Helper variable for capacitor voltage
                state_vars[helper_var] = v_a - v_b  # v_s_dndnode7 = V_A - V_B

        elif comp_type == "inductor":
            # Inductor current state variable
            if comp_id in current_vars:
                helper_var = sp.Symbol(f"I_{comp_id}")  # Helper variable for inductor current
                state_vars[helper_var] = current_vars[comp_id]  # i_s_dndnode12 = I_dndnode12

        elif comp_type == "voltage-source":
            # Voltage source input variable
            if len(terminals) == 2:
                node_a, node_b = terminals.values()
                v_a = voltage_vars.get(node_a, sp.Symbol(f"V_{node_a}"))
                v_b = voltage_vars.get(node_b, sp.Symbol(f"V_{node_b}"))

                helper_var = sp.Symbol(f"V_in_{comp_id}")  # Helper variable for voltage source
                input_vars[helper_var] = v_a - v_b  # v_in_dndnode5 = V_A - V_B

        elif comp_type == "current-source":
            # Current source input variable
            if comp_id in current_vars:
                helper_var = sp.Symbol(f"I_in_{comp_id}")  # Helper variable for current source
                input_vars[helper_var] = current_vars[comp_id]  # i_in_dndnode12 = I_dndnode12

    return state_vars, input_vars



def write_kcl_equations(electrical_nodes, current_vars, circuit_components, ground_node):
    """
    Generates Kirchhoff’s Current Law (KCL) equations for electrical nodes.

    Parameters:
    - electrical_nodes: { electrical_node_id: set((component_id, terminal_id), ...) }
    - current_vars: { component_id: current_variable }
    - circuit_components: { component_id: { "type": str, "value": float, "terminals": {terminal_id: electrical_node} } }
    - ground_node: The chosen ground node (ignored for KCL)

    Returns:
    - kcl_equations: List of KCL equations in symbolic form
    """
    kcl_equations = []
    
    # Identify voltage sources in the cleaned components
    voltage_sources = {comp_id: comp for comp_id, comp in circuit_components.items() if comp["type"] == "voltage-source"}

    # Step 1: Identify supernodes (voltage sources connecting non-ground nodes)
    supernodes = {}  # Maps supernodes to merged electrical nodes
    for vs_id, vs in voltage_sources.items():
        connected_nodes = set(vs["terminals"].values())
        # print(f"Voltage Source {vs_id} - Connected Nodes: {connected_nodes}")

        # Only create a supernode if the voltage source connects **two non-ground nodes**
        if ground_node not in connected_nodes:
            # print(f"Creating supernode for {vs_id}...")
            supernode_id = f"supernode_{vs_id}"
            supernodes[supernode_id] = connected_nodes

    # Step 2: Write KCL for normal electrical nodes (excluding ground, supernodes, and voltage-source-connected-to-ground nodes)
    for node_id, terminals in electrical_nodes.items():
        print(f"Node {node_id} - Terminals: {terminals}")    

        if node_id == ground_node:
            # print(f"{node_id} - Ground node detected, skipping...")
            continue  # Ignore ground node

        # Skip nodes that belong to a supernode
        if any(node_id in nodes for nodes in supernodes.values()):
            # print(f"{node_id} - Supernode detected, skipping...")
            continue

        # Skip nodes connected to a voltage source where the other terminal is grounded
        skip_node = False
        for vs_id, vs in voltage_sources.items():
            connected_nodes = set(vs["terminals"].values())
            if node_id in connected_nodes and ground_node in connected_nodes:
                # print(f"{node_id} - Voltage source connected to ground detected, skipping...")
                skip_node = True
                break

        if skip_node:
            continue

        equation = 0  # Initialize symbolic equation

        # Iterate over terminals in this electrical node
        for comp_id, terminal_id in terminals:
            if comp_id not in circuit_components:
                continue

            comp_data = circuit_components[comp_id]
            comp_type = comp_data["type"]
            value = sp.Symbol(f"{comp_id}_value") if comp_data["value"] is not None else None

            # Express the current depending on the component type
            if comp_id in current_vars:
                if terminal_id == "0":  # Convention: Current entering the node is positive
                    equation += current_vars[comp_id]
                elif terminal_id == "1":  # Current leaving the node is negative
                    equation -= current_vars[comp_id]

            # Warning for unknown component types
            # else:
                # print(f"⚠ Warning: Unknown component type '{comp_type}' for {comp_id} in node {node_id}, kcl equation may be incorrect.")
                

        kcl_equations.append(equation)

    # Step 3: Write KCL for supernodes ### Todo: Fix this
    for supernode_id, nodes in supernodes.items():
        # print(f"Supernode {supernode_id} - Nodes: {nodes}")
        equation = 0  # Initialize symbolic equation

        for node in nodes:
            if node in electrical_nodes:
                for comp_id, terminal_id in electrical_nodes[node]:
                    if comp_id not in circuit_components:
                        continue

                    comp_data = circuit_components[comp_id]
                    comp_type = comp_data["type"]
                    value = sp.Symbol(f"{comp_id}_value") if comp_data["value"] is not None else None

                    # Express current for each component type
                    if comp_id in current_vars:
                        equation += current_vars[comp_id]

                    # else:
                    #     print(f"⚠ Warning: Unknown component type '{comp_type}' for {comp_id} in supernode {supernode_id}, node {node_id} kcl equation may be incorrect.")


        kcl_equations.append(equation)  # Store symbolic equation

    return kcl_equations