import sympy as sp
from collections import defaultdict

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



def find_loops(electrical_nodes, circuit_components):
    """
    Find closed loops in the circuit using Depth-First Search (DFS).

    Parameters:
    - electrical_nodes: { electrical_node_id: set((component_id, terminal_id), ...) }

    Returns:
    - loops: A list of loops, where each loop is a list of electrical nodes forming a cycle.
    """
    # Step 1: Build adjacency list for electrical nodes
    graph = defaultdict(set)

    for node_id, terminals in electrical_nodes.items():
        # For each terminal, find its component and connected node
        for comp_id, terminal_id in terminals:
            comp_data = circuit_components.get(comp_id)
            if not comp_data:
                continue  # Skip if component is not found

            # Get the other terminal of the component
            for other_terminal_id, other_node_id in comp_data["terminals"].items():
                if other_node_id != node_id:  # Ensure it's a different node
                    graph[node_id].add(other_node_id)
                    graph[other_node_id].add(node_id)  # Bidirectional edge

    # Step 2: Find loops using DFS
    loops = []
    visited = set()

    def dfs(node, path, start_node):
        """Recursive DFS to find loops."""
        for neighbor in graph[node]:
            if neighbor == start_node and len(path) > 2:
                # Found a valid loop, normalize order to prevent duplicates
                loop = sorted(path[:])
                if loop not in loops:
                    loops.append(loop)
                continue

            if neighbor not in path:  # Prevent backtracking
                dfs(neighbor, path + [neighbor], start_node)

    # Step 3: Start DFS from each node
    for node in graph:
        if node not in visited:
            dfs(node, [node], node)
            visited.add(node)

    return loops



def write_kvl_equations(loops, voltage_vars, circuit_components, current_vars):
    """
    Generate Kirchhoff's Voltage Law (KVL) equations using component voltage relations.

    Parameters:
    - loops: List of detected loops, each represented as a list of electrical nodes.
    - voltage_vars: { electrical_node_id: voltage_variable } (dictionary of node voltage symbols)
    - circuit_components: { component_id: { "type": str, "value": float, "terminals": {terminal_id: electrical_node} } }
    - current_vars: { component_id: current_variable } (Dictionary mapping components to symbolic current variables)

    Returns:
    - kvl_equations: List of symbolic KVL equations.
    """
    kvl_equations = []

    for loop in loops:
        equation = 0  # Initialize symbolic equation

        # Iterate through the loop, processing components
        for i in range(len(loop)):
            node_a = loop[i]
            node_b = loop[(i + 1) % len(loop)]  # Next node in the loop (wraps around)

            # Find the component connecting these two nodes
            component_id = None
            component = None
            terminal_a = None
            terminal_b = None

            for comp_id, comp_data in circuit_components.items():
                terminals = comp_data["terminals"]
                if set(terminals.values()) == {node_a, node_b}:  # Component connects these two nodes
                    component_id = comp_id
                    component = comp_data
                    # Identify which terminal corresponds to which node
                    for term_id, node in terminals.items():
                        if node == node_a:
                            terminal_a = term_id
                        elif node == node_b:
                            terminal_b = term_id
                    break

            if component is None:
                continue  # No component directly connects these nodes

            comp_type = component["type"]
            value = sp.Symbol(f"{component_id}_value") if component["value"] is not None else None

            # Determine voltage drop direction based on terminal convention
            v_a = voltage_vars.get(node_a, sp.Symbol(f"V_{node_a}"))
            v_b = voltage_vars.get(node_b, sp.Symbol(f"V_{node_b}"))
            voltage_diff = v_a - v_b if terminal_a == "0" else v_b - v_a  # Polarity correction

            # Apply voltage relation based on component type
            if comp_type == "resistor":
                equation += (value * current_vars[component_id]) if terminal_a == "0" else (-value * current_vars[component_id]) 

            elif comp_type == "capacitor":
                equation += sp.Symbol(f"V_{component_id}") if terminal_a == "0" else -sp.Symbol(f"V_{component_id}")

            elif comp_type == "inductor":
                equation += (value * sp.Symbol(f"d{current_vars[component_id]}_dt")) if terminal_a == "0" else (-value * sp.Symbol(f"d{current_vars[component_id]}_dt"))

            elif comp_type == "voltage-source":
                equation += sp.Symbol(f"V_in_{component_id}") if terminal_a == "0" else -sp.Symbol(f"V_in_{component_id}")

            else:
                print(f"⚠ Warning: Unknown component type '{comp_type}' for {component_id}, ignoring in KVL.")

        # Kirchhoff's Voltage Law states ΣV = 0
        kvl_equations.append(equation)

    return kvl_equations


def solve_helper_variables(kcl_eqs, kvl_eqs, voltage_vars, current_vars, state_vars, input_vars, circuit_components):
    """
    Eliminates helper variables by solving for them in terms of state variables and input variables.

    Parameters:
    - kcl_eqs: List of symbolic KCL equations.
    - kvl_eqs: List of symbolic KVL equations.
    - voltage_vars: Dictionary mapping electrical nodes to voltage variables.
    - current_vars: Dictionary mapping components to current variables.
    - state_vars: Dictionary mapping state variable names to their expressions.
    - input_vars: Dictionary mapping input variable names to their expressions.
    - circuit_components: Dictionary containing cleaned component data.
    - electrical_nodes: Dictionary mapping node indices to connected components.

    Returns:
    - Reduced system of KCL and KVL equations without helper variables, fully expressed in terms of state and input variables.
    """

    # Step 1: Identify helper variables (present in voltage_vars or current_vars but not in state_vars)
    helper_vars = set(voltage_vars.values()).union(set(current_vars.values())) - set(state_vars.keys())

    # Step 2: Generate equations for resistors and capacitors 
    helper_eqs = []
    for comp_id, comp_data in circuit_components.items():
        if comp_data["type"] == "resistor":
            r_value = sp.Symbol(f"{comp_id}_value")  # Symbolic resistance value
            i_r = current_vars[comp_id]  # Current through the resistor
            
            # Extract node voltages
            terminals = comp_data["terminals"]
            node_1 = terminals["0"]
            node_2 = terminals["1"]
            
            v_node_1 = voltage_vars.get(node_1, f"V_{node_1}")
            v_node_2 = voltage_vars.get(node_2, f"V_{node_2}")
            
            # Ohm’s Law: (V1 - V2) = IR
            helper_eqs.append((v_node_1 - v_node_2) - r_value * i_r)

        elif comp_data["type"] == "capacitor":
            # Capacitor current equation: i = C dv/dt
            c_value = sp.Symbol(f"{comp_id}_value")  # Symbolic capacitance value
            i_c = current_vars[comp_id]  # Current through the capacitor
            
            # Extract terminal nodes
            terminals = comp_data["terminals"]
            node_1 = terminals["0"]
            node_2 = terminals["1"]

            # Get node voltage symbols
            v_node_1 = voltage_vars.get(node_1, f"V_{node_1}")
            v_node_2 = voltage_vars.get(node_2, f"V_{node_2}")

            # Voltage difference across the capacitor
            v_cap = v_node_1 - v_node_2

            # Define the differential equation for capacitor current
            d_v_cap_dt = sp.Symbol(f"dV_{comp_id}_dt")  # Symbol for dv/dt
            helper_eqs.append(i_c - c_value * d_v_cap_dt)

    # Step 3: Solve for helper variables
    solved_helpers = sp.solve(helper_eqs, helper_vars)


    # Convert state_vars and input_vars into equations of the form: expression - variable = 0
    state_eqs = [var - expr for var, expr in state_vars.items()]
    input_eqs = [var - expr for var, expr in input_vars.items()]

    # Solve for node voltages (e.g., V_1 and V_2)
    node_voltage_subs = sp.solve(state_eqs + input_eqs, list(voltage_vars.values()))


    # Step 5: Substitute helper variables into KCL and KVL equations
    reduced_kcl = [eq.subs(solved_helpers) for eq in kcl_eqs]
    reduced_kvl = [eq.subs(solved_helpers) for eq in kvl_eqs]

    # Apply the substitutions to KCL and KVL equations
    reduced_kcl = [eq.subs(solved_helpers).subs(node_voltage_subs) for eq in kcl_eqs]
    reduced_kvl = [eq.subs(solved_helpers).subs(node_voltage_subs) for eq in kvl_eqs]

    # print("Solved Helpers:")
    # print(solved_helpers)
    # print("\nReduced KCL Equations:")
    # print(reduced_kcl)
    # print("\nReduced KVL Equations:")
    # print(reduced_kvl)

    return reduced_kcl, reduced_kvl



def solve_state_derivatives(reduced_kcl, reduced_kvl, state_vars):
    """
    Solves the system of equations for the time derivatives of state variables.

    Parameters:
    - reduced_kcl: List of KCL equations after eliminating helper variables.
    - reduced_kvl: List of KVL equations after eliminating helper variables.
    - state_vars: List of symbolic state variables.

    Returns:
    - Dictionary mapping {state_variable: derivative_expression}
    """
    # Merge KCL and KVL equations
    system_equations = reduced_kcl + reduced_kvl

    # Define derivative symbols
    state_derivatives = [sp.Symbol(f"d{var}_dt") for var in state_vars]
    print("ℹ️ State Derivatives:", state_derivatives)

    # Solve for the time derivatives of state variables
    solved_derivatives = sp.solve(system_equations, state_derivatives)

    return solved_derivatives



def extract_state_space_matrices(state_derivatives, state_vars, input_vars):
    """
    Converts the state derivative dictionary into matrix form and computes Jacobians for A and B.

    Parameters:
    - state_derivatives: Dictionary {d(state_variable)_dt: derivative_expression}.
    - state_vars: Dictionary {state_variable: expression}.
    - input_vars: Dictionary {input_variable: expression}.

    Returns:
    - A: State matrix (Jacobian of dx/dt w.r.t. state variables).
    - B: Input matrix (Jacobian of dx/dt w.r.t. input variables).
    """
    # Convert dictionaries to lists of variables
    state_vars = list(state_vars.keys())  
    input_vars = list(input_vars.keys())  

    # Create symbolic time derivative variables for state variables
    state_derivative_symbols = [sp.Symbol(f"d{var}_dt") for var in state_vars]

    # Ensure all state derivatives exist in the dictionary
    dx_dt_sol = sp.Matrix([state_derivatives.get(derivative) for derivative in state_derivative_symbols])

    # Compute Jacobians
    A = dx_dt_sol.jacobian(state_vars)  # Partial derivatives of dx/dt w.r.t. state variables
    B = dx_dt_sol.jacobian(input_vars)  # Partial derivatives of dx/dt w.r.t. input variables

    return A, B  # Return the computed matrices



def substitute_component_values(expr, components):
    """
    Substitutes numerical values for component parameters in the symbolic equation.

    Parameters:
    - expr: SymPy expression or matrix containing symbolic component parameters.
    - components: List of circuit components from JSON.

    Returns:
    - expr_substituted: Expression with component values replaced.
    """
    subs_dict = {}

    for component in components:
        comp_id = component["id"]
        value = component["data"].get("value")  # Get numerical value
        if value is not None:
            subs_dict[sp.Symbol(f"{comp_id}_value")] = value  # Replace symbol with actual value

    return expr.subs(subs_dict)