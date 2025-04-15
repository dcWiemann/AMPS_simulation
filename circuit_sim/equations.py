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