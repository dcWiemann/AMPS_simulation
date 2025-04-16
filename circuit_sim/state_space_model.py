# import parsing and equations functions
from circuit_sim.parsing import build_electrical_nodes, build_circuit_components, assign_voltage_variables, assign_current_variables
from circuit_sim.equations import extract_input_and_state_vars, write_kcl_equations, write_kvl_equations, find_loops, solve_helper_variables, solve_state_derivatives, extract_state_space_matrices, substitute_component_values
from scipy.integrate import solve_ivp
import numpy as np

def extract_differential_equations(circuit_json):
    # Step 1: Parse JSON
    components = circuit_json["nodes"]
    connections = circuit_json["edges"]
    print(components)
    # print(connections)

    # Step 2: Identify electrical nodes
    electrical_nodes = build_electrical_nodes(components, connections)
    print("✅ Electrical nodes:", electrical_nodes)

    circuit_components = build_circuit_components(components, electrical_nodes)
    print("✅ Circuit components:", circuit_components)

    # Step 3: Assign voltage variables for electrical nodes
    voltage_vars, ground_node = assign_voltage_variables(electrical_nodes, circuit_components)

    # Step 4: Assign current variables for components
    current_vars = assign_current_variables(circuit_components)
    # print("voltage_vars:", voltage_vars)
    # print("current_vars:", current_vars)

    state_vars, input_vars = extract_input_and_state_vars(circuit_components, voltage_vars, current_vars)
    print("✅ State variables:", state_vars)
    print("✅ Input variables:", input_vars)

    # Step 5: Write KCL equations
    kcl_equations = write_kcl_equations(electrical_nodes, current_vars, circuit_components, ground_node)

    # Step 6: Write KVL equations
    loops = find_loops(electrical_nodes, circuit_components)
    kvl_equations = write_kvl_equations(loops, voltage_vars, circuit_components, current_vars)

    print("✅ KCL equations:", kcl_equations)
    print("✅ KVL equations:", kvl_equations)

    ### ToDo: throw out identify_state_variables and get_input_vars after debugging
    # state_vars = identify_state_variables(voltage_vars, current_vars, components)
    # print("✅ State variables:", state_vars)

    # input_vars = get_input_vars(components)
    # print("✅ Input variables:", input_vars)

    # Step 7: Solve helper variables ### here
    # passive_eqs = write_passive_component_equations(components, connections, voltage_vars, current_vars)

    reduced_kcl, reduced_kvl = solve_helper_variables(kcl_equations, kvl_equations, voltage_vars, current_vars, state_vars, input_vars, circuit_components)
    print("✅ Reduced KCL equations:", reduced_kcl)
    print("✅ Reduced KVL equations:", reduced_kvl)
    

    # Step 8: Solve for state derivatives
    state_derivatives = solve_state_derivatives(reduced_kcl, reduced_kvl, state_vars)
    print("✅ State derivatives:", state_derivatives)

    # Step 9: Extract state space matrices
    A, B = extract_state_space_matrices(state_derivatives, state_vars, input_vars)
    print("✅ State matrix A:")
    print(A)
    print("✅ Input matrix B:")
    print(B)

    # Substitute numerical values into A and B
    A_substituted = substitute_component_values(A, components)
    B_substituted = substitute_component_values(B, components)

    print("✅ State matrix A (after substitution):")
    print(A_substituted)
    print("✅ Input matrix B (after substitution):")
    print(B_substituted)


    return A_substituted, B_substituted, state_vars




def simulate_circuit(A, B, C, t_span, initial_conditions, input_function):
    """
    Numerically solves the ODE system dx/dt = Ax + Bu using solve_ivp.

    Parameters:
    - A: State matrix (numpy array after substitution).
    - B: Input matrix (numpy array after substitution).
    - t_span: Tuple (t_start, t_end) defining the time range.
    - initial_conditions: Initial state vector (same size as state variables).
    - input_function: Function u(t) defining the input voltage/current.

    Returns:
    - t: Time points from simulation.
    - x: State variable trajectories over time.
    """

    # Convert A and B to numerical arrays (ensure float type)
    A_func = np.array(A).astype(float)
    B_func = np.array(B).astype(float)
    C_func = np.array(C).astype(float)

    # Define time points at fixed 0.1s intervals
    t_eval = np.arange(t_span[0], t_span[1], 0.1)  # Time points at 0.1s resolution

    # Define ODE system
    def state_space_ode(t, x):
        return A_func @ x + B_func @ np.array([input_function(t)])  # dx/dt = Ax + Bu

    # Solve ODE system
    sol = solve_ivp(state_space_ode, t_span, initial_conditions, method="RK45", t_eval=t_eval)

    # Compute output y = Cx
    y = C_func @ sol.y

    return sol.t, sol.y, y