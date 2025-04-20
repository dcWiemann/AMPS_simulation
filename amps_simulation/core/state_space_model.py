# import parsing and equations functions
from amps_simulation.core.parsing import build_electrical_nodes, build_circuit_components
from amps_simulation.core.simulation import Simulation
from scipy.integrate import solve_ivp
import numpy as np
import logging



def extract_differential_equations(circuit_json):
    # Step 1: Parse JSON
    components = circuit_json["nodes"]
    connections = circuit_json["edges"]

    # Step 2: Identify electrical nodes
    electrical_nodes = build_electrical_nodes(components, connections)
    logging.info("✅ Electrical nodes: %s", electrical_nodes)

    circuit_components = build_circuit_components(components, electrical_nodes)
    logging.info("✅ Circuit components: %s", circuit_components)

    # Step 3: Create Simulation instance and assign variables
    simulation = Simulation(electrical_nodes, circuit_components)
    simulation.assign_variables()
    
    # Step 4: Extract differential equations
    A, B, state_vars, input_vars = simulation.extract_differential_equations(components)
    
    return A, B, state_vars, input_vars




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
        u = input_function(t)
        return (A_func @ x) + (B_func @ u)  # dx/dt = Ax + Bu


    # Solve ODE system
    # sol = solve_ivp(state_space_ode, t_span, initial_conditions, method="RK45", t_eval=t_eval) # Use t_eval for fixed intervals
    sol = solve_ivp(state_space_ode, t_span, initial_conditions, method="RK45")

    # Compute output y = Cx
    y = C_func @ sol.y

    return sol.t, sol.y, y