# main function to run simulation with test_data/test_rrc_gnd.json
import numpy as np
from amps_simulation.core.state_space_model import extract_differential_equations, simulate_circuit
from amps_simulation.core.utils import plot_results
import logging
import json


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("amps.log", mode='w'),
        logging.StreamHandler()
    ]
)

# Suppress noisy loggers from other libraries
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.ERROR)


def run_simulation_from_file(file_path):
    with open(file_path, 'r') as file:
        circuit_json = json.load(file)
    return run_simulation(circuit_json)


def run_simulation(circuit_json_data):

    print(circuit_json_data)
    # Extract state space matrices
    A, B, state_vars, input_vars = extract_differential_equations(circuit_json_data)

    # Define identity output matrix C (observing all state variables)
    C = np.eye(A.shape[0])  # Identity matrix of size (states x states)


    # Define simulation parameters
    t_span = (0, 10)  # Simulate from 0 to 10 seconds
    initial_conditions = np.zeros(A.shape[0])  # Zero initial state


    ### TODO actual inputs
    # Define step input function (5V at t >= 1s)
    def step_input_function(t, n = len(input_vars.keys())):
        return np.array([5.0 if t >= (i + 1) else 0.0 for i in range(n)])


    # Solve the ODE system
    t, x, y = simulate_circuit(A, B, C, t_span, initial_conditions, step_input_function)
    logging.info("✅ Time points: %s", t[0:10])  # Log first 10 time points
    logging.info("✅ Simulation completed.")
    # Plot the results
    plot_results(t, x, state_vars)



if __name__ == "__main__":
    run_simulation_from_file('test_data/test_lcr.json')