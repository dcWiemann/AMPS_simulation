# main function to run simulation with test_data/test_rrc_gnd.json
import json
import os
import numpy as np
from circuit_sim.state_space_model import extract_differential_equations, simulate_circuit
from circuit_sim.utils import plot_results
import logging


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


def main():
    # Load the JSON file
    json_path = os.path.join(os.path.dirname(__file__), 'test_data', 'test_rcl.json')
    with open(json_path, 'r') as file:
        circuit_json = json.load(file)

    # Extract state space matrices
    A, B, state_vars = extract_differential_equations(circuit_json)

    # Define identity output matrix C (observing all state variables)
    C = np.eye(A.shape[0])  # Identity matrix of size (states x states)


    # Define simulation parameters
    t_span = (0, 10)  # Simulate from 0 to 10 seconds
    initial_conditions = np.zeros(A.shape[0])  # Zero initial state

    # Define step input function (5V at t >= 1s)
    def step_input_function(t):
        return 5.0 if t >= 1 else 0.0  # Single input, no list needed

    # Solve the ODE system
    t, x, y = simulate_circuit(A, B, C, t_span, initial_conditions, step_input_function)
    # Plot the results
    plot_results(t, x, state_vars)



if __name__ == "__main__":
    main()