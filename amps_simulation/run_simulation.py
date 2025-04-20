# main function to run simulation with test_data/test_rrc_gnd.json
import numpy as np
from amps_simulation.core.utils import plot_results
from amps_simulation.core.parser import ParserJson
from amps_simulation.core.simulation import Simulation
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


def run_simulation_from_file(file_path, test_mode=False):
    with open(file_path, 'r') as file:
        circuit_json = json.load(file)
    return run_simulation(circuit_json, test_mode)


def run_simulation(circuit_json_data, test_mode=False):
    # Create a parser instance
    parser = ParserJson()
    
    # Parse the circuit data
    electrical_nodes, circuit_components = parser.parse(circuit_json_data)
    
    # Create a simulation instance and assign variables
    simulation = Simulation(electrical_nodes, circuit_components)
    voltage_vars, current_vars, state_vars, state_derivatives, input_vars, ground_node = simulation.assign_variables()
    
    # Extract state space matrices with numerical values
    A, B, state_vars, input_vars = simulation.extract_differential_equations(circuit_json_data["nodes"])
    
    # Define identity output matrix C (observing all state variables)
    C = np.eye(A.shape[0])  # Identity matrix of size (states x states)
    
    # Define simulation parameters
    t_span = (0, 10)  # Simulate from 0 to 10 seconds
    initial_conditions = np.zeros(A.shape[0])  # Zero initial state
    
    # Create input function using the simulation's create_input_function method
    input_function = simulation.create_input_function()
    
    # Solve the ODE system using the simulation's simulate_circuit method
    t, x, y = simulation.simulate_circuit(A, B, C, t_span, initial_conditions, input_function)
    logging.info("✅ Time points: %s", t[0:10])  # Log first 10 time points
    logging.info("✅ Simulation completed.")
    
    # Plot the results only if not in test mode
    if not test_mode:
        plot_results(t, x, state_vars)
    
    return t, x, y


if __name__ == "__main__":
    run_simulation_from_file('test_data/test_lcr.json')
