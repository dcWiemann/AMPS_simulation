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
    
    # Create a simulation instance
    simulation = Simulation(electrical_nodes, circuit_components)
    
    # Initialize the simulation (assign variables)
    simulation.initialize()
    
    # Run the simulation
    t, x, y = simulation.simulate()
    
    # Plot the results only if not in test mode
    if not test_mode:
        plot_results(t, x, simulation.state_vars)
    
    return t, x, y


if __name__ == "__main__":
    run_simulation_from_file('test_data/test_lcr.json')
