import json
import logging
from amps_simulation.core.parser import ParserJson
from amps_simulation.core.simulation import Simulation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("test_simulation.log", mode='w'),
        logging.StreamHandler()
    ]
)

def test_simulation_class():
    # Load a test circuit
    with open('test_data/test_lcr.json', 'r') as file:
        circuit_json = json.load(file)
    
    # Create a parser instance
    parser = ParserJson()
    
    # Parse the circuit data
    electrical_nodes, circuit_components = parser.parse(circuit_json)
    
    # Create a simulation instance
    simulation = Simulation(electrical_nodes, circuit_components)
    
    # Assign variables
    voltage_vars, current_vars, state_vars, state_derivatives, input_vars, ground_node = simulation.assign_variables()
    
    # Print the results
    print("Voltage variables:", voltage_vars)
    print("Current variables:", current_vars)
    print("State variables:", state_vars)
    print("State derivatives:", state_derivatives)
    print("Input variables:", input_vars)
    print("Ground node:", ground_node)
    
    # Verify that the simulation instance has the correct attributes
    assert simulation.voltage_vars == voltage_vars
    assert simulation.current_vars == current_vars
    assert simulation.state_vars == state_vars
    assert simulation.state_derivatives == state_derivatives
    assert simulation.input_vars == input_vars
    assert simulation.ground_node == ground_node
    
    print("All tests passed!")

if __name__ == "__main__":
    test_simulation_class() 