import json
from amps_simulation.core.parser import ParserJson_todict
from amps_simulation.core.simulation import Simulation


def test_simulation_class():
    # Load a test circuit
    with open('test_data/test_lcr.json', 'r') as file:
        circuit_json = json.load(file)
    
    # Create a parser instance
    parser = ParserJson_todict()
    
    # Parse the circuit data
    electrical_nodes, circuit_components = parser.parse(circuit_json)
    
    # Create a simulation instance
    simulation = Simulation(electrical_nodes, circuit_components)
    
    # Test initialize method
    simulation.initialize()
    
    # Verify that the simulation instance has the correct attributes
    assert simulation.voltage_vars is not None
    assert simulation.current_vars is not None
    assert simulation.state_vars is not None
    assert simulation.state_derivatives is not None
    assert simulation.input_vars is not None
    assert simulation.ground_node is not None
    
    # Test simulate method
    t, x, y = simulation.simulate()
    
    # Verify simulation results
    assert t is not None
    assert x is not None
    assert y is not None
    assert len(t) > 0
    assert x.shape[1] == len(t)  # x has shape (n_states, n_timepoints)
    assert y.shape[1] == len(t)  # y has shape (n_outputs, n_timepoints)
    
    # Test method chaining
    simulation2 = Simulation(electrical_nodes, circuit_components)
    t2, x2, y2 = simulation2.initialize().simulate()
    
    # Verify chained method results
    assert t2 is not None
    assert x2 is not None
    assert y2 is not None
    assert len(t2) > 0
    assert x2.shape[1] == len(t2)  # x2 has shape (n_states, n_timepoints)
    assert y2.shape[1] == len(t2)  # y2 has shape (n_outputs, n_timepoints)
    
    print("All tests passed!")

if __name__ == "__main__":
    test_simulation_class() 