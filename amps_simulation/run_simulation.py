# Main function to run circuit simulation using the refactored Engine structure
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import os
import sys

# Add the parent directory to the path so we can import from amps_simulation
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amps_simulation.core.parser import ParserJson
from amps_simulation.core.engine import Engine
from amps_simulation.core.electrical_model import ElectricalModel


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("amps.log", mode='w'),
        logging.StreamHandler()
    ]
)

# Suppress noisy loggers from other libraries
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.ERROR)


def run_simulation_from_file(file_path, t_span=(0, 1.0), method='RK45', plot_results=True, **kwargs):
    """
    Run circuit simulation from JSON file using the modern Engine architecture.
    
    Args:
        file_path: Path to the JSON circuit file
        t_span: Tuple (t_start, t_end) for simulation time span
        method: Integration method for solve_ivp ('RK45', 'DOP853', etc.)
        plot_results: Whether to plot the simulation results
        **kwargs: Additional arguments passed to solve_ivp
        
    Returns:
        Dictionary with simulation results including 't', 'y', 'out', and metadata
    """
    with open(file_path, 'r') as file:
        circuit_json = json.load(file)
    return run_simulation(circuit_json, t_span, method, plot_results, **kwargs)


def run_simulation(circuit_json_data, t_span=(0, 1.0), method='RK45', plot_results=True, **kwargs):
    """
    Run circuit simulation using the modern Engine architecture.
    
    Args:
        circuit_json_data: JSON circuit data
        t_span: Tuple (t_start, t_end) for simulation time span  
        method: Integration method for solve_ivp ('RK45', 'DOP853', etc.)
        plot_results: Whether to plot the simulation results
        **kwargs: Additional arguments passed to solve_ivp
        
    Returns:
        Dictionary with simulation results including 't', 'y', 'out', and metadata
    """
    # Parse the circuit using the modern NetworkX parser
    parser = ParserJson()
    graph, control_graph = parser.parse(circuit_json_data)

    # Create electrical model and initialize the Engine
    electrical_model = ElectricalModel(graph)
    engine = Engine(electrical_model, control_graph)
    engine.initialize()
    
    # Run the simulation using the new run_simulation method
    result = engine.run_simulation(t_span=t_span, method=method, **kwargs)
    
    # Plot the results if requested and outputs exist
    if plot_results and result['success'] and result.get('out') is not None:
        _plot_simulation_results(result, engine)
    
    return result


def _plot_simulation_results(result, engine):
    """
    Plot simulation results for state variables and outputs.
    
    Args:
        result: Simulation result dictionary from Engine.run_simulation()
        engine: Engine instance containing variable information
    """
    t = result['t']
    y = result['y']
    outputs = result['out']
    
    # Determine subplot layout
    n_states = len(engine.state_vars)
    n_outputs = len(engine.output_vars) if outputs is not None else 0
    total_plots = n_states + n_outputs
    
    if total_plots == 0:
        print("No variables to plot")
        return
        
    # Create subplots
    fig, axes = plt.subplots(total_plots, 1, figsize=(10, 2*total_plots), squeeze=False)
    axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot state variables
    for i, state_var in enumerate(engine.state_vars):
        axes[plot_idx].plot(t, y[i], 'b-', linewidth=2, label=f'State: {state_var}')
        axes[plot_idx].set_ylabel('Value')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].legend()
        plot_idx += 1
    
    # Plot output variables
    if outputs is not None:
        for i, output_var in enumerate(engine.output_vars):
            axes[plot_idx].plot(t, outputs[i], 'r-', linewidth=2, label=f'Output: {output_var}')
            axes[plot_idx].set_ylabel('Value')
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].legend()
            plot_idx += 1
    
    # Set x-label on the bottom plot
    axes[-1].set_xlabel('Time (s)')
    
    plt.suptitle('Circuit Simulation Results', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage with the modern Engine architecture
    run_simulation_from_file('test_data/test_rlc.json', t_span=(0, 2.0))
