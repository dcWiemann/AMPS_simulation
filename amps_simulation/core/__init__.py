# amps_simulation/core/__init__.py

# Import from equations.py
from .equations import (
    extract_input_and_state_vars,
    write_kcl_equations,
    write_kvl_equations,
    find_loops,
    solve_helper_variables,
    solve_state_derivatives,
    extract_state_space_matrices,
    substitute_component_values
)

# Import from parsing.py
from .parsing import (
    build_electrical_nodes,
    build_circuit_components,
    assign_voltage_variables,
    assign_current_variables
)

# Import from state_space_model.py
from .state_space_model import (
    extract_differential_equations,
    simulate_circuit
)

# Import from utils.py

# Define what should be available when someone imports from amps_simulation.core
__all__ = [
    # From equations.py
    'extract_input_and_state_vars',
    'write_kcl_equations',
    'write_kvl_equations',
    'find_loops',
    'solve_helper_variables',
    'solve_state_derivatives',
    'extract_state_space_matrices',
    'substitute_component_values',
    
    # From parsing.py
    'build_electrical_nodes',
    'build_circuit_components',
    'assign_voltage_variables',
    'assign_current_variables',
    
    # From state_space_model.py
    'extract_differential_equations',
    'simulate_circuit',
]