from .core import ElectricalModel, CircuitParser, ParserJson, Simulation
from .run_simulation import run_simulation

__version__ = "0.1.0"

# Export the main classes and functions that users will need
__all__ = [
    'ElectricalModel',
    'CircuitParser',
    'ParserJson',
    'Simulation',
    'run_simulation'
]