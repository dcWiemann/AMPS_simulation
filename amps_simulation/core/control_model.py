from typing import Optional, List, Callable
import numpy as np
from .control_graph import ControlGraph
from .control_signal import ControlSignal
from .control_port import ControlPort


class ControlModel:
    """
    Control model managing control signals, ports, and their connections.

    Similar to ElectricalModel, this class provides a high-level API for
    building and managing the control layer of a simulation.
    """

    def __init__(self, control_graph: Optional[ControlGraph] = None):
        """
        Initialize the ControlModel.

        Args:
            control_graph: ControlGraph instance. If None, creates empty graph.
        """
        self.control_graph = control_graph if control_graph is not None else ControlGraph()

        # Compilation state
        self._input_function = None
        self._port_order = []
        self._compiled = False
        self.initialized = False

    def initialize(self) -> None:
        """Initialize the control model (placeholder for future extensions)"""
        self.initialized = True

    # Delegation to ControlGraph for building
    def add_signal(self, signal: ControlSignal):
        """Add control signal to the model"""
        self.control_graph.add_signal(signal)

    def add_port(self, port: ControlPort):
        """Add control port to the model"""
        self.control_graph.add_port(port)

    def connect_signal_to_port(self, signal_id: str, port_name: str, gain: float = 1.0):
        """Connect signal to control port"""
        self.control_graph.connect_signal_to_port(signal_id, port_name, gain)

    # Compilation API
    def compile_input_function(self, port_order: List[str]) -> Callable[[float], np.ndarray]:
        """Build optimized u(t) function using ControlPort static method

        Args:
            port_order: Ordered list of SOURCE control port names

        Returns:
            Callable that takes time t and returns input vector u(t)
        """
        self._port_order = port_order.copy()
        self._input_function = ControlPort.compile_input_function(
            port_order,
            self.control_graph.ports,
            self.control_graph.connections,
            self.control_graph.signals
        )
        self._compiled = True
        return self._input_function

    def get_input_vector(self, t: float) -> np.ndarray:
        """Fast runtime call - must call compile_input_function first"""
        if not self._compiled:
            raise RuntimeError("Must call compile_input_function() first")
        return self._input_function(t)

    def get_port_order(self) -> List[str]:
        """Get the current port order used for input vector"""
        return self._port_order.copy()

    # Properties for convenience access
    @property
    def signals(self):
        return self.control_graph.signals

    @property
    def ports(self):
        return self.control_graph.ports

    @property
    def connections(self):
        return self.control_graph.connections

    def __repr__(self):
        status = "compiled" if self._compiled else "not compiled"
        return f"ControlModel({status}, ports={len(self._port_order)})"
