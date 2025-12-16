from typing import Optional, List, Callable, Union, Tuple, Sequence
import numpy as np
import networkx as nx
from sympy import Symbol
from .control_block import ControlBlock
from .control_graph import ControlGraph
from .control_signal import ControlSignal
from .control_port import ControlPort


class ControlModel:
    """
    Control model managing control blocks/signals and legacy port wiring.

    Similar to ElectricalModel, this class provides a high-level API for building
    and managing the control layer of a simulation.
    """

    def __init__(self, graph: Optional[nx.MultiDiGraph] = None, control_graph: Optional[ControlGraph] = None):
        """
        Initialize the ControlModel.

        Args:
            graph: NetworkX MultiDiGraph containing ControlBlocks/ControlSignals.
            control_graph: Legacy ControlGraph instance for port-level wiring.
        """
        # Support legacy positional usage ControlModel(control_graph)
        if isinstance(graph, ControlGraph) and control_graph is None:
            control_graph = graph
            graph = None

        self.graph = graph if graph is not None else nx.MultiDiGraph()
        self.control_graph = control_graph if control_graph is not None else ControlGraph()
        self.block_list: List[ControlBlock] = []
        self.state_vars: List[Symbol] = []

        # Compilation state
        self._input_function = None
        self._port_order = []
        self._compiled = False
        self.initialized = False

    def initialize(self) -> None:
        """Initialize the control model (placeholder for future extensions)."""
        self.initialized = True

    # New graph-building API
    def add_block(self, blocks: Union[ControlBlock, Sequence[ControlBlock]]) -> None:
        """Add one or more ControlBlocks to the graph."""
        if isinstance(blocks, ControlBlock):
            blocks = [blocks]

        for block in blocks:
            if self.graph.has_node(block.name):
                raise ValueError(f"Control block '{block.name}' already exists in graph")
            self.graph.add_node(block.name, block=block)
            self.block_list.append(block)

    @staticmethod
    def _port_name_and_index(block: ControlBlock, port: Union[str, int], *, port_kind: str) -> Tuple[str, int]:
        ports = block.outport_names if port_kind == "out" else block.inport_names
        if isinstance(port, int):
            if port < 0 or port >= len(ports):
                raise IndexError(f"{port_kind}port index {port} out of range for block '{block.name}'")
            return ports[port], port
        if port in ports:
            return port, ports.index(port)
        raise ValueError(f"{port_kind}port '{port}' not found on block '{block.name}'")

    def connect(self, from_block: ControlBlock, from_port: Union[str, int],
                to_block: ControlBlock, to_port: Union[str, int],
                signal: Optional[ControlSignal] = None) -> ControlSignal:
        """
        Connect two control blocks via a ControlSignal edge.

        Args:
            from_block: Source block instance
            from_port: Source port name or index
            to_block: Destination block instance
            to_port: Destination port name or index
            signal: Optional pre-constructed ControlSignal to attach

        Returns:
            ControlSignal stored on the created edge
        """
        src_port_name, src_port_idx = self._port_name_and_index(from_block, from_port, port_kind="out")
        dst_port_name, dst_port_idx = self._port_name_and_index(to_block, to_port, port_kind="in")

        signal_name = signal.name if signal else f"{from_block.name}__{src_port_name}__{to_block.name}__{dst_port_name}"
        if signal is None:
                signal = ControlSignal(
                    signal_name,
                    src_block_name=from_block.name,
                    dst_block_name=to_block.name,
                    src_port_name=src_port_name,
                    dst_port_name=dst_port_name,
                    src_port_idx=src_port_idx,
                    dst_port_idx=dst_port_idx,
                    dtype=getattr(from_block, "outport_dtype", None),
                    shape=getattr(from_block, "outport_shape", None),
                )
        else:
            # Fill in structural metadata if missing
            signal.name = signal_name
            signal.signal_id = signal_name
            signal.src_block_name = signal.src_block_name or from_block.name
            signal.dst_block_name = signal.dst_block_name or to_block.name
            signal.src_port_name = signal.src_port_name or src_port_name
            signal.dst_port_name = signal.dst_port_name or dst_port_name
            signal.src_port_idx = signal.src_port_idx if signal.src_port_idx is not None else src_port_idx
            signal.dst_port_idx = signal.dst_port_idx if signal.dst_port_idx is not None else dst_port_idx
            signal.dtype = signal.dtype if signal.dtype is not None else getattr(from_block, "outport_dtype", None)
            signal.shape = signal.shape if signal.shape is not None else getattr(from_block, "outport_shape", None)

        self.graph.add_edge(from_block.name, to_block.name, key=signal.name, signal=signal)
        return signal

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
