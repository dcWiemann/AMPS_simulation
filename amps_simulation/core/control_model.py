from typing import Optional, List, Callable, Union, Tuple, Sequence, Any, Dict
import numpy as np
import networkx as nx
from sympy import Symbol
from .control_block import ControlBlock, ControlPort, InPort, OutPort, LinearControlBlock
from .control_signal import ControlSignal


class ControlModel:
    """
    Control model managing control blocks/signals as a NetworkX MultiDiGraph.

    Similar to ElectricalModel, this class provides a high-level API for building
    and managing the control layer of a simulation.
    """

    def __init__(self, graph: Optional[nx.MultiDiGraph] = None):
        """
        Initialize the ControlModel.

        Args:
            graph: NetworkX MultiDiGraph containing ControlBlocks/ControlSignals.
        """
        self.graph = graph if graph is not None else nx.MultiDiGraph()
        self.block_list: List[ControlBlock] = []
        self.state_vars: List[Symbol] = []

        # Convenience lists (populated in initialize())
        self.list_all_blocks: List[ControlBlock] = []
        self.list_linear_blocks: List[LinearControlBlock] = []
        self.list_inports: List[InPort] = []
        self.list_outports: List[OutPort] = []

        # Compilation state
        self._input_function: Optional[Callable[[float], np.ndarray]] = None
        self._port_order: List[str] = []
        self._compiled = False
        self.initialized = False

    def initialize(self) -> None:
        """Initialize the control model and populate convenience block lists."""
        self.list_all_blocks = []
        self.list_linear_blocks = []
        self.list_inports = []
        self.list_outports = []

        for _, node_data in self.graph.nodes(data=True):
            block = node_data.get("block")
            if not isinstance(block, ControlBlock):
                continue

            self.list_all_blocks.append(block)
            if isinstance(block, LinearControlBlock):
                self.list_linear_blocks.append(block)
            if isinstance(block, InPort):
                self.list_inports.append(block)
            if isinstance(block, OutPort):
                self.list_outports.append(block)

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

    def get_block(self, name: str) -> ControlBlock:
        """Get a block by node name."""
        if not self.graph.has_node(name):
            raise KeyError(f"Control block node '{name}' not found")
        block = self.graph.nodes[name].get("block")
        if not isinstance(block, ControlBlock):
            raise TypeError(f"Control graph node '{name}' does not contain a ControlBlock")
        return block

    def port_blocks(self, *, port_type: Optional[str] = None) -> Dict[str, ControlBlock]:
        """
        Return a mapping of node_name -> block for nodes that are control ports.

        Ports are represented by instances of ControlPort (including InPort/OutPort).
        If `port_type` is provided, it filters by port class:
          - "input"  -> InPort
          - "output" -> OutPort
        """
        result: Dict[str, ControlBlock] = {}
        for node_name, node_data in self.graph.nodes(data=True):
            block = node_data.get("block")
            if not isinstance(block, ControlBlock):
                continue
            if not isinstance(block, ControlPort):
                continue
            if port_type is not None:
                if port_type == "input" and not isinstance(block, InPort):
                    continue
                if port_type == "output" and not isinstance(block, OutPort):
                    continue
                if port_type not in {"input", "output"}:
                    raise ValueError(f"Unsupported port_type '{port_type}' (supported: 'input', 'output')")
            result[str(node_name)] = block
        return result

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

        self.graph.add_edge(from_block.name, to_block.name, key=signal.name, signal=signal)
        return signal

    # Compilation API
    def compile_input_function(self, port_order: List[str]) -> Callable[[float], np.ndarray]:
        """
        Build optimized u(t) function based on incoming ControlSignal edges.

        Args:
            port_order: Ordered list of SOURCE port node names

        Returns:
            Callable that takes time t and returns input vector u(t)
        """
        self._port_order = list(port_order)

        # Pre-resolve the driving signal for each port for fast runtime evaluation.
        drivers: List[Optional[ControlSignal]] = []
        for port_name in self._port_order:
            if not self.graph.has_node(port_name):
                raise ValueError(f"Port node '{port_name}' not found in control graph")

            in_edges = [
                (u, v, k, d)
                for u, v, k, d in self.graph.in_edges(port_name, keys=True, data=True)
                if isinstance(d.get("signal"), ControlSignal)
            ]

            if len(in_edges) == 0:
                drivers.append(None)
                continue

            if len(in_edges) > 1:
                raise ValueError(f"Port node '{port_name}' has multiple driving signals; expected exactly 1")

            _, _, _, data = in_edges[0]
            drivers.append(data["signal"])

        def input_function(t: float) -> np.ndarray:
            u = np.zeros(len(self._port_order), dtype=float)
            for i, signal in enumerate(drivers):
                if signal is None:
                    continue
                u[i] = signal.evaluate(t)
            return u

        self._input_function = input_function
        self._compiled = True
        return self._input_function

    def get_input_vector(self, t: float) -> np.ndarray:
        """Fast runtime call - must call compile_input_function first"""
        if not self._compiled:
            raise RuntimeError("Must call compile_input_function() first")
        assert self._input_function is not None
        return self._input_function(t)

    def get_port_order(self) -> List[str]:
        """Get the current port order used for input vector"""
        return self._port_order.copy()

    def __repr__(self):
        status = "compiled" if self._compiled else "not compiled"
        return f"ControlModel({status}, ports={len(self._port_order)})"
