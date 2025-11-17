from typing import Dict
import networkx as nx
from .control_signal import ControlSignal
from .control_port import ControlPort


class ControlGraph:
    """NetworkX DiGraph for control layer structure"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.signals = {}  # signal_id -> ControlSignal
        self.ports = {}    # port_name -> ControlPort
        self.connections = {}  # port_name -> (signal_id, gain)

    def add_signal(self, signal: ControlSignal):
        """Add control signal as node"""
        self.graph.add_node(signal.signal_id, type='signal', signal=signal)
        self.signals[signal.signal_id] = signal

    def add_port(self, port: ControlPort):
        """Add control port as node"""
        self.graph.add_node(port.name, type='port', port=port)
        self.ports[port.name] = port

    def connect_signal_to_port(self, signal_id: str, port_name: str, gain: float = 1.0):
        """Connect signal to control port"""
        if signal_id not in self.signals:
            raise ValueError(f"Signal '{signal_id}' not found in control graph")
        if port_name not in self.ports:
            raise ValueError(f"Port '{port_name}' not found in control graph")

        self.graph.add_edge(signal_id, port_name, gain=gain)
        self.connections[port_name] = (signal_id, gain)

    def __repr__(self):
        return f"ControlGraph(signals={len(self.signals)}, ports={len(self.ports)}, connections={len(self.connections)})"
