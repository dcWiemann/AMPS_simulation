from typing import Union, Callable, Dict, List, Optional
import numpy as np
import sympy as sp
import networkx as nx
from .control_port import ControlPort


class ControlSignal:
    """Represents a time-dependent control signal u_i(t)"""
    
    def __init__(self, signal_id: str, expression: Union[str, float, int, Callable]):
        self.signal_id = signal_id
        self.expression = expression
        self._compiled_func = self._compile_expression(expression)
        
    def _compile_expression(self, expr) -> Callable[[float], float]:
        """Compile expression for fast evaluation"""
        if callable(expr):
            return expr
        elif isinstance(expr, (int, float)):
            return lambda t: float(expr)  # Constant signal
        elif isinstance(expr, str):
            # Parse string like "sin(t)", "5*sin(2*pi*t)", "sawtooth(t)"
            try:
                t = sp.Symbol('t')
                # Add common functions to namespace for parsing
                namespace = {
                    'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
                    'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
                    'pi': sp.pi, 'e': sp.E,
                    't': t
                }
                sympy_expr = sp.sympify(expr, locals=namespace)
                return sp.lambdify(t, sympy_expr, 'numpy')
            except Exception as e:
                raise ValueError(f"Cannot parse expression '{expr}': {e}")
        else:
            raise ValueError(f"Unsupported expression type: {type(expr)}")
        
    def evaluate(self, t: float) -> float:
        """Fast evaluation at time t"""
        return float(self._compiled_func(t))
    
    def __repr__(self):
        return f"ControlSignal(id='{self.signal_id}', expr='{self.expression}')"


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
        
    def get_unconnected_ports(self) -> List[str]:
        """Return list of ports with no incoming connections"""
        return [port_name for port_name in self.ports.keys() 
                if port_name not in self.connections]
    
    def get_source_ports(self) -> Dict[str, ControlPort]:
        """Return only control ports that belong to sources"""
        return {name: port for name, port in self.ports.items() 
                if port.port_type == "source"}
    
    def get_switch_ports(self) -> Dict[str, ControlPort]:
        """Return only control ports that belong to switches"""
        return {name: port for name, port in self.ports.items() 
                if port.port_type == "switch"}
    
    def __repr__(self):
        return f"ControlGraph(signals={len(self.signals)}, ports={len(self.ports)}, connections={len(self.connections)})"


class ControlOrchestrator:
    """Main control orchestrator for managing control signals and ports"""
    
    def __init__(self, control_graph: ControlGraph):
        self.control_graph = control_graph
        self._input_function = None
        self._port_order = []  # Ordered list of port names for vector creation
        self._compiled = False
        
    def compile_input_function(self, port_order: List[str]) -> Callable[[float], np.ndarray]:
        """Build optimized u(t) function returning input vector for sources only
        
        Args:
            port_order: Ordered list of SOURCE control port names matching input variables order
            
        Returns:
            Callable that takes time t and returns input vector u(t)
        """
        self._port_order = port_order.copy()
        
        # Validate all ports exist and are source ports
        missing_ports = [p for p in port_order if p not in self.control_graph.ports]
        if missing_ports:
            raise ValueError(f"Ports not found in control graph: {missing_ports}")
            
        non_source_ports = [p for p in port_order 
                           if self.control_graph.ports[p].port_type != "source"]
        if non_source_ports:
            raise ValueError(f"Only source ports allowed in input function: {non_source_ports}")
        
        def input_function(t: float) -> np.ndarray:
            u = np.zeros(len(port_order))
            for i, port_name in enumerate(port_order):
                if port_name in self.control_graph.connections:
                    signal_id, gain = self.control_graph.connections[port_name]
                    signal = self.control_graph.signals[signal_id]
                    u[i] = gain * signal.evaluate(t)
                # If no connection, u[i] remains 0 (default)
            return u
            
        self._input_function = input_function
        self._compiled = True
        return input_function
        
    def get_input_vector(self, t: float) -> np.ndarray:
        """Fast runtime call for solver - must call compile_input_function first"""
        if not self._compiled:
            raise RuntimeError("Must call compile_input_function() before get_input_vector()")
        return self._input_function(t)
    
    def get_port_order(self) -> List[str]:
        """Get the current port order used for input vector"""
        return self._port_order.copy()
    
    def __repr__(self):
        status = "compiled" if self._compiled else "not compiled"
        return f"ControlOrchestrator({status}, ports={len(self._port_order)})"