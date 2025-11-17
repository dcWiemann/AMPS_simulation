from typing import ClassVar, Dict, Any, Literal, List, Callable
from pydantic import BaseModel, Field, field_validator
import numpy as np


class ControlPort(BaseModel):
    """
    Represents a control port that exposes a controlled value (e.g., voltage or current).
    Each control port has a unique name.
    """
    name: str = Field(..., description="Unique name for the control port")
    variable: Any = Field(..., description="The controlled variable (e.g., voltage, current)")
    port_type: Literal["source", "switch", "other"] = Field(..., description="Type of component this port belongs to")

    _registry: ClassVar[Dict[str, 'ControlPort']] = {}

    @field_validator('name')
    @classmethod
    def validate_unique_name(cls, v: str) -> str:
        if v in cls._registry:
            raise ValueError(f"ControlPort name '{v}' is already in use")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        self._registry[self.name] = self

    @classmethod
    def get_control_port(cls, name: str) -> 'ControlPort':
        return cls._registry.get(name)

    @classmethod
    def clear_registry(cls) -> None:
        cls._registry.clear()

    @staticmethod
    def compile_input_function(port_order: List[str], ports_dict: Dict[str, 'ControlPort'],
                               connections: Dict[str, tuple], signals_dict: Dict) -> Callable[[float], np.ndarray]:
        """Build optimized u(t) function returning input vector for sources only

        Args:
            port_order: Ordered list of SOURCE control port names matching input variables order
            ports_dict: Dictionary of port_name -> ControlPort
            connections: Dictionary of port_name -> (signal_id, gain)
            signals_dict: Dictionary of signal_id -> ControlSignal

        Returns:
            Callable that takes time t and returns input vector u(t)
        """
        # Validate all ports exist and are source ports
        missing_ports = [p for p in port_order if p not in ports_dict]
        if missing_ports:
            raise ValueError(f"Ports not found: {missing_ports}")

        non_source_ports = [p for p in port_order if ports_dict[p].port_type != "source"]
        if non_source_ports:
            raise ValueError(f"Only source ports allowed: {non_source_ports}")

        def input_function(t: float) -> np.ndarray:
            u = np.zeros(len(port_order))
            for i, port_name in enumerate(port_order):
                if port_name in connections:
                    signal_id, gain = connections[port_name]
                    signal = signals_dict[signal_id]
                    u[i] = gain * signal.evaluate(t)
            return u

        return input_function
