from abc import ABC
from typing import Optional
from pydantic import BaseModel, Field

class Component(BaseModel, ABC):
    """Abstract base class for all circuit components."""
    comp_id: str = Field(..., description="Unique identifier for the component")
    
    class Config:
        frozen = True  # Make components immutable

class Resistor(Component):
    """Resistor component."""
    resistance: float = Field(..., description="Resistance value in ohms", ge=0)

class Capacitor(Component):
    """Capacitor component."""
    capacitance: float = Field(..., description="Capacitance value in farads", ge=0)

class Inductor(Component):
    """Inductor component."""
    inductance: float = Field(..., description="Inductance value in henries", ge=0)
    I_max: Optional[float] = Field(None, description="Maximum current rating in amperes")

class PowerSwitch(Component):
    """Power switch component."""
    pass

class Diode(Component):
    """Diode component."""
    pass

class VoltageSource(Component):
    """Voltage source component."""
    voltage: float = Field(..., description="Voltage value in volts")

class CurrentSource(Component):
    """Current source component."""
    current: float = Field(..., description="Current value in amperes")

class Ground(Component):
    """Ground component."""
    pass