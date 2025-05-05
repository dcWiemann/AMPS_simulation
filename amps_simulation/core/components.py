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
    V_max: Optional[float] = Field(None, description="Maximum voltage rating in volts")

class Inductor(Component):
    """Inductor component."""
    inductance: float = Field(..., description="Inductance value in henries", ge=0)
    I_max: Optional[float] = Field(None, description="Maximum current rating in amperes")

class PowerSwitch(Component):
    """Power switch component."""
    R_on: Optional[float] = Field(..., description="On-state resistance in ohms", ge=0)
    R_off: Optional[float] = Field(..., description="Off-state resistance in ohms", ge=0)
    V_max: Optional[float] = Field(..., description="Maximum blocking voltage rating in volts", gt=0)
    I_max: Optional[float] = Field(..., description="Maximum current rating in amperes", gt=0)

class Diode(Component):
    """Diode component."""
    V_th: Optional[float] = Field(..., description="Forward voltage drop in volts", ge=0)
    V_rb: Optional[float] = Field(..., description="Reverse breakdown voltage in volts", lt=0)

class VoltageSource(Component):
    """Voltage source component."""
    voltage: float = Field(..., description="Voltage value in volts")
    R_s: Optional[float] = Field(0, description="Series resistance in ohms", ge=0)

class CurrentSource(Component):
    """Current source component."""
    current: float = Field(..., description="Current value in amperes")
    R_p: Optional[float] = Field(float('inf'), description="Parallel resistance in ohms", gt=0) 