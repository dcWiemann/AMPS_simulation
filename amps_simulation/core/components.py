from abc import ABC
from typing import Optional, ClassVar, Dict
from pydantic import BaseModel, Field, computed_field, field_validator
from pydantic import ConfigDict

class Component(BaseModel, ABC):
    """Abstract base class for all circuit components."""
    comp_id: str = Field(..., description="Unique identifier for the component", pattern=r"^[A-Za-z].*")
    _registry: ClassVar[Dict[str, 'Component']] = {}
    
    @field_validator('comp_id')
    @classmethod
    def validate_unique_comp_id(cls, v: str) -> str:
        """Validate that the component ID is unique."""
        if v in cls._registry:
            raise ValueError(f"Component ID '{v}' is already in use")
        return v
    
    def __init__(self, **data):
        super().__init__(**data)
        self._registry[self.comp_id] = self
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear the component registry."""
        cls._registry.clear()
    
    @classmethod
    def get_component(cls, comp_id: str) -> Optional['Component']:
        """Get a component by its ID."""
        return cls._registry.get(comp_id)

    @computed_field
    @property
    def current_var(self) -> str:
        """Returns the current variable name for this component."""
        return f"i_{self.comp_id}"
    
    @computed_field
    @property
    def voltage_var(self) -> str:
        """Returns the voltage variable name for this component."""
        return f"v_{self.comp_id}"
    
    # Use ConfigDict for configuration
    model_config = ConfigDict(frozen=False)

class Source(Component):
    """Source component."""
    pass

class Meter(Component):
    """Meter component."""
    pass

class Resistor(Component):
    """Resistor component."""
    resistance: float = Field(..., description="Resistance value in ohms", ge=0)
    
    def get_comp_eq(self) -> str:
        """Returns the symbolic equation for Ohm's law.
        
        Returns:
            str: Symbolic equation representing V = I * R, where V is voltage,
                 I is current, and R is resistance.
        """
        return f"{self.voltage_var} = {self.current_var} * {self.resistance}"

class Capacitor(Component):
    """Capacitor component."""
    capacitance: float = Field(..., description="Capacitance value in farads", ge=0)

class Inductor(Component):
    """Inductor component."""
    inductance: float = Field(..., description="Inductance value in henries", ge=0)

class PowerSwitch(Component):
    """Power switch component."""
    switch_time: float = Field(..., description="Time to switch in seconds")
    is_on: bool = Field(..., description="Whether the switch is on")
    
    def get_comp_eq(self) -> str:
        """Returns the symbolic equation for the switch based on its position.
        
        Returns:
            str: Symbolic equation. If switch is closed (1), returns voltage equation.
                 If switch is open (0), returns current equation.
        """
        if self.is_on:  # closed
            return f"{self.voltage_var} = 0"
        else:  # open
            return f"{self.current_var} = 0"
        
    def control_signal(self, t: float) -> int:
        """Returns the control signal for the switch. 
        If the switch is open, the control signal is 0.
        If the switch is closed, the control signal is 1.
        """
        if self.switch_time:
            if t >= self.switch_time:
                self.is_on = True
                return 1
            else:
                self.is_on = False
                return 0
        else:
            self.is_on = False
            return 0
        

class Diode(Component):
    """Diode component."""
    is_on: bool = Field(False, description="Whether the diode is conducting")
    
    def get_comp_eq(self) -> str:
        """Returns the symbolic equation for the diode based on its state.
        
        Returns:
            str: Symbolic equation. If diode is conducting, returns voltage equation.
                 If diode is not conducting, returns current equation.
        """
        if self.is_on:
            return f"{self.voltage_var} = 0"
        else:
            return f"{self.current_var} = 0"

class VoltageSource(Source):
    """Voltage source component."""
    voltage: float = Field(..., description="Voltage value in volts")

class CurrentSource(Source):
    """Current source component."""
    current: float = Field(..., description="Current value in amperes")

class Ground(Component):
    """Ground component."""
    pass

class Ammeter(Meter):
    """Ammeter component."""
    def get_comp_eq(self) -> str:
        """Returns the symbolic equation for the ammeter. Ideal ammeter has 0 voltage drop.
        
        Returns:
            str: Symbolic equation.
        """
        return f"{self.voltage_var} = 0"

class Voltmeter(Meter):
    """Voltmeter component."""
    def get_comp_eq(self) -> str:
        """Returns the symbolic equation for the voltmeter. Ideal voltmeter has 0 current.
        
        Returns:
            str: Symbolic equation.
        """
        return f"{self.current_var} = 0"

class ElecJunction(BaseModel):
    """Electrical junction component."""
    junction_id: int = Field(..., description="Unique identifier for the junction")
    _registry: ClassVar[Dict[int, 'ElecJunction']] = {}
    is_ground: bool = Field(False, description="Whether the junction is a ground")

    @field_validator('junction_id')
    @classmethod
    def validate_unique_junction_id(cls, v: int) -> int:
        """Validate that the junction ID is unique."""
        if v in cls._registry:
            raise ValueError(f"Junction ID '{v}' is already in use")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        self._registry[self.junction_id] = self

    @classmethod
    def clear_registry(cls) -> None:
        """Clear the junction registry."""
        cls._registry.clear()

    @computed_field
    @property
    def voltage_var(self) -> Optional[str]:
        """Returns the voltage variable name for this junction.
        Returns None if the junction is a ground node."""
        if self.is_ground:
            return None
        return f"V_{self.junction_id}"