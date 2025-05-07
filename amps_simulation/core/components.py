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
    model_config = ConfigDict(frozen=True)

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

class ElecJunction(BaseModel):
    """Electrical junction component."""
    junction_id: int = Field(..., description="Unique identifier for the junction")
    _registry: ClassVar[Dict[int, 'ElecJunction']] = {}

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
    def voltage_var(self) -> str:
        """Returns the voltage variable name for this junction."""
        return f"V_{self.junction_id}"