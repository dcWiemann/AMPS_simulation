from abc import ABC
from typing import Optional, ClassVar, Dict, Any
from pydantic import BaseModel, Field, computed_field, field_validator
from pydantic import ConfigDict
from sympy import symbols, Symbol, Eq, Function
from sympy.abc import t

class Component(BaseModel, ABC):
    """Abstract base class for all circuit components."""
    comp_id: str = Field(..., description="Unique identifier for the component", pattern=r"^[A-Za-z].*")
    _registry: ClassVar[Dict[str, 'Component']] = {}
    # Simulation metadata (to be removed once Engine owns sim context)
    n_control_port: ClassVar[int] = 0
    n_states: ClassVar[int] = 0

    def is_short_circuit_state(self, cp_value: Any = None) -> bool:
        """Check if component behaves as a short circuit (zero impedance)."""
        return False
    
    @property
    def is_short_circuit(self) -> bool:
        """Compatibility property for existing callers."""
        return self.is_short_circuit_state()
    
    def is_open_circuit_state(self, cp_value: Any = None) -> bool:
        """Check if component behaves as an open circuit (infinite impedance)."""
        return False
    
    @property 
    def is_open_circuit(self) -> bool:
        """Compatibility property for existing callers."""
        return self.is_open_circuit_state()
    
    @field_validator('comp_id')
    @classmethod
    def validate_unique_comp_id(cls, v: str) -> str:
        """Validate that the component ID is unique."""
        if v in cls._registry:
            raise ValueError(f"Component ID '{v}' is already in use")
        return v
    
    def model_post_init(self, __context) -> None:
        """Post-initialization to handle registry insertion."""
        self._registry[self.comp_id] = self
    
    def __hash__(self):
        """Make components hashable by using comp_id."""
        return hash(self.comp_id)
    
    def __eq__(self, other):
        """Components are equal if they have the same comp_id."""
        if not isinstance(other, Component):
            return False
        return self.comp_id == other.comp_id
    
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
        return symbols(f"i_{self.comp_id}")
    
    @computed_field
    @property
    def voltage_var(self) -> str:
        """Returns the voltage variable name for this component."""
        return symbols(f"v_{self.comp_id}")
    
    # Use ConfigDict for configuration
    model_config = ConfigDict(frozen=False)

class Source(Component):
    """Source component."""
    input_var: Optional[str] = None
    control_port_name: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)

class Meter(Component):
    """Meter component."""
    output_var: Optional[str] = None

class Resistor(Component):
    """Resistor component."""
    resistance: float = Field(..., description="Resistance value in ohms")
    n_states: ClassVar[int] = 0

    def __init__(self, comp_id: str, resistance: float, **data):
        """Initialize resistor with positional arguments support."""
        super().__init__(comp_id=comp_id, resistance=resistance, **data)

    def is_short_circuit_state(self, cp_value: Any = None) -> bool:
        """Resistor is short circuit if R = 0."""
        return self.resistance == 0.0

    def is_open_circuit_state(self, cp_value: Any = None) -> bool:
        """Resistor is open circuit if R approaches infinity."""
        return self.resistance == float('inf')

    def get_comp_eq(self, voltage_var: Optional[Symbol] = None, current_var: Optional[Symbol] = None, cp_value: Any = None) -> Symbol:
        """Returns the symbolic equation for Ohm's law.
        
        Returns:
            Symbol: Symbolic equation representing V = I * R, where V is voltage,
                 I is current, and R is resistance.
        """
        v = voltage_var if voltage_var is not None else self.voltage_var
        i = current_var if current_var is not None else self.current_var
        return v - i * self.resistance

class Capacitor(Component):
    """Capacitor component."""
    capacitance: float = Field(..., description="Capacitance value in farads", ge=0)
    n_states: ClassVar[int] = 1

    def __init__(self, comp_id: str, capacitance: float, **data):
        """Initialize capacitor with positional arguments support."""
        super().__init__(comp_id=comp_id, capacitance=capacitance, **data)

    def is_short_circuit_state(self, cp_value: Any = None) -> bool:
        """Capacitor is short circuit if C approaches infinity."""
        return self.capacitance == float('inf')
    
    def is_open_circuit_state(self, cp_value: Any = None) -> bool:
        """Capacitor is open circuit if C = 0."""
        return self.capacitance == 0.0

    def get_comp_eq(self, voltage_var: Optional[Symbol] = None, current_var: Optional[Symbol] = None, cp_value: Any = None) -> Symbol:
        """Returns the symbolic equation for the capacitor.
        
        Returns:
            Symbol: Symbolic equation representing dV/dt = 1/C * I, where dV/dt is the derivative of voltage with respect to time,
                 C is capacitance, and I is current.
        """
        v = voltage_var if voltage_var is not None else self.voltage_var
        i = current_var if current_var is not None else self.current_var
        return Eq(v.diff(t), (1/self.capacitance) * i)
    
    @computed_field
    @property
    def voltage_var(self) -> str:
        """Returns the voltage variable as a function of time for this component."""
        return Function(f"v_{self.comp_id}")(t)

class Inductor(Component):
    """Inductor component."""
    inductance: float = Field(..., description="Inductance value in henries", ge=0)
    n_states: ClassVar[int] = 1

    def __init__(self, comp_id: str, inductance: float, **data):
        """Initialize inductor with positional arguments support."""
        super().__init__(comp_id=comp_id, inductance=inductance, **data)

    def is_short_circuit_state(self, cp_value: Any = None) -> bool:
        """Inductor is short circuit if L = 0."""
        return self.inductance == 0.0
    
    def is_open_circuit_state(self, cp_value: Any = None) -> bool:
        """Inductor is open circuit if L approaches infinity."""
        return self.inductance == float('inf')

    def get_comp_eq(self, voltage_var: Optional[Symbol] = None, current_var: Optional[Symbol] = None, cp_value: Any = None) -> Symbol:
        """Returns the symbolic equation for the inductor.
        
        Returns:
            Symbol: Symbolic equation representing dI/dt = 1/L * V, where dI/dt is the derivative of current with respect to time,
                 L is inductance, and V is voltage.
        """
        v = voltage_var if voltage_var is not None else self.voltage_var
        i = current_var if current_var is not None else self.current_var
        return Eq(i.diff(t), (1/self.inductance) * v)

    @computed_field
    @property
    def current_var(self) -> str:
        """Returns the current variable as a function of time for this component."""
        return Function(f"i_{self.comp_id}")(t)

class PowerSwitch(Component):
    """Power switch component."""
    switch_time: float = Field(..., description="Time to switch in seconds")
    is_on: bool = Field(..., description="Whether the switch is on")
    n_control_port: ClassVar[int] = 1
    
    def is_short_circuit_state(self, cp_value: Any = None) -> bool:
        """Switch is short circuit when closed (is_on = True)."""
        state = bool(cp_value) if cp_value is not None else self.is_on
        return state
    
    def is_open_circuit_state(self, cp_value: Any = None) -> bool:
        """Switch is open circuit when open (is_on = False)."""
        state = bool(cp_value) if cp_value is not None else self.is_on
        return not state
    
    def get_comp_eq(self, voltage_var: Optional[Symbol] = None, current_var: Optional[Symbol] = None, cp_value: Any = None) -> Symbol:
        """Returns the symbolic equation for the switch based on its position.
        
        Returns:
            Symbol: Symbolic equation. If switch is closed (1), returns voltage equation.
                 If switch is open (0), returns current equation.
        """
        v = voltage_var if voltage_var is not None else self.voltage_var
        i = current_var if current_var is not None else self.current_var
        state = bool(cp_value) if cp_value is not None else self.is_on
        if state:  # closed
            return v
        else:  # open
            return i
        
    def set_switch_state(self, t: float) -> int:
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
    n_control_port: ClassVar[int] = 0

    def __init__(self, comp_id: str, **data):
        """Initialize diode with positional arguments support."""
        super().__init__(comp_id=comp_id, **data)

    def is_short_circuit_state(self, cp_value: Any = None) -> bool:
        """Diode is short circuit when forward-biased and conducting."""
        state = bool(cp_value) if cp_value is not None else self.is_on
        return state
    
    def is_open_circuit_state(self, cp_value: Any = None) -> bool:
        """Diode is open circuit when reverse-biased or not conducting."""
        state = bool(cp_value) if cp_value is not None else self.is_on
        return not state
    
    def get_comp_eq(self, voltage_var: Optional[Symbol] = None, current_var: Optional[Symbol] = None, cp_value: Any = None) -> Symbol:
        """Returns the symbolic equation for the diode based on its state.
        
        Returns:
            Symbol: Symbolic equation. If diode is conducting, returns voltage equation.
                 If diode is not conducting, returns current equation.
        """
        v = voltage_var if voltage_var is not None else self.voltage_var
        i = current_var if current_var is not None else self.current_var
        state = bool(cp_value) if cp_value is not None else self.is_on
        if state:
            return v
        else:
            return i

class VoltageSource(Source):
    """Voltage source component."""
    voltage: float = Field(..., description="Voltage value in volts")
    
    def is_short_circuit_state(self, cp_value: Any = None) -> bool:
        """Voltage source is short circuit if voltage = 0."""
        return self.voltage == 0.0

    def __init__(self, comp_id: str, voltage: float, **data):
        """Initialize voltage source with positional arguments support."""
        super().__init__(comp_id=comp_id, voltage=voltage, **data)

    def model_post_init(self, __context) -> None:
        """Set input_var after initialization."""
        super().model_post_init(__context)  # Call parent's post_init
        self.input_var = self.voltage_var
        # control_port_name should be set by the parser if needed

    @computed_field
    @property
    def voltage_var(self) -> str:
        return Function(f"v_{self.comp_id}")(t)


class CurrentSource(Source):
    """Current source component."""
    current: float = Field(..., description="Current value in amperes")
    
    def is_open_circuit_state(self, cp_value: Any = None) -> bool:
        """Current source is open circuit if current = 0."""
        return self.current == 0.0

    def __init__(self, comp_id: str, current: float, **data):
        """Initialize current source with positional arguments support."""
        super().__init__(comp_id=comp_id, current=current, **data)

    def model_post_init(self, __context) -> None:
        """Set input_var after initialization."""
        super().model_post_init(__context)  # Call parent's post_init
        self.input_var = self.current_var
        # control_port_name should be set by the parser if needed

    @computed_field
    @property
    def current_var(self) -> str:
        return Function(f"i_{self.comp_id}")(t)

class Ground(Component):
    """Ground component."""
    pass

class Ammeter(Meter):
    """Ammeter component."""
    
    def is_short_circuit_state(self, cp_value: Any = None) -> bool:
        """Ideal ammeter has zero voltage drop (short circuit)."""
        return True

    def __init__(self, **data):
        super().__init__(**data)
        self.output_var = self.current_var

    def get_comp_eq(self, voltage_var: Optional[Symbol] = None, current_var: Optional[Symbol] = None, cp_value: Any = None) -> Symbol:
        """Returns the symbolic equation for the ammeter. Ideal ammeter has 0 voltage drop.
        
        Returns:
            Symbol: Symbolic equation.
        """
        v = voltage_var if voltage_var is not None else self.voltage_var
        return v

class Voltmeter(Meter):
    """Voltmeter component."""
    
    def is_open_circuit_state(self, cp_value: Any = None) -> bool:
        """Ideal voltmeter has infinite impedance (open circuit)."""
        return True

    def __init__(self, **data):
        super().__init__(**data)
        self.output_var = self.voltage_var

    def get_comp_eq(self, voltage_var: Optional[Symbol] = None, current_var: Optional[Symbol] = None, cp_value: Any = None) -> Symbol:
        """Returns the symbolic equation for the voltmeter. Ideal voltmeter has 0 current.
        
        Returns:
            Symbol: Symbolic equation.
        """
        i = current_var if current_var is not None else self.current_var
        return i

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
        return symbols(f"V_{self.junction_id}") if not self.is_ground else 0
