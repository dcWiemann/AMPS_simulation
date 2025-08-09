from typing import Callable, Any, ClassVar, Dict
from pydantic import BaseModel, Field, field_validator

class ControlSignal(BaseModel):
    """
    Represents a control signal with a unique name and a time-dependent control function.
    """
    name: str = Field(..., description="Unique name for the control signal")
    control_function: Callable[..., Any] = Field(..., description="Function of t (and optional args) returning the signal value")

    _registry: ClassVar[Dict[str, 'ControlSignal']] = {}

    @field_validator('name')
    @classmethod
    def validate_unique_name(cls, v: str) -> str:
        """Validate that the control signal name is unique."""
        if v in cls._registry:
            raise ValueError(f"ControlSignal name '{v}' is already in use")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        self._registry[self.name] = self

    @classmethod
    def clear_registry(cls) -> None:
        """Clear the control signal registry."""
        cls._registry.clear()

    @classmethod
    def get_control_signal(cls, name: str) -> 'ControlSignal':
        """Get a control signal by its name."""
        return cls._registry.get(name)

    def __call__(self, t, *args, **kwargs):
        """
        Call the control function with time t and optional arguments.
        """
        return self.control_function(t, *args, **kwargs)
