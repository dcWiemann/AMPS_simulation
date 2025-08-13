from typing import ClassVar, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator

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