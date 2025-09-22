from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from sympy import Symbol
from .components import Component
from .control_port import ControlPort


class Variable(BaseModel):
    """Represents a circuit variable with its associated component and symbolic representation."""

    component: Component = Field(..., description="The component this variable belongs to")
    symbolic: Symbol = Field(..., description="The sympy symbolic variable")
    control_port: Optional[ControlPort] = Field(None, description="Optional control port associated with this variable")

    model_config = ConfigDict(arbitrary_types_allowed=True)