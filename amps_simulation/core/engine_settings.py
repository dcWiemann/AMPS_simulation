from pydantic import BaseModel, Field, field_validator
from typing import Literal

class EngineSettings(BaseModel):
    """
    Settings for the simulation engine.
    
    This class contains all the configuration parameters needed to run a simulation.
    
    Attributes:
        start_time (float): The starting time of the simulation in seconds.
        time_span (float): The duration of the simulation in seconds.
        solver (Literal['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']): 
            The numerical solver to use for the simulation.
        max_step_size (float): Maximum step size for the solver in seconds.
        init_step_size (float): Initial step size for the solver in seconds.
        rel_tol (float): Relative tolerance for the solver.
        abs_tol (float): Absolute tolerance for the solver.
    """
    start_time: float = Field(default=0.0, description="The starting time of the simulation in seconds")
    time_span: float = Field(default=1.0, description="The duration of the simulation in seconds")
    solver: Literal['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'] = Field(
        default='RK45',
        description="The numerical solver to use for the simulation"
    )
    max_step_size: float = Field(
        default=1e-3,
        description="Maximum step size for the solver in seconds"
    )
    init_step_size: float = Field(
        default=None,
        description="Initial step size for the solver in seconds"
    )
    rel_tol: float = Field(
        default=1e-3,
        description="Relative tolerance for the solver"
    )
    abs_tol: float = Field(
        default=1e-6,
        description="Absolute tolerance for the solver"
    )


    @field_validator('max_step_size')
    def validate_max_step_size(cls, v):
        """Validate that max_step_size is positive."""
        if v <= 0:
            raise ValueError("max_step_size must be positive")
        return v

    @field_validator('init_step_size')
    def validate_init_step_size(cls, v):
        """Validate that init_step_size is non-negative."""
        if v < 0:
            raise ValueError("init_step_size must be non-negative")
        return v

    @field_validator('rel_tol', 'abs_tol')
    def validate_tolerances(cls, v):
        """Validate that tolerances are positive."""
        if v <= 0:
            raise ValueError("tolerances must be positive")
        return v 