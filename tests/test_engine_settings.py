import pytest
from pydantic import ValidationError
from amps_simulation.core.engine_settings import EngineSettings

def test_engine_settings_default_values():
    """Test that EngineSettings initializes with correct default values."""
    settings = EngineSettings()
    
    assert settings.start_time == 0.0
    assert settings.time_span == 1.0
    assert settings.solver == 'RK45'
    assert settings.max_step_size == 1e-3
    assert settings.init_step_size == None
    assert settings.rel_tol == 1e-3
    assert settings.abs_tol == 1e-6

def test_engine_settings_custom_values():
    """Test that EngineSettings can be initialized with custom values."""
    settings = EngineSettings(
        start_time=1.0,
        time_span=10.0,
        solver='Radau',
        max_step_size=0.1,
        init_step_size=0.01,
        rel_tol=1e-4,
        abs_tol=1e-7
    )
    
    assert settings.start_time == 1.0
    assert settings.time_span == 10.0
    assert settings.solver == 'Radau'
    assert settings.max_step_size == 0.1
    assert settings.init_step_size == 0.01
    assert settings.rel_tol == 1e-4
    assert settings.abs_tol == 1e-7

def test_engine_settings_invalid_solver():
    """Test that EngineSettings raises an error for invalid solver."""
    with pytest.raises(ValidationError):
        EngineSettings(solver='InvalidSolver')

def test_engine_settings_validation():
    """Test validation of numeric fields."""
    # Test negative max_step_size
    with pytest.raises(ValidationError):
        EngineSettings(max_step_size=-0.1)
    
    # Test negative init_step_size
    with pytest.raises(ValidationError):
        EngineSettings(init_step_size=-0.01)
    
    # Test negative tolerances
    with pytest.raises(ValidationError):
        EngineSettings(rel_tol=-1e-3)
    with pytest.raises(ValidationError):
        EngineSettings(abs_tol=-1e-6)

def test_engine_settings_serialization():
    """Test that EngineSettings can be serialized to and from JSON."""
    settings = EngineSettings(
        start_time=1.0,
        time_span=10.0,
        solver='Radau',
        max_step_size=0.1,
        init_step_size=0.01,
        rel_tol=1e-4,
        abs_tol=1e-7
    )
    
    # Test serialization to dict
    settings_dict = settings.model_dump()
    assert isinstance(settings_dict, dict)
    assert settings_dict['start_time'] == 1.0
    assert settings_dict['solver'] == 'Radau'
    
    # Test serialization to JSON
    settings_json = settings.model_dump_json()
    assert isinstance(settings_json, str)
    
    # Test deserialization from dict
    new_settings = EngineSettings(**settings_dict)
    assert new_settings == settings 