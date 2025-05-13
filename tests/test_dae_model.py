import pytest
from amps_simulation.core.dae_model import DaeModel
from typing import Dict


class SimpleDaeModel(DaeModel):
    """A simple concrete implementation of DaeModel for testing purposes."""
    
    def evaluate(self, t: float, states: Dict[str, float], inputs: Dict[str, float]) -> None:
        """Implement a simple DAE model: dx/dt = -x + u, y = x."""
        x = states.get('x', 0.0)
        u = inputs.get('u', 0.0)
        
        self.derivatives['x'] = -x + u
        self.outputs['y'] = x


def test_dae_model_initialization():
    """Test that a DAE model initializes with empty dictionaries."""
    model = SimpleDaeModel()
    assert model.derivatives == {}
    assert model.outputs == {}


def test_dae_model_getters():
    """Test the getter methods for derivatives and outputs."""
    model = SimpleDaeModel()
    states = {'x': 1.0}
    inputs = {'u': 2.0}
    
    model.evaluate(t=0.0, states=states, inputs=inputs)
    
    derivatives = model.get_derivatives()
    outputs = model.get_outputs()
    
    assert derivatives['x'] == 1.0
    assert outputs['y'] == 1.0 