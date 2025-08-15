"""
Test integration of circuit sanity checks with Engine class.
"""

import pytest
import networkx as nx
import logging
from amps_simulation.core.engine import Engine
from amps_simulation.core.components import (
    VoltageSource, CurrentSource, Resistor, ElecJunction, Component
)
from amps_simulation.core.circuit_sanity_checker import CircuitTopologyError

@pytest.fixture(autouse=True)
def clear_registries():
    """Clear component registries before each test."""
    Component.clear_registry()
    ElecJunction.clear_registry()
    yield
    Component.clear_registry()
    ElecJunction.clear_registry()

@pytest.fixture
def valid_graph():
    """Create a valid circuit graph for testing."""
    graph = nx.MultiDiGraph()
    
    # Create junctions
    j1 = ElecJunction(junction_id=1, is_ground=False)
    j2 = ElecJunction(junction_id=2, is_ground=True)  # Ground
    
    # Add nodes
    graph.add_node("1", junction=j1)
    graph.add_node("2", junction=j2)
    
    # Add valid circuit: voltage source and resistor
    vs = VoltageSource(comp_id="V1", voltage=5.0)
    resistor = Resistor(comp_id="R1", resistance=100.0)
    
    graph.add_edge("1", "2", component=vs)
    graph.add_edge("1", "2", component=resistor)  # Parallel load
    
    return graph

@pytest.fixture  
def invalid_graph():
    """Create an invalid circuit graph for testing."""
    graph = nx.MultiDiGraph()
    
    # Create junctions
    j1 = ElecJunction(junction_id=1, is_ground=False)
    j2 = ElecJunction(junction_id=2, is_ground=True)
    
    # Add nodes
    graph.add_node("1", junction=j1)
    graph.add_node("2", junction=j2)
    
    # Add invalid circuit: two voltage sources in parallel with different voltages
    vs1 = VoltageSource(comp_id="V1", voltage=5.0)
    vs2 = VoltageSource(comp_id="V2", voltage=3.0)
    
    graph.add_edge("1", "2", component=vs1)
    graph.add_edge("1", "2", component=vs2)  # Invalid: parallel voltage sources
    
    return graph

class TestEngineSanityChecks:
    """Test sanity check integration with Engine."""
    
    def test_valid_circuit_initialization(self, valid_graph, caplog):
        """Test that valid circuits initialize without issues."""
        with caplog.at_level(logging.INFO):
            engine = Engine(valid_graph)
            engine.initialize()
            
        # Check that sanity checks ran and passed
        assert "Running circuit topology sanity checks" in caplog.text
        assert "Circuit topology sanity checks passed" in caplog.text
        assert engine.initialized is True
        
    def test_invalid_circuit_initialization(self, invalid_graph):
        """Test that invalid circuits raise errors during initialization."""
        engine = Engine(invalid_graph)
        
        # Should raise CircuitTopologyError during initialization
        with pytest.raises(CircuitTopologyError):
            engine.initialize()
            
        # Engine should not be initialized after error
        assert engine.initialized is False
    
    def test_constraints_detected(self, caplog):
        """Test that circuit constraints are properly detected (but may not be fully applied yet)."""
        # Create a circuit with open circuit inductor
        graph = nx.MultiDiGraph()
        
        j1 = ElecJunction(junction_id=1, is_ground=True)
        j2 = ElecJunction(junction_id=2, is_ground=False)
        
        graph.add_node("1", junction=j1)
        graph.add_node("2", junction=j2)
        
        from amps_simulation.core.components import Inductor, Resistor
        inductor = Inductor(comp_id="L1", inductance=1e-3)
        resistor = Resistor(comp_id="R1", resistance=100.0)  # Complete circuit
        
        # Create a complete circuit with inductor
        graph.add_edge("1", "2", component=inductor)
        graph.add_edge("2", "1", component=resistor)  # Return path
        
        with caplog.at_level(logging.INFO):
            engine = Engine(graph)
            # Just test that sanity checks run, don't initialize fully yet
            engine._run_sanity_checks()
        
        # Check that constraints were detected and stored
        assert hasattr(engine, 'circuit_constraints')
        # Should be no constraints for a complete circuit
        assert engine.circuit_constraints is None or not any(engine.circuit_constraints.values())
        
        # Check that sanity checks passed
        assert "Circuit topology sanity checks passed" in caplog.text
    
    def test_warnings_logged(self, caplog):
        """Test that topology warnings are properly logged."""
        # Create a circuit that generates warnings but not errors
        graph = nx.MultiDiGraph()
        
        j1 = ElecJunction(junction_id=1, is_ground=True)
        j2 = ElecJunction(junction_id=2, is_ground=False)
        
        graph.add_node("1", junction=j1)
        graph.add_node("2", junction=j2)
        
        from amps_simulation.core.components import Capacitor, Ammeter
        capacitor = Capacitor(comp_id="C1", capacitance=1e-6)
        ammeter = Ammeter(comp_id="A1")  # Short circuits the capacitor
        
        graph.add_edge("1", "2", component=capacitor)
        graph.add_edge("1", "2", component=ammeter)  # Parallel short circuit
        
        with caplog.at_level(logging.WARNING):
            engine = Engine(graph)
            # Just test warnings detection, don't initialize fully
            engine._run_sanity_checks()
        
        # Check that warnings were logged
        assert "Circuit topology warnings detected" in caplog.text
        assert "short circuit" in caplog.text
        assert "v_C = 0" in caplog.text
    
    def test_no_constraints_case(self, valid_graph):
        """Test that circuits with no constraints work properly."""
        engine = Engine(valid_graph)
        engine.initialize()
        
        # Should have no constraints
        assert hasattr(engine, 'circuit_constraints')
        assert engine.circuit_constraints is None
        assert engine.initialized is True