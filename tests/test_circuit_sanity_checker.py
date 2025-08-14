"""
Tests for the circuit sanity checker module.
"""

import pytest
import networkx as nx
from amps_simulation.core.components import (
    VoltageSource, CurrentSource, Resistor, Capacitor, Inductor, 
    PowerSwitch, Diode, Ground, ElecJunction, Ammeter, Voltmeter,
    Component
)
from amps_simulation.core.circuit_sanity_checker import (
    CircuitSanityChecker, CircuitTopologyError, 
    has_short_circuit_path, has_current_path
)

@pytest.fixture(autouse=True)
def clear_registries():
    """Clear component registries before each test."""
    Component.clear_registry()
    ElecJunction.clear_registry()
    yield
    Component.clear_registry()
    ElecJunction.clear_registry()

@pytest.fixture
def basic_graph():
    """Create a basic graph for testing."""
    graph = nx.MultiDiGraph()
    
    # Create junctions
    j1 = ElecJunction(junction_id=1, is_ground=False)
    j2 = ElecJunction(junction_id=2, is_ground=False) 
    j3 = ElecJunction(junction_id=3, is_ground=True)
    
    # Add nodes
    graph.add_node("1", junction=j1)
    graph.add_node("2", junction=j2)
    graph.add_node("3", junction=j3)
    
    return graph

class TestComponentAttributes:
    """Test component is_short_circuit and is_open_circuit attributes."""
    
    def test_resistor_attributes(self):
        r_normal = Resistor(comp_id="R1", resistance=100.0)
        assert not r_normal.is_short_circuit
        assert not r_normal.is_open_circuit
        
        r_short = Resistor(comp_id="R2", resistance=0.0)
        assert r_short.is_short_circuit
        assert not r_short.is_open_circuit
        
        r_open = Resistor(comp_id="R3", resistance=float('inf'))
        assert not r_open.is_short_circuit
        assert r_open.is_open_circuit
    
    def test_capacitor_attributes(self):
        c_normal = Capacitor(comp_id="C1", capacitance=1e-6)
        assert not c_normal.is_short_circuit
        assert not c_normal.is_open_circuit
        
        c_open = Capacitor(comp_id="C2", capacitance=0.0)
        assert not c_open.is_short_circuit
        assert c_open.is_open_circuit
        
        c_short = Capacitor(comp_id="C3", capacitance=float('inf'))
        assert c_short.is_short_circuit
        assert not c_short.is_open_circuit
    
    def test_inductor_attributes(self):
        l_normal = Inductor(comp_id="L1", inductance=1e-3)
        assert not l_normal.is_short_circuit
        assert not l_normal.is_open_circuit
        
        l_short = Inductor(comp_id="L2", inductance=0.0)
        assert l_short.is_short_circuit
        assert not l_short.is_open_circuit
        
        l_open = Inductor(comp_id="L3", inductance=float('inf'))
        assert not l_open.is_short_circuit
        assert l_open.is_open_circuit
    
    def test_switch_attributes(self):
        switch_open = PowerSwitch(comp_id="S1", switch_time=1.0, is_on=False)
        assert not switch_open.is_short_circuit
        assert switch_open.is_open_circuit
        
        switch_closed = PowerSwitch(comp_id="S2", switch_time=1.0, is_on=True)
        assert switch_closed.is_short_circuit
        assert not switch_closed.is_open_circuit
    
    def test_diode_attributes(self):
        diode_off = Diode(comp_id="D1", is_on=False)
        assert not diode_off.is_short_circuit
        assert diode_off.is_open_circuit
        
        diode_on = Diode(comp_id="D2", is_on=True)
        assert diode_on.is_short_circuit
        assert not diode_on.is_open_circuit
    
    def test_source_attributes(self):
        vs_normal = VoltageSource(comp_id="V1", voltage=5.0)
        assert not vs_normal.is_short_circuit
        assert not vs_normal.is_open_circuit
        
        vs_short = VoltageSource(comp_id="V2", voltage=0.0)
        assert vs_short.is_short_circuit
        assert not vs_short.is_open_circuit
        
        cs_normal = CurrentSource(comp_id="I1", current=1.0)
        assert not cs_normal.is_short_circuit
        assert not cs_normal.is_open_circuit
        
        cs_open = CurrentSource(comp_id="I2", current=0.0)
        assert not cs_open.is_short_circuit
        assert cs_open.is_open_circuit
    
    def test_meter_attributes(self):
        ammeter = Ammeter(comp_id="A1")
        assert ammeter.is_short_circuit
        assert not ammeter.is_open_circuit
        
        voltmeter = Voltmeter(comp_id="V1")
        assert not voltmeter.is_short_circuit
        assert voltmeter.is_open_circuit

class TestPathFunctions:
    """Test the path detection helper functions."""
    
    def test_has_short_circuit_path(self, basic_graph):
        # Add components
        ammeter = Ammeter(comp_id="A1")  # Short circuit
        resistor = Resistor(comp_id="R1", resistance=100.0)  # Not short circuit
        
        basic_graph.add_edge("1", "2", component=ammeter)
        basic_graph.add_edge("2", "3", component=resistor)
        
        # Should find short circuit path through ammeter
        assert has_short_circuit_path(basic_graph, "1", "2")
        
        # Should not find short circuit path through resistor
        assert not has_short_circuit_path(basic_graph, "2", "3")
        
        # Should not find path from 1 to 3 (resistor blocks)
        assert not has_short_circuit_path(basic_graph, "1", "3")
    
    def test_has_current_path(self, basic_graph):
        # Add components
        voltmeter = Voltmeter(comp_id="V1")  # Open circuit
        resistor = Resistor(comp_id="R1", resistance=100.0)  # Allows current
        
        basic_graph.add_edge("1", "2", component=voltmeter)
        basic_graph.add_edge("2", "3", component=resistor)
        
        # Should not find current path through voltmeter
        assert not has_current_path(basic_graph, "1", "2")
        
        # Should find current path through resistor
        assert has_current_path(basic_graph, "2", "3")
        
        # Should not find path from 1 to 3 (voltmeter blocks)
        assert not has_current_path(basic_graph, "1", "3")

class TestSanityChecker:
    """Test the main sanity checker functionality."""
    
    def test_short_circuited_voltage_source(self, basic_graph):
        # Create voltage source and parallel ammeter (short circuit)
        vs = VoltageSource(comp_id="V1", voltage=5.0)
        ammeter = Ammeter(comp_id="A1")  # Creates short circuit
        
        basic_graph.add_edge("1", "2", component=vs)
        basic_graph.add_edge("1", "2", component=ammeter)  # Parallel short
        
        checker = CircuitSanityChecker(basic_graph)
        result = checker.check_all(raise_on_error=False)
        
        assert len(result['errors']) > 0
        assert any("short-circuited" in error for error in result['errors'])
    
    def test_open_circuit_current_source(self, basic_graph):
        # Create current source with no current path
        cs = CurrentSource(comp_id="I1", current=1.0)
        voltmeter = Voltmeter(comp_id="V1")  # Blocks current
        
        basic_graph.add_edge("1", "2", component=cs)
        basic_graph.add_edge("2", "3", component=voltmeter)  # No return path for current
        
        checker = CircuitSanityChecker(basic_graph)
        result = checker.check_all(raise_on_error=False)
        
        assert len(result['errors']) > 0
        assert any("open circuit" in error for error in result['errors'])
    
    def test_parallel_voltage_sources_always_error(self, basic_graph):
        # Create two voltage sources in parallel - always an error regardless of voltage
        vs1 = VoltageSource(comp_id="V1", voltage=5.0)
        vs2 = VoltageSource(comp_id="V2", voltage=3.0)
        
        basic_graph.add_edge("1", "2", component=vs1)
        basic_graph.add_edge("1", "2", component=vs2)  # Same nodes = parallel
        
        checker = CircuitSanityChecker(basic_graph)
        result = checker.check_all(raise_on_error=False)
        
        assert len(result['errors']) > 0
        assert any("parallel" in error and "not allowed" in error for error in result['errors'])
    
    def test_parallel_voltage_sources_same_voltage_still_error(self, basic_graph):
        # Create two voltage sources in parallel with same voltage - still an error
        vs1 = VoltageSource(comp_id="V1", voltage=5.0)
        vs2 = VoltageSource(comp_id="V2", voltage=5.0)
        
        basic_graph.add_edge("1", "2", component=vs1)
        basic_graph.add_edge("1", "2", component=vs2)
        
        checker = CircuitSanityChecker(basic_graph)
        result = checker.check_all(raise_on_error=False)
        
        assert len(result['errors']) > 0
        assert any("parallel" in error and "not allowed" in error for error in result['errors'])
    
    def test_series_current_sources_always_error(self):
        # Create graph with series current sources sharing a common node with return path
        graph = nx.MultiDiGraph()
        
        j1 = ElecJunction(junction_id=1, is_ground=True)
        j2 = ElecJunction(junction_id=2)
        j3 = ElecJunction(junction_id=3)
        
        graph.add_node("1", junction=j1)  # Ground
        graph.add_node("2", junction=j2)  # Shared node 
        graph.add_node("3", junction=j3)
        
        cs1 = CurrentSource(comp_id="I1", current=1.0)
        cs2 = CurrentSource(comp_id="I2", current=2.0)
        resistor = Resistor(comp_id="R1", resistance=100.0)  # Return path
        
        # Series connection: cs1 from 1 to 2, cs2 from 2 to 3, resistor from 3 back to 1
        graph.add_edge("1", "2", component=cs1)
        graph.add_edge("2", "3", component=cs2)
        graph.add_edge("3", "1", component=resistor)  # Complete the circuit
        
        checker = CircuitSanityChecker(graph)
        result = checker.check_all(raise_on_error=False)
        
        print("Errors:", result['errors'])  # Debug print
        assert len(result['errors']) > 0
        assert any("series" in error and "not allowed" in error for error in result['errors'])
    
    def test_series_current_sources_same_current_still_error(self):
        # Create graph with series current sources with same current - still an error
        graph = nx.MultiDiGraph()
        
        j1 = ElecJunction(junction_id=1, is_ground=True)
        j2 = ElecJunction(junction_id=2)
        j3 = ElecJunction(junction_id=3)
        
        graph.add_node("1", junction=j1)  # Ground
        graph.add_node("2", junction=j2)  # Shared node
        graph.add_node("3", junction=j3)
        
        cs1 = CurrentSource(comp_id="I1", current=1.0)
        cs2 = CurrentSource(comp_id="I2", current=1.0)  # Same current
        resistor = Resistor(comp_id="R1", resistance=100.0)  # Return path
        
        # Series connection: cs1 from 1 to 2, cs2 from 2 to 3, resistor from 3 back to 1
        graph.add_edge("1", "2", component=cs1)
        graph.add_edge("2", "3", component=cs2)
        graph.add_edge("3", "1", component=resistor)  # Complete the circuit
        
        checker = CircuitSanityChecker(graph)
        result = checker.check_all(raise_on_error=False)
        
        assert len(result['errors']) > 0
        assert any("series" in error and "not allowed" in error for error in result['errors'])
    
    def test_floating_nodes(self):
        # Create graph with floating nodes (no ground connection)
        graph = nx.MultiDiGraph()
        
        j1 = ElecJunction(junction_id=1)  # Not ground
        j2 = ElecJunction(junction_id=2)  # Not ground
        j3 = ElecJunction(junction_id=3, is_ground=True)  # Ground
        j4 = ElecJunction(junction_id=4)  # Floating
        
        graph.add_node("1", junction=j1)
        graph.add_node("2", junction=j2)
        graph.add_node("3", junction=j3)
        graph.add_node("4", junction=j4)  # This will be floating
        
        resistor1 = Resistor(comp_id="R1", resistance=100.0)
        resistor2 = Resistor(comp_id="R2", resistance=200.0)
        
        # Connect 1-2-3 (connected to ground)
        graph.add_edge("1", "2", component=resistor1)
        graph.add_edge("2", "3", component=resistor2)
        
        # Node 4 is isolated
        
        checker = CircuitSanityChecker(graph)
        result = checker.check_all(raise_on_error=False)
        
        assert len(result['errors']) > 0
        assert any("Floating nodes" in error for error in result['errors'])
    
    def test_open_circuit_inductor(self, basic_graph):
        # Create inductor that's the only current path
        inductor = Inductor(comp_id="L1", inductance=1e-3)
        voltmeter = Voltmeter(comp_id="V1")  # Blocks alternative path
        
        basic_graph.add_edge("1", "2", component=inductor)
        basic_graph.add_edge("1", "3", component=voltmeter)  # No alternative current path
        
        checker = CircuitSanityChecker(basic_graph)
        result = checker.check_all(raise_on_error=False)
        
        assert len(result['warnings']) > 0
        assert any("open circuit" in warning and "i_L = 0" in warning for warning in result['warnings'])
    
    def test_short_circuit_capacitor(self, basic_graph):
        # Create capacitor with parallel short circuit
        capacitor = Capacitor(comp_id="C1", capacitance=1e-6)
        ammeter = Ammeter(comp_id="A1")  # Creates short circuit
        
        basic_graph.add_edge("1", "2", component=capacitor)
        basic_graph.add_edge("1", "2", component=ammeter)  # Parallel short
        
        checker = CircuitSanityChecker(basic_graph)
        result = checker.check_all(raise_on_error=False)
        
        assert len(result['warnings']) > 0
        assert any("short circuit" in warning and "v_C = 0" in warning for warning in result['warnings'])
    
    def test_constraint_modifications(self, basic_graph):
        # Create scenarios requiring constraints
        inductor = Inductor(comp_id="L1", inductance=1e-3)
        capacitor = Capacitor(comp_id="C1", capacitance=1e-6)
        ammeter = Ammeter(comp_id="A1")
        
        # Open circuit inductor
        basic_graph.add_edge("1", "2", component=inductor)
        
        # Short circuit capacitor  
        basic_graph.add_edge("2", "3", component=capacitor)
        basic_graph.add_edge("2", "3", component=ammeter)
        
        checker = CircuitSanityChecker(basic_graph)
        constraints = checker.get_constraint_modifications()
        
        assert "L1" in constraints['zero_current_inductors']
        assert "L1" in constraints['zero_voltage_inductors']
        assert "C1" in constraints['zero_voltage_capacitors']
    
    def test_raise_on_error(self, basic_graph):
        # Create error condition
        vs1 = VoltageSource(comp_id="V1", voltage=5.0)
        vs2 = VoltageSource(comp_id="V2", voltage=3.0)
        
        basic_graph.add_edge("1", "2", component=vs1)
        basic_graph.add_edge("1", "2", component=vs2)
        
        checker = CircuitSanityChecker(basic_graph)
        
        # Should raise exception
        with pytest.raises(CircuitTopologyError):
            checker.check_all(raise_on_error=True)
        
        # Should not raise exception
        result = checker.check_all(raise_on_error=False)
        assert len(result['errors']) > 0
    
    def test_no_errors_or_warnings(self, basic_graph):
        # Create a valid circuit
        vs = VoltageSource(comp_id="V1", voltage=5.0)
        resistor = Resistor(comp_id="R1", resistance=100.0)
        
        basic_graph.add_edge("1", "2", component=vs)
        basic_graph.add_edge("2", "3", component=resistor)
        
        checker = CircuitSanityChecker(basic_graph)
        result = checker.check_all(raise_on_error=False)
        
        assert len(result['errors']) == 0
        assert len(result['warnings']) == 0