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
from amps_simulation.core.electrical_model import ElectricalModel

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

    def test_ground_node_check_no_ground(self):
        # Create graph without ground node
        graph = nx.MultiDiGraph()

        j1 = ElecJunction(junction_id=1, is_ground=False)
        j2 = ElecJunction(junction_id=2, is_ground=False)

        graph.add_node("1", junction=j1)
        graph.add_node("2", junction=j2)

        r1 = Resistor(comp_id="R1", resistance=100.0)
        graph.add_edge("1", "2", component=r1)

        checker = CircuitSanityChecker(graph)
        result = checker.check_all(raise_on_error=False)

        # Should have error about missing ground
        assert len(result['errors']) > 0
        assert any("ground reference node" in error.lower() for error in result['errors'])

    def test_ground_node_check_with_ground(self):
        # Create graph with ground node
        graph = nx.MultiDiGraph()

        j1 = ElecJunction(junction_id=1, is_ground=True)  # Ground
        j2 = ElecJunction(junction_id=2, is_ground=False)

        graph.add_node("1", junction=j1)
        graph.add_node("2", junction=j2)

        r1 = Resistor(comp_id="R1", resistance=100.0)
        graph.add_edge("1", "2", component=r1)

        checker = CircuitSanityChecker(graph)
        result = checker.check_all(raise_on_error=False)

        # Should not have ground-related errors
        assert not any("ground reference node" in error.lower() for error in result['errors'])

    def test_ground_node_check_raises_error(self):
        # Test that missing ground raises exception when raise_on_error=True
        graph = nx.MultiDiGraph()

        j1 = ElecJunction(junction_id=1, is_ground=False)
        graph.add_node("1", junction=j1)

        checker = CircuitSanityChecker(graph)

        with pytest.raises(CircuitTopologyError):
            checker.check_all(raise_on_error=True)

class TestIslandDetection:
    """Test the island detection functionality."""

    def test_no_islands_when_all_connected_to_ground(self):
        """Test that no islands are detected when all nodes connect to ground."""
        model = ElectricalModel()

        # Add nodes
        model.add_node(1, is_ground=True)
        model.add_node(2)
        model.add_node(3)

        # Add components
        r1 = Resistor(comp_id="R1", resistance=100.0)
        r2 = Resistor(comp_id="R2", resistance=200.0)

        model.add_component(r1, p=1, n=2)
        model.add_component(r2, p=2, n=3)

        checker = CircuitSanityChecker(model.graph)
        islands = checker.detect_islands()

        assert len(islands) == 0

    def test_single_island_isolated_by_open_switch(self):
        """Test detection of single island isolated by an open switch."""
        model = ElectricalModel()

        # Add nodes
        model.add_node(0, is_ground=True)
        model.add_node(1)
        model.add_node(2)  # Island
        model.add_node(3)  # Island

        # Main circuit
        v1 = VoltageSource(comp_id="V1", voltage=10.0)
        r1 = Resistor(comp_id="R1", resistance=100.0)
        model.add_component(v1, p=1, n=0)
        model.add_component(r1, p=1, n=0)

        # Island components (internal)
        r2 = Resistor(comp_id="R2", resistance=50.0)
        c1 = Capacitor(comp_id="C1", capacitance=1e-6)
        model.add_component(r2, p=2, n=3)
        model.add_component(c1, p=3, n=2)

        # Boundary component (open switch)
        switch = PowerSwitch(comp_id="S1", is_on=False, switch_time=1.0)
        model.add_component(switch, p=1, n=2)

        checker = CircuitSanityChecker(model.graph)
        islands = checker.detect_islands()

        # Verify one island detected
        assert len(islands) == 1

        island = islands[0]
        assert island['island_id'] == 0
        assert island['node_count'] == 2
        assert set(island['nodes']) == {2, 3}

        # Verify internal components
        assert len(island['internal_components']) == 2
        internal_ids = {comp.comp_id for _, _, comp in island['internal_components']}
        assert internal_ids == {"R2", "C1"}

        # Verify boundary components
        assert len(island['boundary_components']) == 1
        boundary_comp = island['boundary_components'][0]
        assert boundary_comp[2].comp_id == "S1"
        assert boundary_comp[2].is_open_circuit

    def test_single_island_isolated_by_blocking_diode(self):
        """Test island isolated by a blocking diode."""
        model = ElectricalModel()

        # Add nodes
        model.add_node(0, is_ground=True)
        model.add_node(1)
        model.add_node(2)  # Island node
        model.add_node(3)  # Island node

        # Main circuit
        r1 = Resistor(comp_id="R1", resistance=100.0)
        model.add_component(r1, p=1, n=0)

        # Island circuit (isolated from ground)
        r2 = Resistor(comp_id="R2", resistance=50.0)
        model.add_component(r2, p=2, n=3)

        # Boundary (blocking diode)
        diode = Diode(comp_id="D1", is_on=False)
        model.add_component(diode, p=1, n=2)

        checker = CircuitSanityChecker(model.graph)
        islands = checker.detect_islands()

        assert len(islands) == 1
        assert islands[0]['node_count'] == 2
        assert set(islands[0]['nodes']) == {2, 3}

        # Verify boundary component is the diode
        assert len(islands[0]['boundary_components']) == 1
        assert islands[0]['boundary_components'][0][2].comp_id == "D1"

    def test_multiple_islands(self):
        """Test detection of multiple independent islands."""
        model = ElectricalModel()

        # Add nodes
        model.add_node(0, is_ground=True)
        model.add_node(1)
        model.add_node(2)  # First island
        model.add_node(3)  # First island
        model.add_node(4)  # Second island
        model.add_node(5)  # Second island

        # Main circuit
        r_main = Resistor(comp_id="R_main", resistance=100.0)
        model.add_component(r_main, p=1, n=0)

        # First island
        r2 = Resistor(comp_id="R2", resistance=50.0)
        model.add_component(r2, p=2, n=3)

        # Second island
        r3 = Resistor(comp_id="R3", resistance=75.0)
        model.add_component(r3, p=4, n=5)

        # Boundary components
        s1 = PowerSwitch(comp_id="S1", is_on=False, switch_time=1.0)
        s2 = PowerSwitch(comp_id="S2", is_on=False, switch_time=2.0)
        model.add_component(s1, p=1, n=2)
        model.add_component(s2, p=1, n=4)

        checker = CircuitSanityChecker(model.graph)
        islands = checker.detect_islands()

        # Should detect two islands
        assert len(islands) == 2

        # Verify island IDs are sequential
        island_ids = {island['island_id'] for island in islands}
        assert island_ids == {0, 1}

        # Each island should have 2 nodes
        for island in islands:
            assert island['node_count'] == 2

    def test_island_with_multiple_boundary_components(self):
        """Test island connected via multiple open components."""
        model = ElectricalModel()

        # Add nodes
        model.add_node(0, is_ground=True)
        model.add_node(1)
        model.add_node(2)  # Island
        model.add_node(3)  # Island

        # Main circuit
        r1 = Resistor(comp_id="R1", resistance=100.0)
        model.add_component(r1, p=1, n=0)

        # Island
        r2 = Resistor(comp_id="R2", resistance=50.0)
        model.add_component(r2, p=2, n=3)

        # Multiple boundary components
        s1 = PowerSwitch(comp_id="S1", is_on=False, switch_time=1.0)
        d1 = Diode(comp_id="D1", is_on=False)
        vm = Voltmeter(comp_id="VM1")

        model.add_component(s1, p=1, n=2)
        model.add_component(d1, p=1, n=3)
        model.add_component(vm, p=1, n=2)

        checker = CircuitSanityChecker(model.graph)
        islands = checker.detect_islands()

        assert len(islands) == 1
        island = islands[0]

        # Should have 3 boundary components
        assert len(island['boundary_components']) == 3
        boundary_ids = {comp.comp_id for _, _, comp in island['boundary_components']}
        assert boundary_ids == {"S1", "D1", "VM1"}

    def test_two_islands_connected_to_each_other(self):
        """Test two islands connected to each other but not to ground."""
        model = ElectricalModel()

        # Add nodes
        model.add_node(0, is_ground=True)
        model.add_node(1)
        model.add_node(2)  # Island node
        model.add_node(3)  # Island node

        # Main circuit
        r1 = Resistor(comp_id="R1", resistance=100.0)
        model.add_component(r1, p=1, n=0)

        # Islands isolated from main but connected to each other via closed switch
        s1 = PowerSwitch(comp_id="S1", is_on=False, switch_time=1.0)
        s2 = PowerSwitch(comp_id="S2", is_on=False, switch_time=2.0)
        s3 = PowerSwitch(comp_id="S3", is_on=True, switch_time=3.0)  # Closed switch connecting islands

        model.add_component(s1, p=1, n=2)  # Main to node 2 (open)
        model.add_component(s2, p=1, n=3)  # Main to node 3 (open)
        model.add_component(s3, p=2, n=3)  # Node 2 to node 3 (closed)

        checker = CircuitSanityChecker(model.graph)
        islands = checker.detect_islands()

        # Nodes 2 and 3 should be in SAME island (connected via S3)
        assert len(islands) == 1
        island = islands[0]
        assert island['node_count'] == 2
        assert set(island['nodes']) == {2, 3}

        # S3 is internal (connects island nodes)
        assert len(island['internal_components']) == 1
        assert island['internal_components'][0][2].comp_id == "S3"

        # S1 and S2 are boundary (connect to external)
        assert len(island['boundary_components']) == 2
        boundary_ids = {comp.comp_id for _, _, comp in island['boundary_components']}
        assert boundary_ids == {"S1", "S2"}

    def test_island_structure_fields(self):
        """Test that island dictionary has all required fields."""
        model = ElectricalModel()

        # Add nodes
        model.add_node(0, is_ground=True)
        model.add_node(1)
        model.add_node(2)  # Island

        # Main circuit
        r1 = Resistor(comp_id="R1", resistance=100.0)
        model.add_component(r1, p=1, n=0)

        # Boundary to island
        s1 = PowerSwitch(comp_id="S1", is_on=False, switch_time=1.0)
        model.add_component(s1, p=1, n=2)

        checker = CircuitSanityChecker(model.graph)
        islands = checker.detect_islands()

        assert len(islands) == 1
        island = islands[0]

        # Verify all required fields exist
        assert 'island_id' in island
        assert 'nodes' in island
        assert 'node_count' in island
        assert 'internal_components' in island
        assert 'boundary_components' in island

        # Verify field types
        assert isinstance(island['island_id'], int)
        assert isinstance(island['nodes'], set)
        assert isinstance(island['node_count'], int)
        assert isinstance(island['internal_components'], list)
        assert isinstance(island['boundary_components'], list)

        # Verify consistency
        assert island['node_count'] == len(island['nodes'])