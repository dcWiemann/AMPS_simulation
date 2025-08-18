"""
Test diode integration with LCP solver using the 4-diode rectifier circuit.

This test verifies:
1. Diode components are correctly parsed and initialized
2. LCP solver correctly determines diode conducting states
3. Diode state switching occurs properly during simulation
4. Circuit produces expected rectifier behavior
"""

import pytest
import numpy as np
import json
import os
import sys

# Add project to path for test environment
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from amps_simulation.run_simulation import run_simulation_from_file
from amps_simulation.core.parser import ParserJson
from amps_simulation.core.engine import Engine
from amps_simulation.core.lcp_solver import DiodeLCPSolver


class TestDiodeIntegration:
    """Test suite for diode integration with LCP solver."""
    
    @pytest.fixture
    def rectifier_circuit_file(self):
        """Path to the 4-diode rectifier test circuit."""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           'test_data', 'diodes_rect4.json')
    
    @pytest.fixture
    def rc_diode_on_file(self):
        """Path to the RC-diode circuit with +5V (conducting case)."""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           'test_data', 'diodes_rc_on.json')
    
    @pytest.fixture
    def rc_diode_off_file(self):
        """Path to the RC-diode circuit with -5V (blocking case)."""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           'test_data', 'diodes_rc_off.json')
    
    def test_rectifier_circuit_exists(self, rectifier_circuit_file):
        """Test that the rectifier circuit file exists."""
        assert os.path.exists(rectifier_circuit_file), f"Test circuit file not found: {rectifier_circuit_file}"
        
        # Verify it's valid JSON
        with open(rectifier_circuit_file, 'r') as f:
            circuit_data = json.load(f)
        assert 'nodes' in circuit_data
        assert 'edges' in circuit_data
    
    def test_diode_parsing(self, rectifier_circuit_file):
        """Test that diodes are correctly parsed from JSON."""
        with open(rectifier_circuit_file, 'r') as f:
            circuit_data = json.load(f)
        
        parser = ParserJson()
        graph, control_graph = parser.parse(circuit_data)
        
        # Count diode components
        diode_count = 0
        for _, _, data in graph.edges(data=True):
            if hasattr(data['component'], '__class__') and data['component'].__class__.__name__ == 'Diode':
                diode_count += 1
        
        assert diode_count == 4, f"Expected 4 diodes, found {diode_count}"
    
    def test_engine_initialization_with_diodes(self, rectifier_circuit_file):
        """Test that the engine correctly initializes with diodes."""
        with open(rectifier_circuit_file, 'r') as f:
            circuit_data = json.load(f)
        
        parser = ParserJson()
        graph, control_graph = parser.parse(circuit_data)
        
        engine = Engine(graph, control_graph)
        # Use nonzero initial capacitor voltage to break diode ambiguity
        initial_conditions = [1.0]  # v_C2 = 1V
        initial_inputs = [0.0]      # v_V8 = 0V (sin(0) = 0)
        engine.initialize(initial_conditions=initial_conditions, initial_inputs=initial_inputs)
        
        # Check that diodes are found and initialized
        assert hasattr(engine, 'diode_list'), "Engine should have diode_list attribute"
        assert len(engine.diode_list) == 4, f"Expected 4 diodes in engine, found {len(engine.diode_list)}"
        
        # Verify DAE model has diode support
        assert hasattr(engine.electrical_model, 'diode_list'), "DAE model should have diode_list"
        assert len(engine.electrical_model.diode_list) == 4, "DAE model should track 4 diodes"
        
        # Verify LCP solver is initialized
        assert hasattr(engine.electrical_model, 'lcp_solver'), "DAE model should have LCP solver"
        assert isinstance(engine.electrical_model.lcp_solver, DiodeLCPSolver), "Should use DiodeLCPSolver"
        
        # Test that simulation can run with nonzero initial capacitor voltage to avoid diode ambiguity
        # Using 1V initial capacitor voltage breaks the v_D=0, i_D=0 degeneracy
        try:
            result = engine.run_simulation(
                t_span=(0, 0.1),  # Short test run
                initial_conditions=[1.0]  # v_C2 = 1V to break diode ambiguity
            )
            assert result['success'], f"Engine simulation should succeed with nonzero initial conditions: {result.get('message', 'Unknown error')}"
        except Exception as e:
            pytest.fail(f"Engine simulation with nonzero initial conditions failed: {e}")
    
    def test_lcp_solver_functionality(self, rectifier_circuit_file):
        """Test that the LCP solver can determine diode states."""
        with open(rectifier_circuit_file, 'r') as f:
            circuit_data = json.load(f)
        
        parser = ParserJson()
        graph, control_graph = parser.parse(circuit_data)
        
        engine = Engine(graph, control_graph)
        engine.initialize()
        
        # Get initial state and input values for testing
        n_states = len(engine.state_vars)
        n_inputs = len(engine.input_vars)
        
        state_values = np.zeros(n_states)
        input_values = np.array([1.0])  # Positive voltage
        
        # Test LCP matrix computation
        dae_model = engine.electrical_model
        try:
            M_matrix, q_vector = dae_model.compute_diode_lcp_matrices(state_values, input_values)
            assert M_matrix.shape[0] == 4, f"Expected 4x4 LCP matrix, got {M_matrix.shape}"
            assert q_vector.shape[0] == 4, f"Expected 4x1 LCP vector, got {q_vector.shape}"
        except Exception as e:
            pytest.fail(f"LCP matrix computation failed: {e}")
        
        # Test diode state detection
        try:
            diode_states = dae_model.detect_diode_states(state_values, input_values)
            assert len(diode_states) == 4, f"Expected 4 diode states, got {len(diode_states)}"
            assert all(isinstance(state, bool) for state in diode_states), "All diode states should be boolean"
        except Exception as e:
            pytest.fail(f"Diode state detection failed: {e}")
    
    def test_rectifier_simulation_short(self, rectifier_circuit_file):
        """Test that the rectifier simulation runs successfully for a short duration."""
        try:
            result = run_simulation_from_file(
                rectifier_circuit_file,
                t_span=(0, 0.5),  # Short simulation
                plot_results=False
            )
            
            assert result is not None, "Simulation should return results"
            assert result['success'] == True, f"Simulation should succeed: {result.get('message', 'Unknown error')}"
            assert len(result['t']) > 10, "Should have reasonable number of time points"
            
            # Verify we have state variables (capacitor)
            if 'y' in result and result['y'] is not None:
                states = np.array(result['y'])
                assert states.shape[0] >= 1, "Should have at least one state variable (capacitor)"
                
        except Exception as e:
            pytest.fail(f"Short rectifier simulation failed: {e}")
    
    def test_rectifier_simulation_full_cycle(self, rectifier_circuit_file):
        """Test rectifier simulation over multiple input cycles."""
        try:
            result = run_simulation_from_file(
                rectifier_circuit_file,
                t_span=(0, 2),  # 2 seconds = 2 full cycles of sin(2*pi*t)
                plot_results=False
            )
            
            assert result is not None, "Simulation should return results"
            assert result['success'] == True, f"Simulation should succeed: {result.get('message', 'Unknown error')}"
            
            t = np.array(result['t'])
            assert len(t) > 100, "Should have good time resolution"
            assert t[-1] >= 1.9, "Should simulate for nearly full duration"
            
            # Check that we have state evolution (capacitor charging/discharging)
            if 'y' in result and result['y'] is not None:
                states = np.array(result['y'])
                if states.shape[0] > 0:
                    # Capacitor voltage should show some variation due to rectification
                    capacitor_voltage = states[0, :]  # First (and likely only) state variable
                    voltage_range = np.max(capacitor_voltage) - np.min(capacitor_voltage)
                    assert voltage_range > 0.01, "Capacitor voltage should show variation from rectification"
                    
        except Exception as e:
            pytest.fail(f"Full cycle rectifier simulation failed: {e}")
    
    def test_diode_state_switching(self, rectifier_circuit_file):
        """Test that diodes actually switch states during simulation."""
        with open(rectifier_circuit_file, 'r') as f:
            circuit_data = json.load(f)
        
        parser = ParserJson()
        graph, control_graph = parser.parse(circuit_data)
        
        engine = Engine(graph, control_graph)
        engine.initialize()
        
        dae_model = engine.electrical_model
        
        # Test diode states at different input voltages
        n_states = len(engine.state_vars)
        state_values = np.zeros(n_states)
        
        # Positive input voltage
        input_pos = np.array([1.0])
        states_pos = dae_model.detect_diode_states(state_values, input_pos)
        
        # Negative input voltage  
        input_neg = np.array([-1.0])
        states_neg = dae_model.detect_diode_states(state_values, input_neg)
        
        # At least some diodes should be in different states for opposite input polarities
        assert states_pos != states_neg, "Diode states should change with input polarity"
        
        # Count conducting diodes for each case
        num_conducting_pos = sum(states_pos)
        num_conducting_neg = sum(states_neg)
        
        # For a rectifier, we expect different numbers of conducting diodes
        # or at least some change in the pattern
        assert (num_conducting_pos != num_conducting_neg or 
                any(s1 != s2 for s1, s2 in zip(states_pos, states_neg))), \
               "Diode conduction pattern should change with input polarity"
    
    def test_rectifier_performance(self, rectifier_circuit_file):
        """Test simulation performance and numerical stability."""
        try:
            result = run_simulation_from_file(
                rectifier_circuit_file,
                t_span=(0, 1),  # 1 second
                plot_results=False
            )
            
            assert result is not None, "Simulation should complete"
            assert result['success'] == True, "Simulation should be numerically stable"
            
            # Check for reasonable computational performance
            t = np.array(result['t'])
            time_points = len(t)
            
            # With 1ms max step size, 1 second should have around 1000 points
            assert 500 <= time_points <= 2000, f"Expected 500-2000 time points, got {time_points}"
            
            # Verify no NaN or infinite values
            if 'y' in result and result['y'] is not None:
                states = np.array(result['y'])
                assert np.all(np.isfinite(states)), "All state values should be finite"
                
        except Exception as e:
            pytest.fail(f"Performance test failed: {e}")
    
    def test_diode_ids_and_names(self, rectifier_circuit_file):
        """Test that diodes have correct IDs and can be identified."""
        with open(rectifier_circuit_file, 'r') as f:
            circuit_data = json.load(f)
        
        parser = ParserJson()
        graph, control_graph = parser.parse(circuit_data)
        
        engine = Engine(graph, control_graph)
        engine.initialize()
        
        # Check diode IDs
        expected_diode_ids = {'D3', 'D4', 'D5', 'D6'}
        actual_diode_ids = {diode.comp_id for diode in engine.diode_list}
        
        assert actual_diode_ids == expected_diode_ids, \
               f"Expected diodes {expected_diode_ids}, found {actual_diode_ids}"
        
        # Verify diodes can be accessed by the LCP solver
        dae_model = engine.electrical_model
        diode_names = [diode.comp_id for diode in dae_model.diode_list]
        assert len(diode_names) == 4, "LCP solver should track all 4 diodes"
        assert all(name in expected_diode_ids for name in diode_names), \
               "All diode names should match expected IDs"
    
    def test_rc_diode_on_lcp_formulation(self, rc_diode_on_file):
        """Test LCP formulation for RC-diode circuit with +5V (conducting case).
        
        Expected: M = 2.0 (resistance), q = -5.0 (forward bias)
        Result: Diode should be conducting (True)
        """
        # Verify file exists
        assert os.path.exists(rc_diode_on_file), f"Test circuit file not found: {rc_diode_on_file}"
        
        with open(rc_diode_on_file, 'r') as f:
            circuit_data = json.load(f)
        
        parser = ParserJson()
        graph, control_graph = parser.parse(circuit_data)
        
        engine = Engine(graph, control_graph)
        engine.initialize()
        
        # Verify we have exactly one diode
        assert len(engine.diode_list) == 1, f"Expected 1 diode, found {len(engine.diode_list)}"
        diode = engine.diode_list[0]
        assert diode.comp_id == "D7", f"Expected diode D7, found {diode.comp_id}"
        
        # Set up state and input values for LCP computation
        # State: v_C3 = 0 (capacitor initially uncharged)
        n_states = len(engine.state_vars)
        assert n_states == 1, f"Expected 1 state variable (capacitor), found {n_states}"
        state_values = np.array([0.0])  # v_C3 = 0
        
        # Input: v_V9 = +5V (from circuit file)
        n_inputs = len(engine.input_vars)
        assert n_inputs == 1, f"Expected 1 input variable (voltage source), found {n_inputs}"
        input_values = np.array([5.0])  # v_V9 = +5V
        
        # Compute LCP matrices
        dae_model = engine.electrical_model
        try:
            M_matrix, q_vector = dae_model.compute_diode_lcp_matrices(state_values, input_values)
            
            # Validate matrix dimensions
            assert M_matrix.shape == (1, 1), f"Expected 1x1 M matrix, got {M_matrix.shape}"
            assert q_vector.shape == (1, 1), f"Expected 1x1 q vector, got {q_vector.shape}"
            
            # Convert to numerical values for comparison
            M_val = float(M_matrix[0, 0])
            q_val = float(q_vector[0, 0])
            
            print(f"RC-diode ON: M = {M_val:.6f}, q = {q_val:.6f}")
            
            # Validate expected LCP formulation
            assert abs(M_val - 2.0) < 1e-6, f"Expected M = 2.0 (resistance), got M = {M_val}"
            assert abs(q_val - (-5.0)) < 1e-6, f"Expected q = -5.0 (forward bias), got q = {q_val}"
            
        except Exception as e:
            pytest.fail(f"LCP matrix computation failed: {e}")
        
        # Test diode state detection
        try:
            diode_states = dae_model.detect_diode_states(state_values, input_values)
            assert len(diode_states) == 1, f"Expected 1 diode state, got {len(diode_states)}"
            
            is_conducting = diode_states[0]
            print(f"RC-diode ON: Diode D7 state = {'CONDUCTING' if is_conducting else 'BLOCKING'}")
            
            # Validate expected diode behavior
            assert is_conducting == True, "Diode should be conducting with +5V forward bias"
            
        except Exception as e:
            pytest.fail(f"Diode state detection failed: {e}")
    
    def test_rc_diode_off_lcp_formulation(self, rc_diode_off_file):
        """Test LCP formulation for RC-diode circuit with -5V (blocking case).
        
        Expected: M = 2.0 (resistance), q = +5.0 (reverse bias)
        Result: Diode should be blocking (False)
        """
        # Verify file exists
        assert os.path.exists(rc_diode_off_file), f"Test circuit file not found: {rc_diode_off_file}"
        
        with open(rc_diode_off_file, 'r') as f:
            circuit_data = json.load(f)
        
        parser = ParserJson()
        graph, control_graph = parser.parse(circuit_data)
        
        engine = Engine(graph, control_graph)
        engine.initialize()
        
        # Verify we have exactly one diode
        assert len(engine.diode_list) == 1, f"Expected 1 diode, found {len(engine.diode_list)}"
        diode = engine.diode_list[0]
        assert diode.comp_id == "D7", f"Expected diode D7, found {diode.comp_id}"
        
        # Set up state and input values for LCP computation
        # State: v_C3 = 0 (capacitor initially uncharged)
        n_states = len(engine.state_vars)
        assert n_states == 1, f"Expected 1 state variable (capacitor), found {n_states}"
        state_values = np.array([0.0])  # v_C3 = 0
        
        # Input: v_V9 = -5V (from circuit file)
        n_inputs = len(engine.input_vars)
        assert n_inputs == 1, f"Expected 1 input variable (voltage source), found {n_inputs}"
        input_values = np.array([-5.0])  # v_V9 = -5V
        
        # Compute LCP matrices
        dae_model = engine.electrical_model
        try:
            M_matrix, q_vector = dae_model.compute_diode_lcp_matrices(state_values, input_values)
            
            # Validate matrix dimensions
            assert M_matrix.shape == (1, 1), f"Expected 1x1 M matrix, got {M_matrix.shape}"
            assert q_vector.shape == (1, 1), f"Expected 1x1 q vector, got {q_vector.shape}"
            
            # Convert to numerical values for comparison
            M_val = float(M_matrix[0, 0])
            q_val = float(q_vector[0, 0])
            
            print(f"RC-diode OFF: M = {M_val:.6f}, q = {q_val:.6f}")
            
            # Validate expected LCP formulation
            assert abs(M_val - 2.0) < 1e-6, f"Expected M = 2.0 (resistance), got M = {M_val}"
            assert abs(q_val - 5.0) < 1e-6, f"Expected q = +5.0 (reverse bias), got q = {q_val}"
            
        except Exception as e:
            pytest.fail(f"LCP matrix computation failed: {e}")
        
        # Test diode state detection
        try:
            diode_states = dae_model.detect_diode_states(state_values, input_values)
            assert len(diode_states) == 1, f"Expected 1 diode state, got {len(diode_states)}"
            
            is_conducting = diode_states[0]
            print(f"RC-diode OFF: Diode D7 state = {'CONDUCTING' if is_conducting else 'BLOCKING'}")
            
            # Validate expected diode behavior
            assert is_conducting == False, "Diode should be blocking with -5V reverse bias"
            
        except Exception as e:
            pytest.fail(f"Diode state detection failed: {e}")
    
    def test_rc_diode_lcp_comparison(self, rc_diode_on_file, rc_diode_off_file):
        """Compare LCP formulations for both RC-diode circuits.
        
        Validates:
        - Same M matrix (identical circuit topology)
        - Opposite q vector polarity
        - Opposite diode conducting states
        """
        # Test both circuits
        circuits = [
            ("ON (+5V)", rc_diode_on_file, 5.0, -5.0, True),
            ("OFF (-5V)", rc_diode_off_file, -5.0, 5.0, False)
        ]
        
        results = []
        
        for name, circuit_file, input_val, expected_q, expected_conducting in circuits:
            assert os.path.exists(circuit_file), f"Test circuit file not found: {circuit_file}"
            
            with open(circuit_file, 'r') as f:
                circuit_data = json.load(f)
            
            parser = ParserJson()
            graph, control_graph = parser.parse(circuit_data)
            
            engine = Engine(graph, control_graph)
            engine.initialize()
            
            # Set up test conditions
            state_values = np.array([0.0])  # v_C3 = 0
            input_values = np.array([input_val])
            
            # Compute LCP matrices
            dae_model = engine.electrical_model
            M_matrix, q_vector = dae_model.compute_diode_lcp_matrices(state_values, input_values)
            
            M_val = float(M_matrix[0, 0])
            q_val = float(q_vector[0, 0])
            
            # Detect diode state
            diode_states = dae_model.detect_diode_states(state_values, input_values)
            is_conducting = diode_states[0]
            
            results.append((name, M_val, q_val, is_conducting))
            
            # Validate individual circuit expectations
            assert abs(M_val - 2.0) < 1e-6, f"{name}: Expected M = 2.0, got M = {M_val}"
            assert abs(q_val - expected_q) < 1e-6, f"{name}: Expected q = {expected_q}, got q = {q_val}"
            assert is_conducting == expected_conducting, \
                f"{name}: Expected conducting = {expected_conducting}, got {is_conducting}"
        
        # Compare results between circuits
        on_result = results[0]  # (name, M, q, conducting)
        off_result = results[1]
        
        print(f"LCP Comparison:")
        print(f"  {on_result[0]}: M = {on_result[1]:.6f}, q = {on_result[2]:.6f}, conducting = {on_result[3]}")
        print(f"  {off_result[0]}: M = {off_result[1]:.6f}, q = {off_result[2]:.6f}, conducting = {off_result[3]}")
        
        # Validate comparison expectations
        assert abs(on_result[1] - off_result[1]) < 1e-6, "M matrices should be identical (same topology)"
        assert abs(on_result[2] - (-off_result[2])) < 1e-6, "q vectors should have opposite polarity"
        assert on_result[3] != off_result[3], "Diode states should be opposite"
        
        print("SUCCESS: All LCP formulation validations passed!")


def test_rectifier_integration_standalone():
    """Standalone test that can be run independently."""
    test_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            'test_data', 'diodes_rect4.json')
    
    if not os.path.exists(test_file):
        pytest.skip(f"Test file not found: {test_file}")
    
    try:
        result = run_simulation_from_file(
            test_file,
            t_span=(0, 1),
            plot_results=False
        )
        
        assert result['success'], "Rectifier simulation should succeed"
        assert len(result['t']) > 50, "Should have reasonable time resolution"
        
        print(f"SUCCESS: 4-diode rectifier simulation completed with {len(result['t'])} time points")
        return True
        
    except Exception as e:
        pytest.fail(f"Standalone rectifier test failed: {e}")


if __name__ == "__main__":
    # Allow running this test file directly
    test_rectifier_integration_standalone()