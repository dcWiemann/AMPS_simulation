#!/usr/bin/env python3
"""
Test script for the integrated LCP solver.
"""

import numpy as np
import sys
import os

# Add the project to Python path
sys.path.insert(0, os.path.dirname(__file__))

def test_lcp_solver_basic():
    """Test basic LCP solver functionality."""
    print("=== Test 1: Basic LCP Solver ===")
    
    try:
        from amps_simulation.core.lcp_solver import LCPSolver
        
        # Create solver
        solver = LCPSolver(tolerance=1e-10)
        
        # Simple 2x2 test case
        M = np.array([[2.0, -1.0],
                      [-1.0, 2.0]])
        q = np.array([-1.0, -1.0])
        
        print(f"M = \n{M}")
        print(f"q = {q}")
        
        # Solve LCP
        active, info = solver.solve(M, q)
        
        print(f"Active set: {active}")
        print(f"Converged: {info['converged']}")
        print(f"Pivots: {info['pivots']}")
        print(f"Complementarity: {info['complementarity']:.2e}")
        
        # Verify solution
        verification = solver.verify_solution(M, q, active)
        print(f"Solution feasible: {verification['feasible']}")
        print(f"z = {verification['z']}")
        print(f"w = {verification['w']}")
        
        success = info['converged'] and verification['feasible']
        print(f"Test 1: {'PASSED' if success else 'FAILED'}")
        return success
        
    except Exception as e:
        print(f"Test 1 FAILED: {e}")
        return False

def test_diode_lcp_solver():
    """Test diode-specific LCP solver."""
    print("\n=== Test 2: Diode LCP Solver ===")
    
    try:
        from amps_simulation.core.lcp_solver import DiodeLCPSolver
        
        # Create diode solver
        solver = DiodeLCPSolver(tolerance=1e-10)
        
        # Single diode test case
        M = np.array([[1.0]])  # 1 ohm resistance
        q = np.array([1.0])    # Positive = reverse bias
        diode_names = ["D1"]
        
        print(f"M = {M}")
        print(f"q = {q}")
        
        # Detect diode states
        conducting_states, info = solver.detect_diode_states(M, q, diode_names)
        
        print(f"Diode conducting states: {conducting_states}")
        print(f"Converged: {info['converged']}")
        print(f"Pivots: {info['pivots']}")
        
        # For q > 0, diode should be blocking (False)
        expected = [False]
        success = conducting_states == expected and info['converged']
        print(f"Test 2: {'PASSED' if success else 'FAILED'}")
        return success
        
    except Exception as e:
        print(f"Test 2 FAILED: {e}")
        return False

def test_dae_model_integration():
    """Test LCP solver integration with DAE model."""
    print("\n=== Test 3: DAE Model Integration ===")
    
    try:
        from amps_simulation.core.dae_model import ElectricalDaeModel
        from amps_simulation.core.electrical_graph import ElectricalGraph
        import networkx as nx
        
        # Create minimal graph
        graph = nx.MultiDiGraph()
        graph.add_node(0, junction=type('obj', (object,), {'is_ground': True})())
        graph.add_node(1, junction=type('obj', (object,), {'is_ground': False})())
        
        # Create electrical graph
        electrical_graph = ElectricalGraph(graph)
        
        # Create DAE model
        dae_model = ElectricalDaeModel(electrical_graph)
        
        # Check that LCP solver is initialized
        assert hasattr(dae_model, 'lcp_solver'), "LCP solver not initialized"
        assert dae_model.lcp_solver is not None, "LCP solver is None"
        
        print("DAE model successfully created with LCP solver")
        print(f"LCP solver type: {type(dae_model.lcp_solver).__name__}")
        
        success = True
        print(f"Test 3: {'PASSED' if success else 'FAILED'}")
        return success
        
    except Exception as e:
        print(f"Test 3 FAILED: {e}")
        return False

def test_edge_cases():
    """Test edge cases."""
    print("\n=== Test 4: Edge Cases ===")
    
    try:
        from amps_simulation.core.lcp_solver import LCPSolver
        
        solver = LCPSolver()
        success = True
        
        # Test empty problem
        M_empty = np.array([]).reshape(0, 0)
        q_empty = np.array([])
        active, info = solver.solve(M_empty, q_empty)
        
        if len(active) == 0 and info['converged']:
            print("Empty problem handled correctly")
        else:
            print("Failed on empty problem")
            success = False
        
        # Test dimension mismatch
        try:
            M_bad = np.array([[1.0, 2.0]])  # 1x2 matrix
            q_bad = np.array([1.0])         # 1x1 vector
            solver.solve(M_bad, q_bad)
            print("Should have failed on dimension mismatch")
            success = False
        except ValueError:
            print("Correctly caught dimension mismatch")
        
        print(f"Test 4: {'PASSED' if success else 'FAILED'}")
        return success
        
    except Exception as e:
        print(f"Test 4 FAILED: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing integrated LCP solver...")
    
    tests = [
        test_lcp_solver_basic,
        test_diode_lcp_solver,
        test_dae_model_integration,
        test_edge_cases,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n=== Summary ===")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("SUCCESS: All tests passed!")
        return 0
    else:
        print("FAILED: Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())