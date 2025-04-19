import pytest
import json
import os
import numpy as np
from amps_simulation.core.state_space_model import extract_differential_equations
from sympy import Matrix, Float, zoo, oo, nan

# Test data configuration
TEST_DATA_DIR = "test_data"
TEST_FILES = [
    "test_rc.json",       # RC circuit
    "test_rlc.json",      # RLC circuit
    "test_2v.json",       # Circuit with 2 voltage sources
]

def load_test_file(filename):
    """Load a test circuit file and return its contents."""
    file_path = os.path.join(TEST_DATA_DIR, filename)
    with open(file_path, 'r') as f:
        return json.load(f)

def check_matrix_values(matrix, test_file):
    """Check if a SymPy matrix contains any invalid values."""
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            value = matrix[i, j]
            # Check for SymPy's infinity and NaN
            assert value != zoo, f"Matrix contains complex infinity at [{i}, {j}] for {test_file}"
            assert value != oo, f"Matrix contains infinity at [{i}, {j}] for {test_file}"
            assert value != -oo, f"Matrix contains negative infinity at [{i}, {j}] for {test_file}"
            assert value != nan, f"Matrix contains NaN at [{i}, {j}] for {test_file}"
            # Check if value can be converted to float
            try:
                float(value)
            except (TypeError, ValueError):
                pytest.fail(f"Matrix contains non-numeric value {value} at [{i}, {j}] for {test_file}")

@pytest.mark.parametrize("test_file", TEST_FILES)
def test_extract_differential_equations(test_file):
    """Test that extract_differential_equations correctly processes circuit files."""
    try:
        # Load circuit data
        circuit_data = load_test_file(test_file)
        
        # Extract differential equations
        A, B, state_vars, input_vars = extract_differential_equations(circuit_data)
        
        # Basic validation checks
        assert isinstance(A, Matrix), f"A should be a SymPy Matrix for {test_file}"
        assert isinstance(B, Matrix), f"B should be a SymPy Matrix for {test_file}"
        assert isinstance(state_vars, dict), f"state_vars should be a dictionary for {test_file}"
        assert isinstance(input_vars, dict), f"input_vars should be a dictionary for {test_file}"
        
        # Check dimensions
        assert A.rows == A.cols, f"A should be square for {test_file}"
        assert A.rows == len(state_vars), f"A dimensions should match state_vars count for {test_file}"
        assert B.cols == len(input_vars), f"B columns should match input_vars count for {test_file}"
        
        # Check for invalid values
        check_matrix_values(A, test_file)
        check_matrix_values(B, test_file)
        
    except Exception as e:
        pytest.fail(f"extract_differential_equations failed for {test_file} with error: {str(e)}")

def test_extract_differential_equations_specific_circuit():
    """Test extract_differential_equations with a specific circuit (RC) for detailed validation."""
    # Load RC circuit
    circuit_data = load_test_file("test_rc.json")
    
    # Extract differential equations
    A, B, state_vars, input_vars = extract_differential_equations(circuit_data)
    
    # For RC circuit, we expect:
    # - One state variable (capacitor voltage)
    # - One input variable (voltage source)
    assert len(state_vars) == 1, "RC circuit should have one state variable"
    assert len(input_vars) == 1, "RC circuit should have one input variable"
    
    # A should be 1x1 for RC circuit
    assert A.rows == 1 and A.cols == 1, "A should be 1x1 for RC circuit"
    assert B.rows == 1 and B.cols == 1, "B should be 1x1 for RC circuit"
    
    # For RC circuit, A should be negative (discharging)
    assert float(A[0, 0]) < 0, "A[0,0] should be negative for RC circuit (discharging)" 