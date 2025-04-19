import pytest
import json
import os
from amps_simulation.simulation import run_simulation

# Test data configuration
TEST_DATA_DIR = "test_data"
TEST_FILES = [
    "test.json",          # Basic RC circuit
    "test_rc.json",       # RC circuit
    "test_rrc.json",      # RRC circuit
    "test_rlc.json",      # RLC circuit
    "test_lcr.json",      # LCR circuit
    "test_rrc_gnd.json",  # RRC with ground
    "test_rcrc.json",     # RC-RC circuit
    "test_rcrc_sn.json",  # RC-RC with serial node
    "test_rcl.json",      # RCL circuit
    "test_2v.json",       # Circuit with 2 voltage sources
    "test_2v2.json",      # Another 2 voltage source circuit
    "test_2v3.json",      # Complex 2 voltage source circuit
    "test_2v_series.json", # Series circuit with 2 voltage sources
    "test_2v_series2.json", # Another series circuit with 2 voltage sources
    "test_nonplanar.json"  # Non-planar circuit
]

def load_test_file(filename):
    """Load a test circuit file and return its contents."""
    file_path = os.path.join(TEST_DATA_DIR, filename)
    with open(file_path, 'r') as f:
        return json.load(f)

@pytest.mark.parametrize("test_file", TEST_FILES)
def test_circuit_simulation(test_file):
    """Test that each circuit simulation runs without errors."""
    try:
        circuit_data = load_test_file(test_file)
        run_simulation(circuit_data, test_mode=True)
    except Exception as e:
        pytest.fail(f"Simulation failed for {test_file} with error: {str(e)}") 