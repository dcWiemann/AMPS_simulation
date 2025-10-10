import pytest
import numpy as np
from amps_simulation.core.lcp_solver import LCP


class TestLCPSolver:
    """Test suite for the LCP solver class."""

    @pytest.fixture
    def synthetic_lcp_data(self):
        """Provide synthetic LCP test data."""
        M = np.array([[3., -1., 0.],
                      [2.,  4., 1.],
                      [1., -2., 2.]])
        q = np.array([-3., -3., -7.])
        # Known solution
        z = np.array([1., 0., 3.])
        w = M @ z + q
        return M, q, z, w

    def test_lcp_initialization_without_warm_start(self, synthetic_lcp_data):
        """Test LCP initialization without warm start."""
        M, q, _, _ = synthetic_lcp_data
        lcp = LCP(M, q)
        assert lcp is not None
        assert np.allclose(lcp.M, M)
        assert np.allclose(lcp.q, q)

    def test_lcp_initialization_with_warm_start(self, synthetic_lcp_data):
        """Test LCP initialization with warm start."""
        M, q, _, _ = synthetic_lcp_data
        z_init = np.array([True, False, True])
        lcp = LCP(M, q, z_init)
        assert lcp is not None
        assert np.allclose(lcp.M, M)
        assert np.allclose(lcp.q, q)
        assert np.allclose(lcp.z_init, z_init)

    def test_lcp_solve_without_warm_start(self, synthetic_lcp_data):
        """Test LCP solve method without warm start."""
        M, q, z_expected, w_expected = synthetic_lcp_data
        lcp = LCP(M, q)
        w, z = lcp.solve()

        # Check solution dimensions
        assert len(w) == len(q)
        assert len(z) == len(q)

        # Check that w = M @ z + q
        assert np.allclose(w, M @ z + q)

        # Check LCP feasibility conditions
        assert np.all(w >= -1e-10), "w should be non-negative"
        assert np.all(z >= -1e-10), "z should be non-negative"

        # Check complementarity condition: w_i * z_i = 0 for all i
        assert np.allclose(w * z, 0, atol=1e-8), "Complementarity condition violated"

    def test_lcp_solve_with_warm_start(self, synthetic_lcp_data):
        """Test LCP solve method with warm start."""
        M, q, z_expected, w_expected = synthetic_lcp_data

        # Create a warm start hint (boolean array indicating which variables should be active)
        z_init = np.array([True, False, True])

        lcp = LCP(M, q, z_init)
        w, z = lcp.solve()

        # Check solution dimensions
        assert len(w) == len(q)
        assert len(z) == len(q)

        # Check that w = M @ z + q
        assert np.allclose(w, M @ z + q)

        # Check LCP feasibility conditions
        assert np.all(w >= -1e-10), "w should be non-negative"
        assert np.all(z >= -1e-10), "z should be non-negative"

        # Check complementarity condition: w_i * z_i = 0 for all i
        assert np.allclose(w * z, 0, atol=1e-8), "Complementarity condition violated"

    def test_lcp_solve_returns_expected_solution(self, synthetic_lcp_data):
        """Test that LCP solver returns the expected solution for the synthetic problem."""
        M, q, z_expected, w_expected = synthetic_lcp_data
        lcp = LCP(M, q)
        w, z = lcp.solve()

        # The solution should match the known solution
        assert np.allclose(z, z_expected, atol=1e-6), f"Expected z={z_expected}, got z={z}"
        assert np.allclose(w, w_expected, atol=1e-6), f"Expected w={w_expected}, got w={w}"

    def test_lcp_invalid_dimensions(self):
        """Test that LCP raises error for mismatched dimensions."""
        M = np.array([[1., 2.], [3., 4.]])
        q = np.array([1., 2., 3.])  # Wrong dimension

        with pytest.raises((ValueError, AssertionError)):
            lcp = LCP(M, q)

    def test_lcp_non_square_matrix(self):
        """Test that LCP raises error for non-square matrix."""
        M = np.array([[1., 2., 3.], [4., 5., 6.]])  # Non-square
        q = np.array([1., 2.])

        with pytest.raises((ValueError, AssertionError)):
            lcp = LCP(M, q)

    def test_lcp_warm_start_wrong_dimension(self, synthetic_lcp_data):
        """Test that LCP raises error when warm start has wrong dimension."""
        M, q, _, _ = synthetic_lcp_data
        z_init = np.array([True, False])  # Wrong dimension (should be 3)

        with pytest.raises((ValueError, AssertionError)):
            lcp = LCP(M, q, z_init)
