import pytest
import numpy as np
from amps_simulation.core.lcp import LCP


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
        w, z, info = lcp.solve()

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

        # Check info dict
        assert info['converged'] is True
        assert info['termination_reason'] == 'success'
        assert info['complementarity'] < 1e-7

    def test_lcp_solve_with_warm_start(self, synthetic_lcp_data):
        """Test LCP solve method with warm start."""
        M, q, z_expected, w_expected = synthetic_lcp_data

        # Create a warm start hint (boolean array indicating which variables should be active)
        z_init = np.array([True, False, True])

        lcp = LCP(M, q, z_init)
        w, z, info = lcp.solve()

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

        # Check info dict
        assert info['converged'] is True
        assert info['warm_start_used'] is True
        assert info['termination_reason'] == 'success'
        assert info['complementarity'] < 1e-7

    def test_lcp_solve_returns_expected_solution(self, synthetic_lcp_data):
        """Test that LCP solver returns the expected solution for the synthetic problem."""
        M, q, z_expected, w_expected = synthetic_lcp_data
        lcp = LCP(M, q)
        w, z, info = lcp.solve()

        # The solution should match the known solution
        assert np.allclose(z, z_expected, atol=1e-6), f"Expected z={z_expected}, got z={z}"
        assert np.allclose(w, w_expected, atol=1e-6), f"Expected w={w_expected}, got w={w}"

        # Check info dict
        assert info['converged'] is True
        assert info['termination_reason'] == 'success'

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

    def test_lcp_info_dict_structure(self, synthetic_lcp_data):
        """Test that info dict has all required keys."""
        M, q, _, _ = synthetic_lcp_data
        lcp = LCP(M, q)
        w, z, info = lcp.solve()

        # Check that all expected keys are present
        assert 'converged' in info
        assert 'pivots' in info
        assert 'complementarity' in info
        assert 'termination_reason' in info
        assert 'warm_start_used' in info
        assert 'warm_start_feasible' in info

    def test_lcp_info_trivial_solution(self):
        """Test info dict for trivial solution (q >= 0)."""
        M = np.array([[1., 0.], [0., 1.]])
        q = np.array([1., 2.])  # Non-negative, so z=0 is solution
        lcp = LCP(M, q)
        w, z, info = lcp.solve()

        # Should have trivial solution
        assert np.allclose(z, 0)
        assert np.allclose(w, q)
        assert info['converged'] is True
        assert info['termination_reason'] == 'trivial_solution'
        assert info['pivots'] == 0
        assert info['complementarity'] == 0.0

    def test_lcp_info_pivot_count(self, synthetic_lcp_data):
        """Test that pivot count is tracked correctly."""
        M, q, _, _ = synthetic_lcp_data
        lcp = LCP(M, q)
        w, z, info = lcp.solve()

        # Should have performed at least one pivot for this non-trivial problem
        assert info['pivots'] > 0
        assert isinstance(info['pivots'], int)

    def test_lcp_info_complementarity(self, synthetic_lcp_data):
        """Test that complementarity is computed correctly."""
        M, q, _, _ = synthetic_lcp_data
        lcp = LCP(M, q)
        w, z, info = lcp.solve()

        # Complementarity should match w^T * z
        expected_complementarity = np.dot(w, z)
        assert np.isclose(info['complementarity'], expected_complementarity)
        # Should be very small (near zero) for valid LCP solution
        assert info['complementarity'] < 1e-7

    def test_lcp_info_warm_start_tracking(self, synthetic_lcp_data):
        """Test that warm start usage is tracked correctly."""
        M, q, _, _ = synthetic_lcp_data

        # Test without warm start
        lcp_no_warm = LCP(M, q)
        _, _, info_no_warm = lcp_no_warm.solve()
        assert info_no_warm['warm_start_used'] is False

        # Test with warm start (all zeros = standard start, not really "used")
        lcp_warm_zeros = LCP(M, q, z_init=np.array([False, False, False]))
        _, _, info_warm_zeros = lcp_warm_zeros.solve()
        # May or may not be marked as used depending on implementation

        # Test with non-trivial warm start
        lcp_warm = LCP(M, q, z_init=np.array([True, False, False]))
        _, _, info_warm = lcp_warm.solve()
        assert info_warm['warm_start_used'] is True

    def test_lcp_info_termination_reasons(self):
        """Test different termination reasons."""
        # Test success termination
        M = np.array([[1., 0.], [0., 1.]])
        q = np.array([-1., -1.])
        lcp = LCP(M, q)
        w, z, info = lcp.solve()
        assert info['termination_reason'] in ['success', 'trivial_solution']

        # Test max iterations (with very small limit)
        M_hard = np.array([[2., -1.], [-1., 2.]])
        q_hard = np.array([-5., -5.])
        lcp_hard = LCP(M_hard, q_hard, max_iter=1)
        with pytest.raises(RuntimeError, match="Maximum iterations"):
            lcp_hard.solve()
