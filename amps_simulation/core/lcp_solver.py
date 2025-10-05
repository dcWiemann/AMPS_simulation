"""
Linear Complementarity Problem (LCP) solver for diode state detection.

This module provides an LCP solver specifically designed for determining
diode conducting states in electrical circuit simulation.
"""

import numpy as np
from scipy.sparse import isspmatrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Tuple, Dict, Optional, Union
import logging


class LCPSolver:
    """
    Linear Complementarity Problem solver using simple principal pivoting.
    
    Solves LCP: find z >= 0, w = q + M*z >= 0, z^T*w = 0
    
    This solver is specifically designed for diode state detection where:
    - z represents diode currents (>= 0)
    - w represents negative diode voltages (>= 0, i.e., v_D <= 0)
    - Complementarity: either z > 0 (conducting) or w > 0 (blocking), but not both
    """
    
    def __init__(self, tolerance: float = 1e-10, max_pivots: Optional[int] = None):
        """
        Initialize the LCP solver.
        
        Args:
            tolerance: Nonnegativity tolerance for violations
            max_pivots: Maximum number of pivots allowed (default: 10*n)
        """
        self.tolerance = tolerance
        self.max_pivots = max_pivots
        self.logger = logging.getLogger(__name__)
    
    def solve(self, M: Union[np.ndarray, csr_matrix], q: np.ndarray, 
              initial_active: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Solve the Linear Complementarity Problem.
        
        Args:
            M: (n,n) LCP matrix (can be dense numpy array or sparse)
            q: (n,) LCP right-hand side vector
            initial_active: (n,) boolean array for warm start
                          True = variable is active (z > 0, w = 0)
                          False = variable is inactive (z = 0, w > 0)
                          If None, uses heuristic: active where q < 0
        
        Returns:
            Tuple containing:
            - active_set: (n,) boolean array where True means z > 0 (conducting)
            - info: Dict with solver diagnostics
        """
        # Input validation and normalization
        q = np.asarray(q, dtype=float).ravel()
        n = q.size
        
        if isspmatrix(M):
            M = M.tocsr()
        else:
            M = csr_matrix(np.asarray(M, dtype=float))
        
        if M.shape != (n, n):
            raise ValueError(f"M must be square {n}x{n} matrix, got {M.shape}")
        
        # Initialize active set
        if initial_active is None:
            # Heuristic: start with variables active where q < 0
            active = q < 0.0
        else:
            active = np.asarray(initial_active, dtype=bool).copy()
            if active.size != n:
                raise ValueError(f"initial_active must have size {n}, got {active.size}")
        
        max_pivots = self.max_pivots if self.max_pivots is not None else max(10 * n, 1)
        
        # Handle empty problem case
        if n == 0:
            return np.array([], dtype=bool), {
                "converged": True,
                "pivots": 0,
                "last_violation": None,
                "final_z": np.array([]),
                "final_w": np.array([]),
                "complementarity": 0.0
            }
        
        # Solver state
        pivots = 0
        last_violation = None
        converged = False
        
        # Working vectors
        z = np.zeros(n, dtype=float)
        w = np.zeros(n, dtype=float)
        
        self.logger.debug(f"Starting LCP solver: n={n}, max_pivots={max_pivots}")
        self.logger.debug(f"Initial active set: {np.sum(active)}/{n} active")
        
        while pivots < max_pivots:
            # Solve for z values where active
            if np.any(active):
                try:
                    # Extract submatrix and solve M_AA * z_A = -q_A
                    M_AA = M[active][:, active]
                    rhs = -q[active].copy()
                    z_active = spsolve(M_AA, rhs)
                    
                    # Update z vector
                    z.fill(0.0)
                    z[active] = z_active
                    
                except Exception as e:
                    self.logger.error(f"Linear solve failed at pivot {pivots}: {e}")
                    break
            else:
                z.fill(0.0)
            
            # Compute w = q + M*z
            w[:] = q
            if np.any(active):
                w += M @ z
            
            # Check for violations
            z_violations = np.zeros(n, dtype=bool)
            w_violations = np.zeros(n, dtype=bool)
            
            if np.any(active):
                z_violations[active] = z[active] < -self.tolerance
            
            inactive = ~active
            if np.any(inactive):
                w_violations[inactive] = w[inactive] < -self.tolerance
            
            # Check convergence
            if not (np.any(z_violations) or np.any(w_violations)):
                converged = True
                last_violation = None
                break
            
            # Choose pivot: prioritize z violations (numerical stability)
            pivot_idx = None
            if np.any(z_violations):
                pivot_idx = np.nonzero(z_violations)[0][0]
                active[pivot_idx] = False  # Remove from active set
                last_violation = ("z", pivot_idx)
                self.logger.debug(f"Pivot {pivots}: removing z[{pivot_idx}] from active set")
            elif np.any(w_violations):
                pivot_idx = np.nonzero(w_violations)[0][0]
                active[pivot_idx] = True   # Add to active set
                last_violation = ("w", pivot_idx)
                self.logger.debug(f"Pivot {pivots}: adding z[{pivot_idx}] to active set")
            else:
                break  # No violations found
            
            pivots += 1
        
        # Prepare results
        info = {
            "converged": converged,
            "pivots": pivots,
            "last_violation": last_violation,
            "final_z": z.copy(),
            "final_w": w.copy(),
            "complementarity": np.dot(z, w)
        }
        
        if converged:
            verification = self.verify_solution(M, q, active)
            info.update(verification)
            self.logger.debug(f"LCP converged in {pivots} pivots")
            self.logger.debug(f"Final active set: {np.sum(active)}/{n} active")
            self.logger.debug(f"Complementarity: {info['complementarity']:.2e}")
        else:
            self.logger.warning(f"LCP did not converge after {pivots} pivots")
            if last_violation:
                self.logger.warning(f"Last violation: {last_violation}")
        
        return active, info
    
    def verify_solution(self, M: Union[np.ndarray, csr_matrix], q: np.ndarray, 
                       active: np.ndarray) -> Dict:
        """
        Verify that a given active set satisfies the LCP conditions.
        
        Args:
            M: LCP matrix
            q: LCP vector
            active: Boolean array indicating active variables
            
        Returns:
            Dict with verification results
        """
        q = np.asarray(q, dtype=float).ravel()
        active = np.asarray(active, dtype=bool)
        n = q.size
        
        if isspmatrix(M):
            M = M.tocsr()
        else:
            M = csr_matrix(np.asarray(M, dtype=float))
        
        # Compute solution
        z = np.zeros(n, dtype=float)
        if np.any(active):
            M_AA = M[active][:, active]
            z_active = spsolve(M_AA, -q[active])
            z[active] = z_active
        
        w = q + M @ z
        
        # Check feasibility conditions
        z_feasible = np.all(z >= -self.tolerance)
        w_feasible = np.all(w >= -self.tolerance)
        complementarity = abs(np.dot(z, w))
        comp_satisfied = complementarity < self.tolerance
        
        return {
            "z_feasible": z_feasible,
            "w_feasible": w_feasible,
            "complementarity_satisfied": comp_satisfied,
            "complementarity_value": complementarity,
            "z": z,
            "w": w,
            "feasible": z_feasible and w_feasible and comp_satisfied
        }


class DiodeLCPSolver(LCPSolver):
    """
    Specialized LCP solver for diode state detection in circuit simulation.
    
    This class provides a domain-specific interface for determining whether
    diodes are conducting or blocking based on circuit conditions.
    """
    
    def detect_diode_states(self, M: Union[np.ndarray, csr_matrix], q: np.ndarray,
                           initial_active: Optional[np.ndarray] = None, 
                           diode_names: Optional[list] = None) -> Tuple[list, Dict]:
        """
        Detect diode conducting states using LCP formulation.
        
        Args:
            M: Impedance matrix relating diode currents to voltages
            q: Offset vector (voltages when diode currents = 0)
            diode_names: Optional list of diode names for logging
            
        Returns:
            Tuple containing:
            - conducting_states: List of boolean values (True = conducting)
            - info: Solver diagnostics
        """
        active_set, info = self.solve(M, q, initial_active=initial_active)
        conducting_states = active_set.tolist()
        
        # Enhanced logging for diode context
        if diode_names and len(diode_names) == len(conducting_states):
            for i, (name, conducting) in enumerate(zip(diode_names, conducting_states)):
                state_str = "conducting" if conducting else "blocking"
                self.logger.debug(f"Diode {name}: {state_str}")
        
        return conducting_states, info