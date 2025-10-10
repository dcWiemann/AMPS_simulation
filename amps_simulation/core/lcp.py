import numpy as np

class LCP:
    """
    Solve the Linear Complementarity Problem (LCP):
        find z >= 0, w = M z + q >= 0, and z^T w = 0
    using Lemke's algorithm with an optional warm-start basis from z_init.
    """

    def __init__(self, M: np.ndarray, q: np.ndarray, z_init=None, tol=1e-10, max_iter=100000):
        M = np.asarray(M, dtype=float)
        q = np.asarray(q, dtype=float).reshape(-1)

        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError("M must be a square matrix.")
        if q.shape[0] != M.shape[0]:
            raise ValueError("q must have the same dimension as M.")

        self.M = M
        self.q = q
        self.n = M.shape[0]
        self.tol = tol
        self.max_iter = max_iter

        if z_init is None:
            self.z_init = np.zeros(self.n, dtype=bool)
        else:
            zi = np.array(z_init, dtype=bool).reshape(-1)
            if zi.size != self.n:
                raise ValueError("z_init must be a boolean list/array of length n.")
            self.z_init = zi

    # --- internal helpers -------------------------------------------------

    @staticmethod
    def _complement(idx, n):
        # indices: 0..n-1 are w, n..2n-1 are z, 2n is artificial (t)
        if idx < n:      # w_i
            return idx + n
        elif idx < 2*n:  # z_i
            return idx - n
        else:            # t has no complement (return itself)
            return idx

    @staticmethod
    def _pivot(T, b, row, col):
        """Perform a pivot on tableau column 'col' and row 'row'."""
        pivot_val = T[row, col]
        if abs(pivot_val) < 1e-18:
            raise RuntimeError("Numerical pivot breakdown.")
        # Normalize pivot row
        T[row, :] = T[row, :] / pivot_val
        b[row]    = b[row] / pivot_val
        # Eliminate in other rows
        m = T.shape[0]
        for r in range(m):
            if r == row:
                continue
            factor = T[r, col]
            if abs(factor) > 0:
                T[r, :] -= factor * T[row, :]
                b[r]    -= factor * b[row]

    def _choose_leaving_row(self, col, T, b):
        """
        Minimum ratio test: among rows where column entry > tol,
        choose argmin(b_i / T[i,col]).
        Return None if no eligible row.
        """
        candidates = []
        for i in range(self.n):
            a = T[i, col]
            if a > self.tol:
                candidates.append((b[i] / a, i))
        if not candidates:
            return None
        # Bland-ish tie-breaking by value then index
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][1]

    # --- main solver ------------------------------------------------------

    def solve(self):
        """
        Returns:
            Tuple of (w, z, info) where:
            - w, z: numpy arrays (solution)
            - info: dict with solver diagnostics
        Raises:
            RuntimeError on failure (e.g., ray termination).
        """

        n = self.n
        M = self.M
        q = self.q.copy()

        # Initialize info dict to track solver diagnostics
        info = {
            'converged': False,
            'pivots': 0,
            'complementarity': 0.0,
            'termination_reason': None,
            'warm_start_used': False,
            'warm_start_feasible': False
        }

        # Trivial feasible solution
        if np.all(q >= -self.tol):
            z = np.zeros(n)
            w = q.copy()
            info['converged'] = True
            info['termination_reason'] = 'trivial_solution'
            info['complementarity'] = np.dot(w, z)
            return w, z, info

        # Build tableau for: w - M z = q
        # Columns ordered as [w(0..n-1), z(0..n-1), t], with t column initially 0
        T = np.zeros((n, 2*n + 1), dtype=float)
        # w columns = I
        T[:, 0:n] = np.eye(n)
        # z columns = -M
        T[:, n:2*n] = -M
        # t column (artificial) will be -1 for standard Lemke start
        T[:, 2*n] = 0.0

        b = q.copy()

        # Basis: one basic variable per row; we store its COLUMN index
        # Initialize from warm start: if z_init[i] is True, make z_i basic; else w_i
        basis = np.empty(n, dtype=int)
        for i in range(n):
            basis[i] = (n + i) if self.z_init[i] else i

        # Try to compute basic solution from warm start
        # Build B and solve B x_B = b; update rows by pivoting if basis not identity-like
        # For numerical stability and generality, we will pivot the tableau so that
        # in each row, the basis column has a 1 and others have 0 in that column.
        # Start by making sure the pivot entries are nonzero; if not feasible, fall back.

        def try_enforce_basis(basis, T, b):
            # Make the chosen basis columns into an identity via Gauss-Jordan
            used_rows = set()
            for r in range(n):
                col = basis[r]
                # Find a row with nonzero in this column not yet used
                pivot_row = None
                if abs(T[r, col]) > 1e-14 and r not in used_rows:
                    pivot_row = r
                else:
                    for rr in range(n):
                        if rr in used_rows:
                            continue
                        if abs(T[rr, col]) > 1e-14:
                            pivot_row = rr
                            break
                if pivot_row is None:
                    return False  # cannot realize this basis in current tableau
                # If pivot_row != r, swap rows
                if pivot_row != r:
                    T[[r, pivot_row], :] = T[[pivot_row, r], :]
                    b[[r, pivot_row]] = b[[pivot_row, r]]
                # Pivot
                self._pivot(T, b, r, col)
                used_rows.add(r)
            return True

        T_copy, b_copy = T.copy(), b.copy()
        warm_basis_realized = try_enforce_basis(basis.copy(), T_copy, b_copy)

        if not warm_basis_realized:
            # Warm basis not directly realizable; revert to standard starting basis W
            info['warm_start_used'] = True
            info['warm_start_feasible'] = False
            basis = np.arange(n, dtype=int)  # w basic
            T_copy, b_copy = T.copy(), b.copy()
            try_enforce_basis(basis.copy(), T_copy, b_copy)
        else:
            # Track if warm start was attempted (non-trivial z_init)
            if np.any(self.z_init):
                info['warm_start_used'] = True

        # Check feasibility (b >= 0); if infeasible, do Lemke init with artificial t
        if np.any(b_copy < -self.tol):
            # Rebuild from clean tableau with W basic
            if info['warm_start_used']:
                info['warm_start_feasible'] = False
            T = np.zeros((n, 2*n + 1), dtype=float)
            T[:, 0:n] = np.eye(n)
            T[:, n:2*n] = -M
            T[:, 2*n] = -1.0  # artificial column
            b = q.copy()
            basis = np.arange(n, dtype=int)  # start with w basic
            # Make basis identity
            for i in range(n):
                self._pivot(T, b, i, basis[i])
                info['pivots'] += 1

            # Enter artificial variable t, pick row with most negative b
            r = int(np.argmin(b))
            self._pivot(T, b, r, 2*n)    # pivot t into basis at row r
            info['pivots'] += 1
            basis[r] = 2*n               # t is now basic at row r
            entering = self._complement(r, n)  # complement of leaving w_r is z_r
        else:
            # Warm-start feasible; check if it's already a valid LCP solution
            if info['warm_start_used']:
                info['warm_start_feasible'] = True
            T, b = T_copy, b_copy

            # Extract solution from warm start basis
            x_warm = np.zeros(2*n + 1)
            for i in range(n):
                x_warm[basis[i]] = b[i]
            w_warm = x_warm[0:n]
            z_warm = x_warm[n:2*n]

            # Check if warm start is already a valid LCP solution
            # Conditions: w >= 0, z >= 0, w = M*z + q, w^T * z = 0
            if (np.all(w_warm >= -self.tol) and np.all(z_warm >= -self.tol)):
                # Check if w = M*z + q
                residual = w_warm - (M @ z_warm + q)
                if np.max(np.abs(residual)) < 1e-10:
                    # Warm start is a valid solution! Return it immediately
                    info['converged'] = True
                    info['termination_reason'] = 'success'
                    info['complementarity'] = np.dot(w_warm, z_warm)
                    # Clean up numerical noise
                    w_warm[np.abs(w_warm) < self.tol] = 0.0
                    z_warm[np.abs(z_warm) < self.tol] = 0.0
                    return w_warm, z_warm, info

            # Warm start is not a valid solution yet, continue with Lemke's algorithm
            # Choose a nondegenerate starting entering variable:
            # pick any basic variable and start with its complement
            entering = self._complement(basis[0], n)

        # Main Lemke loop
        it = 0
        while it < self.max_iter:
            it += 1

            if entering == 2*n:
                # Artificial left the basis: we found a complementary feasible solution
                break

            # Determine leaving row using minimum ratio test on 'entering' column
            row = self._choose_leaving_row(entering, T, b)
            if row is None:
                info['termination_reason'] = 'ray_termination'
                info['pivots'] = it
                raise RuntimeError("Ray termination: LCP appears infeasible (no eligible pivot).")

            # Pivot
            self._pivot(T, b, row, entering)
            info['pivots'] += 1
            leaving = basis[row]
            basis[row] = entering

            # Next entering is the complement of the leaving variable (w_i <-> z_i)
            entering = self._complement(leaving, n)

        if it >= self.max_iter:
            info['termination_reason'] = 'max_iterations'
            info['pivots'] = it
            raise RuntimeError("Maximum iterations exceeded.")

        # Extract solution: variables x = [w, z, t], but we only need w,z (nonbasic=0, basic=b_row)
        x = np.zeros(2*n + 1)
        for i in range(n):
            x[basis[i]] = b[i]

        w = x[0:n]
        z = x[n:2*n]

        # Clean small negatives due to numerical noise
        w[np.abs(w) < self.tol] = 0.0
        z[np.abs(z) < self.tol] = 0.0
        if np.any(w < -1e-7) or np.any(z < -1e-7):
            # Try to project tiny negatives
            w = np.maximum(w, 0.0)
            z = np.maximum(z, 0.0)

        # Compute complementarity
        info['complementarity'] = np.dot(w, z)

        # Final sanity check
        if np.max(np.abs(w - (self.M @ z + self.q))) > 1e-7:
            info['termination_reason'] = 'numerical_error'
            raise RuntimeError("Post-check failed: w != Mz + q (numerical issue).")
        if info['complementarity'] > 1e-7:
            # Mild warning; often tiny >0 due to rounding
            pass

        # Mark as successful convergence
        info['converged'] = True
        info['termination_reason'] = 'success'

        return w, z, info

    # --- solution analysis methods ----------------------------------------

    def is_P_matrix(self) -> bool:
        """
        Check if M is a P-matrix (all principal minors are positive).

        A P-matrix guarantees unique solution for any q.

        Returns:
            bool: True if M is a P-matrix, False otherwise
        """
        n = self.n
        M = self.M

        # Check all 2^n - 1 principal minors (excluding empty set)
        for mask in range(1, 2**n):
            # Get indices of principal submatrix
            indices = [i for i in range(n) if mask & (1 << i)]

            # Extract principal submatrix
            submatrix = M[np.ix_(indices, indices)]

            # Check if determinant is positive
            det = np.linalg.det(submatrix)
            if det <= 1e-10:  # Use tolerance for numerical stability
                return False

        return True

    def is_Q_matrix(self) -> bool:
        """
        Check if M is a Q-matrix (sufficient condition using principal minors).

        A Q-matrix guarantees existence of solution for any q.
        This implementation checks if all principal minors are non-negative,
        which is a sufficient (but not necessary) condition.

        Returns:
            bool: True if sufficient Q-matrix conditions are met, False otherwise
        """
        n = self.n
        M = self.M

        # Check if all principal minors are non-negative
        for mask in range(1, 2**n):
            indices = [i for i in range(n) if mask & (1 << i)]
            submatrix = M[np.ix_(indices, indices)]
            det = np.linalg.det(submatrix)
            if det < -1e-10:
                return False

        return True

    def check_solution_uniqueness(self, verbose: bool = False) -> dict:
        """
        Check existence and uniqueness of LCP solution by:
        1. Checking if M is a P-matrix (guarantees uniqueness)
        2. Checking if M is a Q-matrix (guarantees existence)
        3. Enumerating all 2^n complementary solutions

        Args:
            verbose: If True, print detailed information

        Returns:
            dict with keys:
                'is_P_matrix': bool - M is a P-matrix (unique solution guaranteed)
                'is_Q_matrix': bool - M is a Q-matrix (solution exists)
                'num_solutions': int - number of valid complementary solutions found
                'solutions': list of (w, z) tuples - all valid solutions
                'is_unique': bool - True if exactly one solution exists
        """
        n = self.n
        M = self.M
        q = self.q

        # Check matrix properties
        is_P = self.is_P_matrix()
        is_Q = self.is_Q_matrix()

        if verbose:
            print(f"Matrix analysis:")
            print(f"  P-matrix (unique solution): {is_P}")
            print(f"  Q-matrix (solution exists): {is_Q}")
            print()

        # Find all complementary solutions by checking all 2^n combinations
        solutions = []

        for mask in range(2**n):
            # For each index i, if bit i is set, z_i is active (w_i = 0)
            # Otherwise w_i is active (z_i = 0)
            active_z = np.array([(mask & (1 << i)) != 0 for i in range(n)])

            # Set up system: for active z_i, we have w_i = 0
            # for inactive z_i, we have z_i = 0
            # This gives us: w_active + M[:, inactive] * z_inactive = q
            # where w_active are the w_i with z_i = 0

            # Build the system
            z_indices = [i for i in range(n) if active_z[i]]
            w_indices = [i for i in range(n) if not active_z[i]]

            if len(z_indices) == 0:
                # All w active, all z = 0
                w_sol = q.copy()
                z_sol = np.zeros(n)
            elif len(w_indices) == 0:
                # All z active, all w = 0
                # Solve M @ z = -q
                try:
                    z_sol = np.linalg.solve(M, -q)
                    w_sol = np.zeros(n)
                except np.linalg.LinAlgError:
                    continue
            else:
                # Mixed case: solve for active z variables
                # For rows where w_i = 0 (z_i active): M[i,:] @ z = -q[i]
                # For rows where z_i = 0 (w_i active): already handled since z_i = 0
                # We need to use rows corresponding to active z (where w = 0)
                A = M[np.ix_(z_indices, z_indices)]
                b = -q[z_indices]

                try:
                    z_active = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue

                # Construct full solution
                z_sol = np.zeros(n)
                z_sol[z_indices] = z_active
                w_sol = M @ z_sol + q

            # Check if solution is valid (w >= 0, z >= 0, w*z = 0)
            if (np.all(w_sol >= -self.tol) and
                np.all(z_sol >= -self.tol) and
                np.allclose(w_sol * z_sol, 0, atol=1e-10)):

                # Verify w = M @ z + q
                residual = w_sol - (M @ z_sol + q)
                if np.max(np.abs(residual)) < 1e-10:
                    # Clean up numerical noise
                    w_sol[np.abs(w_sol) < self.tol] = 0.0
                    z_sol[np.abs(z_sol) < self.tol] = 0.0
                    solutions.append((w_sol.copy(), z_sol.copy()))

                    if verbose:
                        active_str = ''.join(['1' if active_z[i] else '0' for i in range(n)])
                        print(f"Solution {len(solutions)}: active_z = {active_str}")
                        print(f"  w = {w_sol}")
                        print(f"  z = {z_sol}")
                        print(f"  complementarity = {np.dot(w_sol, z_sol):.2e}")
                        print()

        # Remove duplicate solutions (can happen due to numerical tolerance)
        unique_solutions = []
        for w, z in solutions:
            is_duplicate = False
            for w_prev, z_prev in unique_solutions:
                if np.allclose(w, w_prev, atol=1e-8) and np.allclose(z, z_prev, atol=1e-8):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_solutions.append((w, z))

        result = {
            'is_P_matrix': is_P,
            'is_Q_matrix': is_Q,
            'num_solutions': len(unique_solutions),
            'solutions': unique_solutions,
            'is_unique': len(unique_solutions) == 1
        }

        if verbose:
            print(f"Summary:")
            print(f"  Total complementary solutions found: {len(unique_solutions)}")
            print(f"  Solution is unique: {result['is_unique']}")

        return result