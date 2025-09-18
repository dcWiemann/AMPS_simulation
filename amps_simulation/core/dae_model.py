from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import numpy as np
import networkx as nx
from amps_simulation.core.components import Resistor, PowerSwitch, Inductor, Capacitor, Source, Meter, Diode
from amps_simulation.core.electrical_model import ElectricalModel
from amps_simulation.core.lcp_solver import DiodeLCPSolver
from sympy import Matrix, Symbol, sympify, solve
import logging
import itertools

class DaeModel(ABC):
    """Abstract base class for Differential-Algebraic Equation (DAE) models.
    
    This class defines the interface that all DAE models must implement.
    A DAE model consists of differential equations (derivatives) and algebraic equations (outputs).
    
    Attributes:
        derivatives (Dict[str, float]): Dictionary mapping state variable names to their derivatives
        outputs (Dict[str, float]): Dictionary mapping output variable names to their values
    """
    
    def __init__(self, graph: nx.Graph):
        """Initialize the DAE model with empty derivatives and outputs dictionaries."""
        self.graph = graph
        self.derivatives: Dict[str, float] = {}
        self.output_eqs: Dict[Symbol, Symbol] = {}
    
    
    def get_derivatives(self) -> Dict[str, float]:
        """Get the current derivatives of the state variables.
        
        Returns:
            Dict[str, float]: Dictionary mapping state variable names to their derivatives
        """
        return self.derivatives
    
    def get_outputs(self) -> Dict[Symbol, Symbol]:
        """Get the current output values.
        
        Returns:
            Dict[str, float]: Dictionary mapping output variable names to their values
        """
        return self.output_eqs
    

class ElectricalDaeModel(DaeModel):
    """DAE model for electrical systems.
    
    This class extends the DaeModel class to handle electrical systems.
    It includes methods for evaluating the model and updating the state.

    Attributes:
        derivatives (Dict[str, float]): Dictionary mapping state variable names to their derivatives
        outputs (Dict[str, float]): Dictionary mapping output variable names to their values
    """
    # Configurable tolerances for diode state checks
    diode_current_tol = 1e-6
    diode_voltage_tol = 1e-6
    def __init__(self, electrical_model: ElectricalModel):
        super().__init__(electrical_model.graph)
        self.electrical_model = electrical_model
        self.initialized = False
        
        # Initialize LCP solver for diode state detection
        self.lcp_solver = DiodeLCPSolver(tolerance=1e-10)

    def initialize(self, initial_state_values=None, initial_input_values=None):
        # Initialize the electrical graph first
        if not self.electrical_model.initialized:
            self.electrical_model.initialize()
        
        # Find all circuit variables
        self.state_vars = self.find_state_vars()
        self.output_vars = self.find_output_vars()
        self.input_vars = self.find_input_vars()
        
        # Get graph structure from electrical_model
        self.junction_voltage_var_list = self.electrical_model.junction_voltage_var_list
        self.component_current_var_list = self.electrical_model.component_current_var_list
        self.component_voltage_var_list = self.electrical_model.component_voltage_var_list
        self.incidence_matrix = self.electrical_model.incidence_matrix
        self.switch_list = self.electrical_model.switch_list
        self.diode_list = self.electrical_model.diode_list
        
        # Compute circuit equations (order matters!)
        self.kcl_eqs = self.compute_kcl_equations()
        self.kvl_eqs = self.compute_kvl_equations()
        self.static_eqs = self.compute_static_component_equations()
        self.switch_eqs = self.compute_switch_equations()
        
        # Initialize diode modes using iterative approach if we have diodes and initial values
        if self.electrical_model.diode_list and initial_state_values is not None:
            # Use the iterative diode state detection approach
            conducting_states = self.detect_diode_states(
                initial_state_values, 
                initial_input_values or np.zeros(len(self.input_vars))
            )
            self._initialize_diode_modes(conducting_states)
        
        # Compute diode equations with proper modes
        self.diode_eqs = self.compute_diode_equations()
        self.circuit_eqs = self.compute_circuit_equations()
        self.derivatives = self.compute_derivatives()
        self.output_eqs = self.compute_output_equations()
        self.initialized = True
    
    def _solve_circuit_equations(self, equations: List, vars_to_solve: List) -> Dict:
        """Generic circuit equation solver (DRY - used by both circuit solving and LCP).
        
        Args:
            equations: List of equations to solve
            vars_to_solve: List of variables to solve for
            
        Returns:
            Dictionary mapping variables to their symbolic expressions
        """
        logging.debug(f"Circuit solver: {len(equations)} equations, {len(vars_to_solve)} variables")
        
        # Log equations and variables for debugging
        logging.debug("Equations to solve:")
        for i, eq in enumerate(equations):
            logging.debug(f"  [{i}] {eq}")
        
        logging.debug("Variables to solve for:")
        for i, var in enumerate(vars_to_solve):
            logging.debug(f"  [{i}] {var}")
        
        # Check equation/variable balance
        if len(equations) != len(vars_to_solve):
            raise ValueError(f"Equation/variable mismatch: {len(equations)} equations, {len(vars_to_solve)} variables")
        
        # Solve the equations
        solution = solve(equations, vars_to_solve)
        
        logging.debug(f"Circuit solver: Found {len(solution)} solutions")
        
        # Log the solutions found
        if solution:
            logging.debug("Solutions found:")
            for var, expr in solution.items():
                logging.debug(f"  {var} = {expr}")
        
        if len(solution) != len(vars_to_solve):
            # Log which variables have no solution
            solved_vars = set(solution.keys())
            unsolved_vars = [var for var in vars_to_solve if var not in solved_vars]
            if unsolved_vars:
                logging.debug(f"Variables without solutions: {unsolved_vars}")
            raise ValueError(f"Did not find a solution for every variable: {len(solution)} solutions, {len(vars_to_solve)} variables")
        
        return solution

    def _solve_circuit_equations_safe(self, equations: List, vars_to_solve: List) -> Dict:
        """Safe circuit equation solver that handles cases where sympy.solve() returns empty list.
        
        Args:
            equations: List of equations to solve
            vars_to_solve: List of variables to solve for
            
        Returns:
            Dictionary mapping variables to their symbolic expressions, or None if no solution exists
        """
        logging.debug(f"Safe circuit solver: {len(equations)} equations, {len(vars_to_solve)} variables")
        
        # Check equation/variable balance
        if len(equations) != len(vars_to_solve):
            logging.debug(f"Equation/variable mismatch: {len(equations)} equations, {len(vars_to_solve)} variables")
            return None
        
        try:
            # Solve the equations
            solution = solve(equations, vars_to_solve)
            
            # Handle different return types from sympy.solve()
            if isinstance(solution, dict):
                # Normal case: solution is a dictionary
                if len(solution) == len(vars_to_solve):
                    logging.debug(f"Safe solver: Found complete solution with {len(solution)} variables")
                    return solution
                else:
                    logging.debug(f"Safe solver: Incomplete solution - {len(solution)} solutions for {len(vars_to_solve)} variables")
                    return None
            elif isinstance(solution, list):
                # sympy returned a list (usually empty when no solution exists)
                if len(solution) == 0:
                    logging.debug("Safe solver: No solution found (empty list returned)")
                    return None
                elif len(solution) == 1 and isinstance(solution[0], dict):
                    # Sometimes sympy returns [solution_dict]
                    logging.debug("Safe solver: Found solution in list format")
                    return solution[0]
                else:
                    logging.debug(f"Safe solver: Unexpected list format with {len(solution)} items")
                    return None
            else:
                logging.debug(f"Safe solver: Unexpected solution type: {type(solution)}")
                return None
                
        except Exception as e:
            logging.debug(f"Safe solver: Exception during solving: {e}")
            return None

    def _extract_diode_voltage_expressions(self, solution: Dict) -> List:
        """Extract diode voltage expressions in ElectricalModel.diode_list order.
        
        Args:
            solution: Dictionary mapping variables to their symbolic expressions
            
        Returns:
            List of symbolic diode voltage expressions in consistent order
        """
        diode_voltage_exprs = []
        
        # Iterate through diodes in ElectricalModel order to maintain consistency
        for diode in self.electrical_model.diode_list:
            voltage_var = diode.voltage_var
            
            if voltage_var in solution:
                # Extract v_D expression 
                expr = solution[voltage_var]
                diode_voltage_exprs.append(expr)
            else:
                raise ValueError(f"Could not find solution for diode voltage {voltage_var} (diode {diode.comp_id})")
        
        logging.debug(f"Extracted {len(diode_voltage_exprs)} diode voltage expressions in order")
        return diode_voltage_exprs

    def _solve_lcp(self, M: np.ndarray, q: np.ndarray) -> List[bool]:
        """Solve LCP and return diode modes in ElectricalModel.diode_list order.
        
        Args:
            M: LCP matrix
            q: LCP vector
            
        Returns:
            List of diode conducting states (True = conducting) in consistent order
        """
        # Get diode names in ElectricalModel order for logging
        diode_names = [diode.comp_id for diode in self.electrical_model.diode_list]
        
        # Solve LCP
        conducting_states, info = self.lcp_solver.detect_diode_states(M, q, diode_names)
        
        logging.debug(f"LCP solver: converged={info['converged']}, pivots={info['pivots']}")
        if info['converged']:
            logging.debug(f"LCP complementarity: {info['complementarity']:.2e}")
        
        return conducting_states

    def _initialize_diode_modes(self, conducting_states: List[bool]) -> None:
        """Update diode component modes based on LCP solution.
        
        Args:
            conducting_states: List of diode conducting states in ElectricalModel.diode_list order
        """
        if len(conducting_states) != len(self.electrical_model.diode_list):
            raise ValueError(f"Conducting states length {len(conducting_states)} != diode count {len(self.electrical_model.diode_list)}")
        
        # Update diode modes in order
        for diode, is_conducting in zip(self.electrical_model.diode_list, conducting_states):
            diode.is_on = is_conducting
            logging.debug(f"Set diode {diode.comp_id}: {'CONDUCTING' if is_conducting else 'BLOCKING'}")

    def find_state_vars(self) -> List[Symbol]:
        """Find the state variables in the graph.
        
        Returns:
            List[Symbol]: List of state variables
        """
        state_vars = []
        for _, _, data in self.graph.edges(data=True):
            if isinstance(data['component'], Inductor):
                state_vars.append(data['component'].current_var)
            elif isinstance(data['component'], Capacitor):
                state_vars.append(data['component'].voltage_var)
        return state_vars
    
    def find_output_vars(self) -> List[Symbol]:
        """Find the output variables in the graph.
        
        Returns:
            List[Symbol]: List of output variables
        """
        output_vars = []
        for _, _, data in self.graph.edges(data=True):  # Correctly unpack edge data
            if isinstance(data['component'], Meter):
                output_vars.append(data['component'].output_var)
        return output_vars

    def find_input_vars(self) -> List[Symbol]:
        """Find the input variables in the graph.
        
        Returns:
            List[Symbol]: List of input variables
        """
        input_vars = []
        for _, _, data in self.graph.edges(data=True):  # Correctly unpack edge data
            if isinstance(data['component'], Source):
                input_vars.append(data['component'].input_var)
        return input_vars
    
    
    def compute_kcl_equations(self) -> List[Symbol]:
        """Compute Kirchhoff's Current Law equations.
        
        Returns:
            List[Symbol]: List of KCL equations in symbolic form, excluding the ground node equation.
        """
        if self.initialized == False:
            _, comp_current_vars, _ = self.electrical_model.variable_lists()
            incidence_matrix = self.electrical_model.compute_incidence_matrix()
        else:
            comp_current_vars = self.component_current_var_list
            incidence_matrix = self.incidence_matrix
        
        # Create current vector and multiply with incidence matrix
        current_vector = Matrix(comp_current_vars)
        kcl_equations = incidence_matrix * current_vector
        
        # Find the ground node index (where voltage_var is None)
        ground_node_idx = None
        for i, node in enumerate(self.graph.nodes()):
            if self.graph.nodes[node]['junction'].is_ground:
                ground_node_idx = i
                break
        
        # Remove the ground node equation if found
        if ground_node_idx is not None:
            kcl_equations = [eq for i, eq in enumerate(kcl_equations) if i != ground_node_idx]
        
        # Convert kcl_eqs to a list if it is a MutableDenseMatrix
        if isinstance(kcl_equations, Matrix):
            kcl_equations = list(kcl_equations)
        
        return kcl_equations
    
    def compute_kvl_equations(self) -> List[Symbol]:
        """Compute Kirchhoff's Voltage Law equations.
        
        Returns:
            List[Symbol]: List of KVL equations in symbolic form.
        """
        if self.initialized == False:
            junction_voltage_var_list, _, comp_voltage_vars = self.electrical_model.variable_lists()
            incidence_matrix = self.electrical_model.compute_incidence_matrix()
        else:
            junction_voltage_var_list = self.junction_voltage_var_list
            comp_voltage_vars = self.component_voltage_var_list
            incidence_matrix = self.incidence_matrix
        
        # Create voltage vector and multiply with transpose of incidence matrix
        voltage_vector = Matrix(junction_voltage_var_list)
        kvl_equations = incidence_matrix.T * voltage_vector
        
        # Add component voltage variables
        kvl_eqs = [eq - v_comp for eq, v_comp in zip(kvl_equations, comp_voltage_vars)]
        
        return kvl_eqs
    
    def compute_static_component_equations(self) -> List[Symbol]:
        """Compute the static component equations of the graph.
        
        Returns:
            List[Symbol]: The static component equations of the graph.
        """
        static_eqs = []
        for _, _, data in self.graph.edges(data=True):
            component = data['component']
            if isinstance(component, (Resistor, Meter)):  # Exclude diodes - they are state-dependent
                static_eqs.append(component.get_comp_eq())
        return static_eqs
    
        
    def compute_switch_equations(self) -> List[Symbol]:
        """Compute the switch equations of the graph.
        
        Returns:
            List[Symbol]: The switch equations of the graph.
        """
        if self.initialized == False:
            switch_list = self.electrical_model.find_switches()
        else:
            switch_list = self.switch_list
        
        switch_eqs = []
        for switch in switch_list:
            switch_eqs.append(switch.get_comp_eq())

        return switch_eqs
    
    def compute_diode_equations(self) -> List[Symbol]:
        """Compute the diode equations of the graph.
        
        Diode equations are state-dependent and applied based on conducting/blocking state:
        - Conducting (is_on=True): voltage equation (v_D = 0)  
        - Blocking (is_on=False): current equation (i_D = 0)
        
        Returns:
            List[Symbol]: The diode equations of the graph.
        """
        if self.initialized == False:
            diode_list = self.electrical_model.find_diodes()
        else:
            diode_list = self.diode_list
        
        diode_eqs = []
        for diode in diode_list:
            diode_eqs.append(diode.get_comp_eq())

        return diode_eqs
    
    def compute_circuit_equations(self) -> Dict:
        """Solve the full circuit variables including diode equations.
        
        Returns:
            Dict: Dictionary mapping variables to their symbolic expressions
        """
        # Get all equations including diode equations (this is the full circuit)
        if self.initialized:
            # Use precomputed equations during normal operation
            equations = self.kcl_eqs + self.kvl_eqs + self.static_eqs + self.switch_eqs + self.diode_eqs
            input_vars = self.input_vars
            state_vars = self.state_vars
        else:
            # Compute equations during initialization
            equations = (self.compute_kcl_equations() + self.compute_kvl_equations() + 
                        self.compute_static_component_equations() + self.compute_switch_equations() + 
                        self.compute_diode_equations())
            input_vars = self.find_input_vars()
            state_vars = self.find_state_vars()
        
        logging.debug(f"Full circuit: input_vars={[str(v) for v in input_vars]}")
        logging.debug(f"Full circuit: state_vars={[str(v) for v in state_vars]}")
        
        # Get all variables from electrical graph
        junction_voltage_var_list, component_current_var_list, component_voltage_var_list = self.electrical_model.variable_lists()
        
        # Remove ground node (0) from junction voltage variables
        junction_voltage_var_list_cleaned = [var for var in junction_voltage_var_list if var != 0]
        
        # Combine all variables
        combined_vars = junction_voltage_var_list_cleaned + component_current_var_list + component_voltage_var_list
        
        # Remove excluded variables (input_vars and state_vars only)
        excluded = set(input_vars) | set(state_vars)
        vars_to_solve = [var for var in combined_vars if var not in excluded]
        
        # Use safe solver that handles empty solutions
        solution = self._solve_circuit_equations_safe(equations, vars_to_solve)
        if solution is None:
            raise ValueError(f"Could not solve circuit equations: {len(equations)} equations, {len(vars_to_solve)} variables")
        return solution
    

    def compute_derivatives(self) -> List[Symbol]:
        """Compute the derivatives of the state variables.
        
        Returns:
            List[Symbol]: List of derivatives of the state variables
        """
        derivatives = []
        for _, _, data in self.graph.edges(data=True):
            if isinstance(data['component'], Inductor):
                derivatives.append(data['component'].get_comp_eq())
            elif isinstance(data['component'], Capacitor):
                derivatives.append(data['component'].get_comp_eq())
        
        if self.circuit_eqs is not None:
            circuit_eqs = self.circuit_eqs
        else:
            circuit_eqs = self.compute_circuit_equations()
        
        #substitute circuit_vars into derivatives
        for var in circuit_eqs:
            derivatives = [eq.subs(var, circuit_eqs[var]) for eq in derivatives]
        
        # logging.debug("derivatives: ", derivatives)

        return derivatives
    

    def find_output_vars(self) -> List[Symbol]:
        """Find the outputs in the graph.
        
        Returns:
            List[Symbol]: List of outputs
        """
        outputs = []
        for _, _, data in self.graph.edges(data=True):
            if isinstance(data['component'], Meter):
                outputs.append(data['component'].output_var)
        return outputs
    
    def compute_output_equations(self) -> Dict[Symbol, Symbol]:
        """Compute the outputs of the circuit.
        
        Returns:
            Dict[Symbol, Symbol]: Dictionary mapping output variables to their expressions
        """
        if self.initialized == False:
            outputs = self.find_output_vars()
        else:
            outputs = self.output_vars
        
        output_eqs = {output: self.circuit_eqs[output] for output in outputs}
        return output_eqs
    
    def update_switch_states(self, t = None) -> None:
        """Update the switch states. This method is called during simulation when a switch event is detected.
        It uses the kcl, kvl and component equations previously computed and only updates the switch equations.
        Based on that, it recomputes the circuit variables and derivatives.

        """
        if t is not None:
            assert isinstance(t, float), "Time must be a float"
            switchmap = {switch.comp_id: switch.set_switch_state(t) for switch in self.switch_list}
        
        assert self.initialized == True, "Model must be initialized before updating switch states"
        self.switch_eqs = self.compute_switch_equations()
        self.diode_eqs = self.compute_diode_equations()  # Ensure diode equations are current
        self.circuit_eqs = self.compute_circuit_equations()
        self.derivatives = self.compute_derivatives()
        self.output_eqs = self.compute_output_equations()
        
        if t is not None:
            return self.circuit_eqs, self.derivatives, switchmap
        else:
            return self.circuit_eqs, self.derivatives
    
    def compute_diode_lcp_matrices(self, state_values: np.ndarray, input_values: np.ndarray) -> Tuple[Matrix, Matrix]:
        """Compute Linear Complementarity Problem matrices for diode state detection.
        
        Formulates the diode problem as: -v_D = M*i_D + q
        where -v_D is the vector of negative diode voltages, i_D is the vector of diode currents,
        M is the diode impedance matrix, and q is the offset vector.
        
        LCP constraints: -v_D ≥ 0, i_D ≥ 0, (-v_D)·i_D = 0
        - Blocking: -v_D > 0 (i.e., v_D < 0, reverse bias), i_D = 0
        - Conducting: -v_D = 0 (i.e., v_D = 0, forward bias), i_D > 0
        
        Args:
            state_values: Current values of state variables (inductor currents, capacitor voltages)
            input_values: Current values of input variables (source values)
            
        Returns:
            Tuple[Matrix, Matrix]: (M matrix, q vector) for LCP formulation -v_D = M*i_D + q
        """
        if not self.electrical_model.diode_list:
            # No diodes in circuit, return empty matrices
            return Matrix([]), Matrix([])
        
        # Get equations EXCEPT diode equations (since we're solving for diode voltages)
        equations = self.kcl_eqs + self.kvl_eqs + self.static_eqs + self.switch_eqs
        logging.debug(f"LCP: Base equations count: KCL={len(self.kcl_eqs)}, KVL={len(self.kvl_eqs)}, Static={len(self.static_eqs)}, Switch={len(self.switch_eqs)}")
        
        # Add initial condition equations: state_var = state_value, input_var = input_value
        initial_condition_count = 0
        for i, state_var in enumerate(self.state_vars):
            if i < len(state_values):
                equations.append(state_var - state_values[i])
                initial_condition_count += 1
                logging.debug(f"LCP: Added state equation: {state_var} = {state_values[i]}")
        for i, input_var in enumerate(self.input_vars):
            if i < len(input_values):
                equations.append(input_var - input_values[i])
                initial_condition_count += 1
                logging.debug(f"LCP: Added input equation: {input_var} = {input_values[i]}")
        
        logging.debug(f"LCP: Added {initial_condition_count} initial condition equations")
        logging.debug(f"LCP: Total equations before solving: {len(equations)}")
        
        # Get diode current variables to exclude from solving
        diode_current_vars = [diode.current_var for diode in self.electrical_model.diode_list]
        logging.debug(f"LCP: Diode current vars to exclude: {[str(v) for v in diode_current_vars]}")
        
        # Get all variables from electrical graph
        junction_voltage_var_list, component_current_var_list, component_voltage_var_list = self.electrical_model.variable_lists()
        
        # Remove ground node (0) from junction voltage variables
        junction_voltage_var_list_cleaned = [var for var in junction_voltage_var_list if var != 0]
        
        logging.debug(f"LCP: Variable counts: Junction={len(junction_voltage_var_list_cleaned)}, Current={len(component_current_var_list)}, Voltage={len(component_voltage_var_list)}")
        
        # Combine all variables
        combined_vars = junction_voltage_var_list_cleaned + component_current_var_list + component_voltage_var_list
        logging.debug(f"LCP: Total combined variables: {len(combined_vars)}")
        
        # Remove excluded variables (only diode_current_vars)
        excluded = set(diode_current_vars)
        vars_to_solve = [var for var in combined_vars if var not in excluded]
        logging.debug(f"LCP: Variables to solve: {len(vars_to_solve)} (excluded {len(excluded)} diode currents)")
        
        # Solve circuit without diodes using generic solver
        print("\n" + "="*80)
        print("LCP FORMULATION DEBUG - EQUATIONS, VARIABLES, AND SOLUTIONS")
        print("="*80)
        
        print(f"\nEQUATIONS ({len(equations)} total):")
        for i, eq in enumerate(equations):
            print(f"  [{i:2d}] {eq} = 0")
        
        print(f"\nVARIABLES TO SOLVE ({len(vars_to_solve)} total):")
        for i, var in enumerate(vars_to_solve):
            print(f"  [{i:2d}] {var}")
        
        solution = self._solve_circuit_equations(equations, vars_to_solve)
        
        print(f"\nSOLUTIONS ({len(solution)} total):")
        for var, expr in solution.items():
            print(f"  {var} = {expr}")
        
        print("="*80)
        
        # Extract diode voltage expressions in correct order
        diode_voltage_exprs = self._extract_diode_voltage_expressions(solution)
        
        # Apply -1 factor for LCP formulation: -v_D = M*i_D + q (positive = blocking)
        diode_voltage_exprs = [-expr for expr in diode_voltage_exprs]
        
        # Extract M matrix and q vector by collecting coefficients
        n_diodes = len(self.electrical_model.diode_list)
        M_matrix = Matrix.zeros(n_diodes, n_diodes)
        q_vector = Matrix.zeros(n_diodes, 1)
        
        for i, expr in enumerate(diode_voltage_exprs):
            # Extract coefficients for each diode current (maintain order)
            for j, diode_current_var in enumerate(diode_current_vars):
                coeff = expr.coeff(diode_current_var, 1)
                if coeff is not None:
                    M_matrix[i, j] = coeff
            
            # Get constant term (expression with all diode currents set to zero)
            constant_term = expr.subs({var: 0 for var in diode_current_vars})
            q_vector[i] = constant_term
        
        logging.debug(f"LCP: Generated M matrix shape: {M_matrix.shape}")
        logging.debug(f"LCP: Generated q vector shape: {q_vector.shape}")
        
        return M_matrix, q_vector
    
    def detect_diode_states(self, state_values: np.ndarray, input_values: np.ndarray, t: float = 0.0) -> List[bool]:
        """Detect the conducting state of all diodes in the circuit.
        
        This is a modular interface that can be replaced with different detection algorithms:
        - LCP solver (current implementation)
        - Iterative methods
        - Heuristic approaches
        
        Args:
            state_values: Current values of state variables
            input_values: Current values of input variables  
            t: Current simulation time
            
        Returns:
            List[bool]: List of diode conducting states (True = conducting, False = blocking)
        """
        if not self.diode_list:
            return []
        
        # Use iterative diode state detection approach
        return self._detect_diode_states_iterative(state_values, input_values)
    
    def _detect_diode_states_lcp(self, state_values: np.ndarray, input_values: np.ndarray) -> List[bool]:
        """Detect diode states using Linear Complementarity Problem formulation.
        
        Solves the LCP: -v_D = M*i_D + q with complementarity constraints:
        - -v_D >= 0, i_D >= 0, (-v_D) * i_D = 0
        - If -v_D > 0: diode is reverse-biased (blocking), i_D = 0
        - If i_D > 0: diode is forward-biased (conducting), -v_D = 0
        
        Uses the integrated LCP solver for robust solution.
        
        Args:
            state_values: Current values of state variables
            input_values: Current values of input variables
            
        Returns:
            List[bool]: List of diode conducting states
        """
        try:
            # Get LCP matrices
            M_matrix, q_vector = self.compute_diode_lcp_matrices(state_values, input_values)
            
            if M_matrix.shape[0] == 0:
                return []  # No diodes
            
            # Convert to numpy for numerical solving
            M_np = np.array(M_matrix.tolist(), dtype=float)
            q_np = np.array(q_vector.tolist(), dtype=float).flatten()
            
            # Get diode names for logging
            diode_names = [diode.comp_id for diode in self.diode_list]
            
            # Use integrated LCP solver for robust solution
            conducting_states, info = self.lcp_solver.detect_diode_states(M_np, q_np, diode_names)
            
            # Log solver results
            if info["converged"]:
                logging.debug(f"LCP solver converged in {info['pivots']} pivots")
                logging.debug(f"Complementarity: {info['complementarity']:.2e}")
            else:
                logging.warning(f"LCP solver did not converge after {info['pivots']} pivots")
                if info['last_violation']:
                    logging.warning(f"Last violation: {info['last_violation']}")
            
            return conducting_states
            
        except Exception as e:
            logging.warning(f"LCP diode state detection failed: {e}")
            # Fallback: assume all diodes are blocking
            return [False] * len(self.diode_list)
    
    def _detect_diode_states_iterative(self, state_values: np.ndarray, input_values: np.ndarray, max_iterations: int = 100) -> List[bool]:
        """Detect diode states using iterative approach.
        
        Algorithm:
        1. Start with all diodes blocking (i_D = 0)
        2. Solve circuit equations with current diode assumptions  
        3. Check if resulting diode currents/voltages are consistent
        4. Update diode states if inconsistencies found
        5. Repeat until convergence or max iterations reached
        
        Args:
            state_values: Current values of state variables
            input_values: Current values of input variables
            max_iterations: Maximum number of iterations to try
            
        Returns:
            List[bool]: List of diode conducting states (True = conducting, False = blocking)
        """
        if not self.diode_list:
            return []
        
        # Input validation
        if state_values is None or len(state_values) != len(self.state_vars):
            logging.error(f"Invalid state_values: expected {len(self.state_vars)} values, got {len(state_values) if state_values is not None else None}")
            return [False] * len(self.diode_list)
        
        if input_values is None or len(input_values) != len(self.input_vars):
            logging.error(f"Invalid input_values: expected {len(self.input_vars)} values, got {len(input_values) if input_values is not None else None}")
            return [False] * len(self.diode_list)
        
        if max_iterations <= 0:
            logging.error(f"Invalid max_iterations: {max_iterations}")
            return [False] * len(self.diode_list)
        
        # # Initialize all diodes as blocking
        # current_states = [False] * len(self.diode_list)
        
        # Get all possible combinations of diode states
        all_combinations = list(itertools.product([True, False], repeat=len(self.diode_list)))
        
        # Initialize diodes with first combination
        current_states = all_combinations[0]
        visited_combinations = [current_states]

        logging.debug(f"Starting iterative diode detection with {len(self.diode_list)} diodes")
        
        for iteration in range(max_iterations):
            logging.debug(f"Iteration {iteration + 1}: Current states = {current_states}")
            
            # Set diode modes based on current assumptions
            for diode, is_conducting in zip(self.diode_list, current_states):
                diode.is_on = is_conducting

            # Recompute diode equations with current states
            diode_eqs = self.compute_diode_equations()
            
            # Try to solve circuit with current diode assumptions
            try:
                # Get all equations including current diode equations
                equations = self.kcl_eqs + self.kvl_eqs + self.static_eqs + self.switch_eqs + diode_eqs
                
                # Add initial condition equations
                for i, state_var in enumerate(self.state_vars):
                    if i < len(state_values):
                        equations.append(state_var - state_values[i])
                for i, input_var in enumerate(self.input_vars):
                    if i < len(input_values):
                        equations.append(input_var - input_values[i])
                
                # Get all variables except diode currents for blocking diodes and diode voltages for conducting diodes
                junction_voltage_var_list, component_current_var_list, component_voltage_var_list = self.electrical_model.variable_lists()
                junction_voltage_var_list_cleaned = [var for var in junction_voltage_var_list if var != 0]
                combined_vars = junction_voltage_var_list_cleaned + component_current_var_list + component_voltage_var_list

                vars_to_solve = combined_vars

                # Attempt to solve
                solution = self._solve_circuit_equations_safe(equations, vars_to_solve)
                
                if solution is None:
                    current_states = all_combinations[iteration + 1]
                    visited_combinations.append(current_states)
                    continue
                
                # Check solution consistency
                new_states = list(current_states)
                states_changed = False
                
                for i, diode in enumerate(self.diode_list):
                    # Check if diode current is negative (not allowed)
                    if diode.current_var not in solution:
                        logging.debug(f"Diode {diode.comp_id}: current variable {diode.current_var} not in solution, skipping consistency check")
                        new_states = all_combinations[iteration + 1]
                        new_states = self._check_diode_states_visited(new_states, visited_combinations, iteration, all_combinations)
                        states_changed = True
                        continue
                    
                    if diode.voltage_var not in solution:
                        logging.debug(f"Diode {diode.comp_id}: voltage variable {diode.voltage_var} not in solution, skipping consistency check")
                        new_states = all_combinations[iteration + 1]
                        new_states = self._check_diode_states_visited(new_states, visited_combinations, iteration, all_combinations)
                        states_changed = True
                        continue

                    
                    i_diode_val = float(solution[diode.current_var].subs({var: val for var, val in zip(self.state_vars + self.input_vars, 
                                                                                                            list(state_values) + list(input_values))}))
                    if i_diode_val < -self.diode_current_tol:  # Small tolerance for numerical errors
                        logging.debug(f"Diode {diode.comp_id}: negative current {i_diode_val:.6f} while conducting, switch to blocking")
                        new_states[i] = not current_states[i]     
                        # Check if new combination has been visited
                        new_states = self._check_diode_states_visited(new_states, visited_combinations, iteration, all_combinations)
                        states_changed = True
                        continue
                    # Check if diode voltage is positive (not allowed)
                    v_diode_val = float(solution[diode.voltage_var].subs({var: val for var, val in zip(self.state_vars + self.input_vars,
                                                                                                            list(state_values) + list(input_values))}))
                    if v_diode_val > self.diode_voltage_tol:  # Small tolerance for numerical errors
                        logging.debug(f"Diode {diode.comp_id}: positive voltage {v_diode_val:.6f} while blocking, switch to conducting")
                        new_states[i] = not current_states[i]
                        # Check if new combination has been visited
                        new_states = self._check_diode_states_visited(new_states, visited_combinations, iteration, all_combinations)
                        states_changed = True   
                        continue

                    if current_states[i]:  # Currently conducting
                        # Check if voltage is negative (reverse bias means should be blocking)
                        v_diode_val = float(solution[diode.voltage_var].subs({var: val for var, val in zip(self.state_vars + self.input_vars,
                                                                                                            list(state_values) + list(input_values))}))
                        if v_diode_val < -self.diode_voltage_tol:  # Small tolerance for numerical errors
                            logging.debug(f"Diode {diode.comp_id}: negative voltage {v_diode_val:.6f} while conducting, switch to blocking")
                            new_states[i] = not current_states[i]
                            # Check if new combination has been visited
                            new_states = self._check_diode_states_visited(new_states, visited_combinations, iteration, all_combinations)
                            states_changed = True
                            continue
                    else:  # Currently blocking
                        # Check if current is positive (forward bias means should be conducting)
                        i_diode_val = float(solution[diode.current_var].subs({var: val for var, val in zip(self.state_vars + self.input_vars,
                                                                                                            list(state_values) + list(input_values))}))
                        if i_diode_val > self.diode_current_tol:  # Small tolerance for numerical errors
                            logging.debug(f"Diode {diode.comp_id}: positive current {i_diode_val:.6f} while blocking, switch to conducting")
                            new_states[i] = not current_states[i]
                            # Check if new combination has been visited
                            new_states = self._check_diode_states_visited(new_states, visited_combinations, iteration, all_combinations)
                            states_changed = True
                            continue
                
                if not states_changed:
                    # Converged!
                    logging.debug(f"Iterative diode detection converged in {iteration + 1} iterations")
                    states_str = [f'{diode.comp_id}={"CONDUCTING" if state else "BLOCKING"}' for diode, state in zip(self.diode_list, current_states)]
                    logging.debug(f"Final states: {states_str}")
                    return current_states
                
                # Update states for next iteration
                current_states = new_states
                if current_states not in visited_combinations:
                    visited_combinations.append(current_states)
                else:
                    logging.warning(f"Degeneracy detected after {iteration + 1} iterations. Diode states: {current_states}")
                
            except Exception as e:
                logging.debug(f"Iteration {iteration + 1}: Exception during solving: {e}")
                # Try different diode state combination
                if iteration == 0:
                    current_states = [True] * len(self.diode_list)  # Try all conducting
                else:
                    # For now, just return all blocking as fallback
                    logging.warning(f"Iterative diode detection failed after {iteration + 1} iterations: {e}")
                    return [False] * len(self.diode_list)
        
        # Max iterations reached
        logging.warning(f"Iterative diode detection did not converge after {max_iterations} iterations")
        return current_states  # Return best attempt
    
    def _check_diode_states_visited(self, new_states: List[bool], visited: List[List[bool]], iteration: int, all_combinations: List[List[bool]]) -> List[bool]:
        """Check if the new diode states have been visited before. If so, try the next combination.
        Args:
            new_states: Newly detected diode states
            visited: List of previously visited diode state combinations
            iteration: Current iteration number
            all_combinations: All possible diode state combinations
        Returns:
            List[bool]: Updated diode states
        """
        if new_states not in visited:
            return new_states
        else:
            new_combination_found = False
            j = iteration + 1
            while not new_combination_found and j < len(all_combinations):
                new_states = all_combinations[j]  # Try next combination
                j += 1
                if new_states not in visited:
                    new_combination_found = True
                    return new_states
            if not new_combination_found:
                logging.warning("No valid diode conduction mode found. Stopping iteration.")
                return new_states

    def update_diode_states(self, state_values: np.ndarray, input_values: np.ndarray, t: float = 0.0) -> None:
        """Update the conducting states of all diodes and recompute circuit equations.
        
        Args:
            state_values: Current values of state variables
            input_values: Current values of input variables
            t: Current simulation time
        """
        if not self.diode_list:
            return
        
        # Detect new diode states
        new_states = self.detect_diode_states(state_values, input_values, t)
        
        # Update diode components
        states_changed = False
        for i, (diode, new_state) in enumerate(zip(self.diode_list, new_states)):
            if diode.is_on != new_state:
                diode.is_on = new_state
                states_changed = True
                logging.debug(f"Diode {diode.comp_id} state changed to {'conducting' if new_state else 'blocking'}")
        
        # Recompute circuit equations if any diode state changed
        if states_changed:
            # Recompute diode equations (state-dependent)
            self.diode_eqs = self.compute_diode_equations()
            # Recompute all circuit variables
            self.circuit_eqs = self.compute_circuit_equations()
            self.derivatives = self.compute_derivatives()
            self.output_eqs = self.compute_output_equations()
            logging.debug("Circuit equations recomputed due to diode state changes")