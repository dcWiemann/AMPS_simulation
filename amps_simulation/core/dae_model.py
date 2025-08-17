from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import numpy as np
import networkx as nx
from amps_simulation.core.components import Resistor, PowerSwitch, Inductor, Capacitor, Source, Meter, Diode
from amps_simulation.core.electrical_graph import ElectricalGraph
from amps_simulation.core.lcp_solver import DiodeLCPSolver
from sympy import Matrix, Symbol, sympify, solve
import logging

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
    def __init__(self, electrical_graph: ElectricalGraph):
        super().__init__(electrical_graph.graph)
        self.electrical_graph = electrical_graph
        self.initialized = False
        
        # Initialize LCP solver for diode state detection
        self.lcp_solver = DiodeLCPSolver(tolerance=1e-10)

    def initialize(self):
        # Initialize the electrical graph first
        if not self.electrical_graph.initialized:
            self.electrical_graph.initialize()
        
        # Find all circuit variables
        self.state_vars = self.find_state_vars()
        self.output_vars = self.find_output_vars()
        self.input_vars = self.find_input_vars()
        
        # Get graph structure from electrical_graph
        self.junction_voltage_var_list = self.electrical_graph.junction_voltage_var_list
        self.component_current_var_list = self.electrical_graph.component_current_var_list
        self.component_voltage_var_list = self.electrical_graph.component_voltage_var_list
        self.incidence_matrix = self.electrical_graph.incidence_matrix
        self.switch_list = self.electrical_graph.switch_list
        self.diode_list = self.electrical_graph.diode_list
        
        # Compute circuit equations
        self.kcl_eqs = self.compute_kcl_equations()
        self.kvl_eqs = self.compute_kvl_equations()
        self.static_eqs = self.compute_static_component_equations()
        self.switch_eqs = self.compute_switch_equations()
        self.diode_eqs = self.compute_diode_equations()
        self.circuit_eqs = self.compute_circuit_equations()
        self.derivatives = self.compute_derivatives()
        self.output_eqs = self.compute_output_equations()
        self.initialized = True

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
            _, comp_current_vars, _ = self.electrical_graph.variable_lists()
            incidence_matrix = self.electrical_graph.compute_incidence_matrix()
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
            junction_voltage_var_list, _, comp_voltage_vars = self.electrical_graph.variable_lists()
            incidence_matrix = self.electrical_graph.compute_incidence_matrix()
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
            switch_list = self.electrical_graph.find_switches()
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
            diode_list = self.electrical_graph.find_diodes()
        else:
            diode_list = self.diode_list
        
        diode_eqs = []
        for diode in diode_list:
            diode_eqs.append(diode.get_comp_eq())

        return diode_eqs
    
    def compute_circuit_equations(self) -> List[str]:
        """Solve the circuit variables.
        
        Returns:
            List[str]: List of circuit variables in symbolic form.
        """
        # for testing purposes
        if self.initialized == False:
            input_vars = self.find_input_vars()
            output_vars = self.find_output_vars()
            state_vars = self.find_state_vars()
            kcl_eqs = self.compute_kcl_equations()
            kvl_eqs = self.compute_kvl_equations()
            static_eqs = self.compute_static_component_equations()
            switch_eqs = self.compute_switch_equations()
            junction_voltage_var_list, component_current_var_list, component_voltage_var_list = self.electrical_graph.variable_lists()
        else:
            input_vars = self.input_vars
            output_vars = self.output_vars
            state_vars = self.state_vars
            kcl_eqs = self.kcl_eqs
            kvl_eqs = self.kvl_eqs
            static_eqs = self.static_eqs
            switch_eqs = self.switch_eqs
            junction_voltage_var_list = self.junction_voltage_var_list
            component_current_var_list = self.component_current_var_list
            component_voltage_var_list = self.component_voltage_var_list
        
        logging.debug(f"input_vars: {input_vars}")
        logging.debug(f"output_vars: {output_vars}")
        logging.debug(f"state_vars: {state_vars}")

        # Get diode equations when initialized
        if self.initialized == False:
            diode_eqs = self.compute_diode_equations()
        else:
            diode_eqs = self.diode_eqs
        
        # Combine all equations
        all_eqs = kcl_eqs + kvl_eqs + static_eqs + switch_eqs + diode_eqs
        # Find vars to solve for:
        # Remove 0 (ground node) from junction_voltage_var_list
        junction_voltage_var_list_cleaned = [var for var in junction_voltage_var_list if var != 0]
        # Combine all variables
        combined_vars = junction_voltage_var_list_cleaned + component_current_var_list + component_voltage_var_list
        # Remove input_vars and state_vars
        excluded = set(input_vars) | set(state_vars)
        all_vars = [var for var in combined_vars if var not in excluded]

        logging.debug(f"all_eqs: {all_eqs}")
        logging.debug(f"all_vars: {all_vars}")

        number_of_equations = len(all_eqs)
        number_of_variables = len(all_vars)
        logging.debug(f"number of eqs: {number_of_equations}")
        logging.debug(f"number of vars: {number_of_variables}")

        if number_of_equations != number_of_variables:
            raise Warning("The number of equations and variables must be the same. (%d equations, %d variables)" % (number_of_equations, number_of_variables))
        
        # Solve the equations
        solution = solve(all_eqs, all_vars)
        number_of_solutions = len(solution)
        logging.debug(f"number of sols: {number_of_solutions}")
        logging.debug(f"solutions: {solution}")

        if number_of_solutions != number_of_variables:
            raise Warning("Did not find a solution for every variable. (%d solutions, %d variables)" % (number_of_solutions, number_of_variables))
        
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
        assert self.initialized, "Model must be initialized before computing LCP matrices"
        
        if not self.diode_list:
            # No diodes in circuit, return empty matrices
            return Matrix([]), Matrix([])
        
        # Get diode current variables that we'll exclude from solving
        diode_current_vars = [diode.current_var for diode in self.diode_list]
        
        # Get all equations EXCEPT diode equations (since we're solving for diode voltages)
        kcl_eqs = self.kcl_eqs
        kvl_eqs = self.kvl_eqs 
        static_eqs = self.static_eqs  # Already excludes diodes
        switch_eqs = self.switch_eqs
        
        # Combine equations without diode equations - this is the key for LCP
        all_eqs = kcl_eqs + kvl_eqs + static_eqs + switch_eqs
        
        # Get all variables, excluding diode currents, state vars, and input vars
        junction_voltage_var_list_cleaned = [var for var in self.junction_voltage_var_list if var != 0]
        combined_vars = junction_voltage_var_list_cleaned + self.component_current_var_list + self.component_voltage_var_list
        
        # Exclude: input_vars, state_vars, and diode_current_vars
        excluded = set(self.input_vars) | set(self.state_vars) | set(diode_current_vars)
        vars_to_solve = [var for var in combined_vars if var not in excluded]
        
        logging.debug(f"LCP: Excluding {len(excluded)} variables (state/input/diode currents)")
        logging.debug(f"LCP: Solving for {len(vars_to_solve)} variables")
        logging.debug(f"LCP: Diode currents: {[str(var) for var in diode_current_vars]}")
        
        # Check equation/variable balance
        if len(all_eqs) != len(vars_to_solve):
            raise ValueError(f"LCP: Equation/variable mismatch: {len(all_eqs)} equations, {len(vars_to_solve)} variables")
        
        # Solve symbolically for all variables except diode currents
        try:
            solution = solve(all_eqs, vars_to_solve)
            logging.debug(f"LCP: Found {len(solution)} solutions")
        except Exception as e:
            raise ValueError(f"LCP: Failed to solve circuit equations: {e}")
        
        # Extract diode voltage expressions
        diode_voltage_vars = [diode.voltage_var for diode in self.diode_list]
        diode_voltage_exprs = []
        
        for voltage_var in diode_voltage_vars:
            if voltage_var in solution:
                # Extract -v_D expression for LCP formulation (positive = blocking)
                expr = -solution[voltage_var]  # Apply -1 factor for correct LCP signs
                diode_voltage_exprs.append(expr)
            else:
                raise ValueError(f"LCP: Could not find solution for diode voltage {voltage_var}")
        
        # Create substitution dictionary for current state and input values
        state_substitutions = {}
        input_substitutions = {}
        
        # Map state variables to their current values
        for i, state_var in enumerate(self.state_vars):
            if i < len(state_values):
                state_substitutions[state_var] = state_values[i]
        
        # Map input variables to their current values  
        for i, input_var in enumerate(self.input_vars):
            if i < len(input_values):
                input_substitutions[input_var] = input_values[i]
        
        # Combine substitutions
        substitutions = {**state_substitutions, **input_substitutions}
        
        # Extract M matrix and q vector by collecting coefficients
        n_diodes = len(self.diode_list)
        M_matrix = Matrix.zeros(n_diodes, n_diodes)
        q_vector = Matrix.zeros(n_diodes, 1)
        
        for i, expr in enumerate(diode_voltage_exprs):
            # Substitute state and input values
            expr_substituted = expr.subs(substitutions)
            
            # Extract coefficients for each diode current
            for j, diode_current_var in enumerate(diode_current_vars):
                # Get coefficient of this diode current
                coeff = expr_substituted.coeff(diode_current_var, 1)
                if coeff is not None:
                    M_matrix[i, j] = coeff
            
            # Get constant term (coefficient of diode currents set to 0)
            constant_term = expr_substituted.subs({var: 0 for var in diode_current_vars})
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
        
        # For now, use a simple LCP-based detection
        # In the future, this can be replaced with more sophisticated solvers
        return self._detect_diode_states_lcp(state_values, input_values)
    
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