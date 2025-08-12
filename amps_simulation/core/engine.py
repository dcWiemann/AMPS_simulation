import sympy as sp
import logging
from typing import Dict, Set, Tuple, List, Any
import networkx as nx
from scipy.integrate import solve_ivp
import numpy as np
from .components import Component
from .dae_model import ElectricalDaeModel
from .engine_settings import EngineSettings
from .control_orchestrator import ControlOrchestrator, ControlGraph

class Engine:
    """
    Class for handling circuit simulation using a NetworkX graph structure.
    
    This class takes a NetworkX graph and handles the simulation of the circuit.
    """
    
    def __init__(self, graph: nx.MultiDiGraph, control_graph: ControlGraph = None):
        """
        Initialize the Engine class.
        
        Args:
            graph: NetworkX MultiDiGraph representing the circuit
            control_graph: ControlGraph representing the control layer
        """
        self.graph = graph
        self.control_graph = control_graph or ControlGraph()
        self.control_orchestrator = ControlOrchestrator(self.control_graph)
        
        # Initialize simulation variables
        self.components_list = []
        self.state_vars = {}  # Dictionary of state variables
        self.input_vars = {}  # Dictionary of input variables
        self.switch_list = None  # Tuple of power switches
        self.switch_events = None  # List of switch events
        # Simulation settings
        self.engine_settings = EngineSettings()
        # Ground node reference
        self.ground_node = None
        self.initialized = False

    def initialize(self) -> None:
        """
        Initialize all variables needed for simulation.
        This method should be called before running any simulation.
        """
        # Create components list from edge data
        self.components_list = []
        for _, _, edge_data in self.graph.edges(data=True):
            component = edge_data.get('component')
            if component is not None:
                self.components_list.append(component)
        
        self.electrical_model = ElectricalDaeModel(self.graph)
        self.electrical_model.initialize()

        # Set up all necessary variables
        self.state_vars = tuple(self.electrical_model.state_vars)
        self.input_vars = tuple(self.electrical_model.input_vars)
        self.output_vars = tuple(self.electrical_model.output_vars)
        self.switch_list = tuple(self.electrical_model.switch_list)
        
        # Build control orchestrator input function for sources only
        source_ports = self.control_graph.get_source_ports()
        if source_ports:
            # Create ordered list of SOURCE port names matching input_vars order
            # Engine determines and maintains this order throughout simulation
            port_order = []
            for input_var in self.input_vars:
                # Find the SOURCE control port that corresponds to this input variable
                for port_name, port in source_ports.items():
                    if port.variable == input_var:
                        port_order.append(port_name)
                        break
            
            if port_order:
                self.control_input_function = self.control_orchestrator.compile_input_function(port_order)
                logging.debug(f"Control input function compiled with port order: {port_order}")

        if self.switch_list is not None:
            self.switch_events = self._get_switch_events()

        # Set initialized flag to True
        self.initialized = True

    def _get_switch_events(self):
        """
        Creates event functions for each switch in the circuit.
        
        Returns:
            List[callable]: A list of event functions that return 0 when a switch changes state.
        """


        def create_event_function(switch_time):
            def event(t, x):
                return t - switch_time
            event.terminal = False  # Allow simulation to continue past switch events
            event.direction = 1  # Detect only positive crossings (from negative to positive)
            return event

        # Create a list of event functions for each switch
        switch_events = [create_event_function(switch.switch_time) for switch in self.switch_list]
        
        return switch_events
    
    # def _get_possible_switch_positions(self):
    #     """
    #     Get all possible switch positions for the circuit.
        
    #     Returns:
    #         List[Tuple[int, ...]]: A list of tuples representing the possible switch positions
    #     """
    #     return list(itertools.product([0, 1], repeat=len(self.power_switches)))

    def _sort_derivatives_by_state_vars(self, derivatives: List[sp.Eq]) -> List[sp.Eq]:
        """
        Sort derivatives to match the order of state variables.
        
        Args:
            derivatives: List of derivative equations
            
        Returns:
            List[sp.Eq]: Sorted list of derivative equations matching state_vars order
        """
        # Create a mapping of state variable to its derivative equation
        derivative_map = {}
        for eq in derivatives:
            # Get the state variable from the derivative
            state_var = eq.lhs.args[0]  # The variable being differentiated
            derivative_map[state_var] = eq
            
        # Create sorted list based on state_vars order
        sorted_derivatives = []
        for state_var in self.state_vars:
            if state_var in derivative_map:
                sorted_derivatives.append(derivative_map[state_var])
                
        return sorted_derivatives

    def _sort_output_eqs_by_output_vars(self, output_eqs) -> List[sp.Eq]:
        """
        Sort output equations to match the order of output variables.
        
        Args:
            output_eqs: Dictionary or list of output equations
            
        Returns:
            List[sp.Eq]: Sorted list of output equations matching output_vars order
        """
        # Handle dictionary format (output_var -> expression)
        if isinstance(output_eqs, dict):
            # Create sorted list based on output_vars order
            sorted_output_eqs = []
            for output_var in self.output_vars:
                if output_var in output_eqs:
                    # Create equation: output_var = expression
                    eq = sp.Eq(output_var, output_eqs[output_var])
                    sorted_output_eqs.append(eq)
            return sorted_output_eqs
            
        # Handle list format (assuming equations with lhs and rhs)
        else:
            output_map = {}
            for eq in output_eqs:
                # Get the output variable from the equation
                output_var = eq.lhs  # The output variable is on the left side
                output_map[output_var] = eq
                
            # Create sorted list based on output_vars order
            sorted_output_eqs = []
            for output_var in self.output_vars:
                if output_var in output_map:
                    sorted_output_eqs.append(output_map[output_var])
                    
            return sorted_output_eqs
    
    def compute_state_space_model(self, derivatives: List[sp.Eq], output_eqs: List[sp.Eq]) -> None:
        """
        Compute the state space model of the circuit.

        Args:
            derivatives: ordered list of derivative equations
            output_eqs: ordered list of output equations

        Returns:
            A, B, C, D: State space model matrices
        """
        n_states = len(self.state_vars)
        n_inputs = len(self.input_vars)
        n_outputs = len(self.output_vars)
        assert n_states == len(derivatives), "Number of state variables does not match number of derivatives"
        assert n_outputs == len(output_eqs), "Number of output variables does not match number of output equations"
        

        # Create dx_dt matrix from derivatives
        if derivatives:
            dx_dt = sp.Matrix([eq.rhs for eq in derivatives])
            A = dx_dt.jacobian(self.state_vars)
            B = dx_dt.jacobian(self.input_vars)
        else:
            A = sp.zeros(n_states, n_states)
            B = sp.zeros(n_states, n_inputs)

        # Create y matrix from output equations
        if output_eqs:
            y = sp.Matrix([eq.rhs for eq in output_eqs])
            C = y.jacobian(self.state_vars)
            D = y.jacobian(self.input_vars)
        else:
            C = sp.zeros(n_outputs, n_states)
            D = sp.zeros(n_outputs, n_inputs)

        return A, B, C, D
    
    def _create_state_space_ode(self, A, B, input_function):
        """Create ODE function for solve_ivp from state space matrices.
        
        Args:
            A: State matrix (symbolic or numeric)
            B: Input matrix (symbolic or numeric)  
            input_function: Function that returns input vector u(t)
            
        Returns:
            Callable function compatible with solve_ivp: f(t, x) -> dx/dt
        """
        # Convert symbolic matrices to numeric functions if needed
        if hasattr(A, 'subs'):  # Symbolic matrix
            A_func = sp.lambdify([], A, 'numpy')
            A_num = A_func()
        else:
            A_num = np.array(A, dtype=float)
            
        if hasattr(B, 'subs'):  # Symbolic matrix  
            B_func = sp.lambdify([], B, 'numpy')
            B_num = B_func()
        else:
            B_num = np.array(B, dtype=float)
        
        def ode_function(t, x):
            """ODE function: dx/dt = Ax + Bu"""
            u = input_function(t)
            if len(u.shape) == 1:
                u = u.reshape(-1, 1)
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
                
            dx_dt = A_num @ x + B_num @ u
            return dx_dt.flatten()
            
        return ode_function
    
    def run_simulation(self, t_span, initial_conditions=None, method='RK45', **kwargs):
        """Run circuit simulation using solve_ivp with proper switch handling.
        
        This method handles switching circuits by:
        1. Generating input function from control orchestrator
        2. Checking switch combinations and computing DAE models as needed
        3. Creating callable functions for solve_ivp that return dx/dt
        4. Using switchmap to cache models for different switch combinations
        5. Using switch events to interrupt simulation when switches change
        6. Generating outputs using out = Cy + Du
        
        Args:
            t_span: Tuple (t_start, t_end) for simulation time span
            initial_conditions: Initial state vector (defaults to zeros)
            method: Integration method for solve_ivp ('RK45', 'DOP853', etc.)
            **kwargs: Additional arguments passed to solve_ivp
            
        Returns:
            Dictionary with 't', 'y', 'out', and other solution information
        """
        if not self.initialized:
            raise ValueError("Engine must be initialized before running simulation")
            
        t_start, t_end = t_span
        
        # Set default initial conditions
        if initial_conditions is None:
            current_state = np.zeros(len(self.state_vars))
        else:
            current_state = np.array(initial_conditions)
            
        # Generate input function from control orchestrator
        if hasattr(self, 'control_input_function'):
            input_function = self.control_input_function
            logging.debug("Using control orchestrator input function")
        else:
            input_function = lambda t: np.zeros(len(self.input_vars))
            logging.debug("Using zero input function")
            
        # Create switchmap for caching different switch combinations
        switchmap = {}  # switch_combination -> (A, B, C, D, ode_function)
        
        # Create switch events for interrupting simulation
        events = None
        if self.switch_list and self.switch_events:
            events = self.switch_events
            logging.debug(f"Using {len(events)} switch events")
            
        # Run simulation with event handling
        solution = solve_ivp(
            fun=lambda t, y: self._get_ode_function_for_time(t, y, switchmap, input_function),
            t_span=t_span,
            y0=current_state,
            method=method,
            events=events,
            **kwargs
        )
        
        if not solution.success:
            logging.error(f"Integration failed: {solution.message}")
            return {
                't': solution.t,
                'y': solution.y,
                'out': None,
                'success': False,
                'message': solution.message,
                'switchmap_size': len(switchmap)
            }
        
        # Generate outputs using out = Cy + Du
        if len(self.output_vars) > 0:
            # Get the final switch combination to determine C and D matrices
            final_time = solution.t[-1]
            if self.switch_list:
                final_switch_states = tuple(
                    switch.set_switch_state(final_time) for switch in self.switch_list
                )
            else:
                final_switch_states = ()
                
            if final_switch_states in switchmap:
                A, B, C, D, _ = switchmap[final_switch_states]
            else:
                # Compute final state space model if not cached
                A, B, C, D = self._compute_state_space_for_switches(final_switch_states, final_time)
            
            # Convert symbolic matrices to numeric if needed
            if hasattr(C, 'subs'):
                C_func = sp.lambdify([], C, 'numpy')
                C_num = C_func()
            else:
                C_num = np.array(C, dtype=float)
                
            if hasattr(D, 'subs'):
                D_func = sp.lambdify([], D, 'numpy')
                D_num = D_func()
            else:
                D_num = np.array(D, dtype=float)
            
            # Compute outputs: out = Cy + Du
            outputs = []
            for i, t_val in enumerate(solution.t):
                y_val = solution.y[:, i].reshape(-1, 1)
                u_val = input_function(t_val).reshape(-1, 1)
                out_val = C_num @ y_val + D_num @ u_val
                outputs.append(out_val.flatten())
                
            outputs = np.array(outputs).T
        else:
            outputs = None
            
        result = {
            't': solution.t,
            'y': solution.y,
            'out': outputs,
            'success': True,
            'message': 'Simulation completed successfully',
            'switchmap_size': len(switchmap),
            'nfev': solution.nfev if hasattr(solution, 'nfev') else None,
            't_events': solution.t_events if hasattr(solution, 't_events') else None
        }
        
        return result
    
    def _get_ode_function_for_time(self, t, y, switchmap, input_function):
        """Get ODE function for given time t, checking switch states and using cache.
        
        Args:
            t: Current time
            y: Current state vector
            switchmap: Dictionary caching switch combinations -> (A, B, C, D, ode_func)
            input_function: Function to get input vector u(t)
            
        Returns:
            dy/dt: State derivative vector for solve_ivp
        """
        # Determine current switch states
        if self.switch_list:
            # Use exact time without offset to test numerical stability
            current_switch_states = tuple(
                switch.set_switch_state(t) for switch in self.switch_list
            )
        else:
            current_switch_states = ()
            
        # Check if this switch combination is already cached
        if current_switch_states in switchmap:
            A, B, C, D, ode_function = switchmap[current_switch_states]
            # logging.debug(f"Using cached ODE function for switch states: {current_switch_states}")
        else:
            # Compute new DAE model for this switch combination
            A, B, C, D = self._compute_state_space_for_switches(current_switch_states, t)
            
            # Create ODE function from state space matrices
            ode_function = self._create_state_space_ode(A, B, input_function)
            
            # Cache the result
            switchmap[current_switch_states] = (A, B, C, D, ode_function)
            logging.debug(f"Computed new DAE model for switch states: {current_switch_states}")
            
        return ode_function(t, y)
    
    def _compute_state_space_for_switches(self, switch_states, t):
        """Compute state space matrices for given switch combination.
        
        Args:
            switch_states: Tuple of switch states (True/False for each switch)
            t: Current time
            
        Returns:
            A, B, C, D: State space matrices
        """
        # Update switch states in electrical model
        if self.switch_list:
            self.electrical_model.update_switch_states(t)
        
        # Get derivatives and output equations from DAE model
        derivatives = self.electrical_model.derivatives
        output_eqs = self.electrical_model.output_eqs
        
        # Sort equations to match variable order
        sorted_derivatives = self._sort_derivatives_by_state_vars(derivatives)
        sorted_output_eqs = self._sort_output_eqs_by_output_vars(output_eqs)
        
        # Compute state space matrices
        A, B, C, D = self.compute_state_space_model(sorted_derivatives, sorted_output_eqs)
        
        return A, B, C, D
