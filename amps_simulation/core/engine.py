import logging
from typing import List, Optional, Union
import networkx as nx
from scipy.integrate import solve_ivp
import numpy as np
from .components import Source
from .dae_system import ElectricalDaeSystem
from .electrical_model import ElectricalModel
from .engine_settings import EngineSettings
from .control_model import ControlModel

class Engine:
    """
    Class for handling circuit simulation using an ElectricalModel.

    This class takes an ElectricalModel and handles the simulation of the circuit.
    """
    
    def __init__(self, electrical_model: ElectricalModel, control_model: Optional[Union[ControlModel, nx.MultiDiGraph]] = None):
        """
        Initialize the Engine class.

        Args:
            electrical_model: ElectricalModel representing the circuit
            control_model: ControlModel representing the control layer
        """
        self.electrical_model = electrical_model
        self.graph = electrical_model.graph  # Keep reference to graph for compatibility
        if isinstance(control_model, nx.MultiDiGraph):
            self.control_model = ControlModel(control_model)
        else:
            self.control_model = control_model or ControlModel()
        
        # Initialize simulation variables
        self.components_list = []
        self.state_vars = {}  # Dictionary of state variables
        self.input_vars = {}  # Dictionary of input variables
        self.switch_list = None  # Tuple of power switches
        self.diode_list = None   # Tuple of diodes
        self.switch_events = None  # List of switch events
        # Simulation settings
        self.engine_settings = EngineSettings()
        # Ground node reference
        self.ground_node = None
        self.initialized = False

    def initialize(self, initial_conditions=None, initial_inputs=None) -> None:
        """
        Initialize all variables needed for simulation.
        This method should be called before running any simulation.
        
        Args:
            initial_conditions: Initial state values for diode state detection
            initial_inputs: Initial input values for diode state detection
        """
        
        # Create components list from edge data
        self.components_list = []
        for _, _, edge_data in self.graph.edges(data=True):
            component = edge_data.get('component')
            if component is not None:
                self.components_list.append(component)

        # Create DAE system from electrical model
        self.electrical_dae_system = ElectricalDaeSystem(self.electrical_model)
        self.electrical_dae_system.initialize(initial_conditions, initial_inputs)

        # Set up all necessary variables
        self.state_vars = tuple(self.electrical_dae_system.state_vars)
        self.input_vars = tuple(self.electrical_dae_system.input_vars)
        self.output_vars = tuple(self.electrical_dae_system.output_vars)
        self.switch_list = tuple(self.electrical_model.switch_list)
        self.diode_list = tuple(self.electrical_model.diode_list)
        
        # Build control model input function for source inputs.
        # Mapping from electrical input_vars -> control port nodes is carried on Source.control_port_name.
        if self.input_vars:
            port_order: List[str] = []
            for input_var in self.input_vars:
                for component in self.components_list:
                    if not isinstance(component, Source):
                        continue
                    if component.input_var != input_var:
                        continue
                    if not component.control_port_name:
                        continue
                    port_order.append(component.control_port_name)
                    break

            if port_order:
                self.control_input_function = self.control_model.compile_input_function(port_order)
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

    def run_simulation(self, t_span, initial_conditions=None, method='RK45', **kwargs):
        """Run circuit simulation using solve_ivp with proper switch handling.
        
        This method handles switching circuits by:
        1. Generating input function from control orchestrator
        2. Checking switch combinations and computing DAE systems as needed
        3. Creating callable functions for solve_ivp that return dx/dt
        4. Delegating switch/diode handling and caching to ElectricalDaeSystem
        5. Using switch events to interrupt simulation when switches change
        6. Generating outputs using out = Cx + Du
        
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
            
        # Create switch events for interrupting simulation
        events = None
        if self.switch_list and self.switch_events:
            events = self.switch_events
            logging.debug(f"Using {len(events)} switch events")
            
        # Apply engine settings to solver parameters
        solver_kwargs = {
            'max_step': self.engine_settings.max_step_size,
            'rtol': self.engine_settings.rel_tol,
            'atol': self.engine_settings.abs_tol,
        }
        
        # Add initial step size if specified
        if self.engine_settings.init_step_size is not None:
            solver_kwargs['first_step'] = self.engine_settings.init_step_size
            
        # Merge with any user-provided kwargs (user kwargs take precedence)
        solver_kwargs.update(kwargs)
        
        # Run simulation with event handling
        solution = solve_ivp(
            fun=lambda t, x: self._get_ode_function_for_time(t, x, input_function),
            t_span=t_span,
            y0=current_state,
            method=method,
            events=events,
            **solver_kwargs
        )
        
        if not solution.success:
            logging.error(f"Integration failed: {solution.message}")
            return {
                't': solution.t,
                'y': solution.y,
                'out': None,
                'success': False,
                'message': solution.message,
                'switchmap_size': len(self.electrical_dae_system.switchmap)
            }
        
        # Generate outputs using out = Cy + Du
        if len(self.output_vars) > 0:
            # Get the final switch and diode combination to determine C and D matrices
            final_time = solution.t[-1]
            final_state = solution.y[:, -1]
            final_input = input_function(final_time)
            
            if self.switch_list:
                final_switch_states = [
                    switch.set_switch_state(final_time) for switch in self.switch_list
                ]
            else:
                final_switch_states = []

            A, B, C, D = self.electrical_dae_system.update_ode(
                final_switch_states, final_state, final_input
            )
            
            # Compute outputs: out = Cy + Du
            y_out = []
            for i, t_val in enumerate(solution.t):
                x_val = solution.y[:, i].reshape(-1, 1)
                u_val = input_function(t_val).reshape(-1, 1)
                out_val = C @ x_val + D @ u_val
                y_out.append(out_val.flatten())
                
            outputs = np.array(y_out).T
        else:
            outputs = None
            
        result = {
            't': solution.t,
            'y': solution.y,
            'out': outputs,
            'success': True,
            'message': 'Simulation completed successfully',
            'switchmap_size': len(self.electrical_dae_system.switchmap),
            'nfev': solution.nfev if hasattr(solution, 'nfev') else None,
            't_events': solution.t_events if hasattr(solution, 't_events') else None
        }
        
        return result
    
    def _get_ode_function_for_time(self, t, x, input_function):
        """Get ODE function for given time t, checking switch and diode states and using cache.
        
        Args:
            t: Current time
            x: Current state vector
            input_function: Function to get input vector u(t)
            
        Returns:
            dx/dt: State derivative vector for solve_ivp
        """
        if len(self.state_vars) == 0:
            return np.array([])

        if self.switch_list:
            current_switch_states = [
                switch.set_switch_state(t) for switch in self.switch_list
            ]
        else:
            current_switch_states = []
        
        u = input_function(t)
        A, B, _, _ = self.electrical_dae_system.update_ode(current_switch_states, x, u)

        x_col = x.reshape(-1, 1) if len(x.shape) == 1 else x
        u_col = u.reshape(-1, 1) if len(u.shape) == 1 else u
        dx_dt = A @ x_col + B @ u_col
        return dx_dt.flatten()
