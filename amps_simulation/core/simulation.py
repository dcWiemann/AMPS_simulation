import sympy as sp
import logging
from typing import Dict, Set, Tuple, List, Any
from amps_simulation.core.electrical_model import ElectricalModel
from scipy.integrate import solve_ivp
import numpy as np

class Simulation:
    """
    Class for handling circuit simulation.
    
    This class takes circuit components and electrical nodes from a ParserJson instance
    and assigns all necessary variables for simulation.
    """
    
    def __init__(self, electrical_nodes: Dict[int, Set[Tuple[str, str]]], 
                 circuit_components: Dict[str, Dict]):
        """
        Initialize the Simulation class.
        
        Args:
            electrical_nodes: { electrical_node_id: set((component_id, terminal_id), ...) }
            circuit_components: { component_id: { "type": str, "value": float, "terminals": {terminal_id: electrical_node} } }
        """
        self.electrical_nodes = electrical_nodes
        self.circuit_components = circuit_components
        self.power_switches = []
        self.voltage_vars = {}
        self.current_vars = {}
        self.state_vars = {}
        self.state_derivatives = {}
        self.input_vars = {}
        self.ground_node = None
        
    def initialize(self):
        """
        Initialize all variables needed for simulation.
        This method should be called before running any simulation.
        
        Returns:
            self: Returns self for method chaining
        """
        # First assign voltage and current variables
        self.voltage_vars, self.ground_node = self._assign_voltage_variables()
        self.current_vars = self._assign_current_variables()
        
        # Then extract state and input variables
        self.state_vars, self.state_derivatives, self.input_vars = self._extract_input_and_state_vars()
        self.power_switches = self._extract_power_switches()
        logging.info(f"✅ State variables: {self.state_vars}, ✅ Input variables: {self.input_vars}")
        logging.info(f"✅ Power switches: {self.power_switches}")
        
        return self
        
    def simulate(self, t_span=(0, 10)):
        """
        Run a simulation of the circuit using the already assigned variables and matrices.
        This method assumes that the simulation instance has already been initialized
        with electrical nodes, circuit components, and all necessary variables.
        
        Args:
            t_span: Tuple (t_start, t_end) defining the time range
        
        Returns:
            - t: Time points from simulation
            - x: State variable trajectories over time
            - y: Output trajectories over time
        """
        # Ensure variables are initialized
        if not self.voltage_vars:
            self.initialize()
        
        control_signals = self._get_control_signals()
        logging.info("👾 Control signals: %s", control_signals)
        for control_signal in control_signals:
            logging.info("👾 Control signal at t=1: %s", control_signal(2))

        # # Step 1: Create electrical model and extract matrices
        number_of_piecewise_linear_models, possible_switch_positions = self._get_switch_positions()
        logging.info("👾 Number of piecewise linear models: %s", number_of_piecewise_linear_models)
        logging.info("👾 Possible switch positions: %s", possible_switch_positions)
        switch_positions_models = {}
        for i in range(number_of_piecewise_linear_models):
            switch_position = possible_switch_positions[i]
            model = self.create_model(switch_position)
            # create a hashmap of switch positions and models
            switch_positions_models[switch_position] = model
            logging.info("🎒 Model %s: %s", i, model)

        # Step 2: Substitute component values in A and B matrices
        component_values = self._get_component_values()
        A_symbolic, B_symbolic = self.extract_state_space_matrices(model.differential_equations)
        logging.info("👾 Symbolic State matrix A: %s", A_symbolic)
        logging.info("👾 Symbolic Input matrix B: %s", B_symbolic)

        A = self.substitute_component_values(A_symbolic, component_values)
        B = self.substitute_component_values(B_symbolic, component_values)
        logging.info("👾 Substituted state matrix A: %s", A)
        logging.info("👾 Substituted input matrix B: %s", B)
        
        # Define identity output matrix C (observing all state variables)
        C = np.eye(A.shape[0])  # Identity matrix of size (states x states)
        
        # Define initial conditions
        initial_conditions = np.zeros(A.shape[0])  # Zero initial state
        
        # Create input function
        input_function = self.create_input_function()
        
        # Run simulation
        t, x, y = self.run_solver(A, B, C, t_span, initial_conditions, input_function, control_signals)
        logging.info("✅ Time points: %s", t[0:10])  # Log first 10 time points
        logging.info("✅ Simulation completed.")
        
        return t, x, y
    

    def _get_component_values(self) -> List[Dict[str, Any]]:
        """
        Get component values from circuit components.

        Returns:
            - component_values: List of dictionaries containing component ID and value
        """
        component_values = []
        for comp_id, comp_data in self.circuit_components.items():
            if "value" in comp_data:
                component_values.append({
                    "id": comp_id,
                    "data": {"value": comp_data["value"]}
                })
        return component_values

    def _assign_voltage_variables(self) -> Tuple[Dict[int, sp.Symbol], int]:
        """
        Assigns voltage variables to electrical nodes.
        If a ground node exists in the components, it is used as the reference (0V).
        
        Returns:
            - voltage_vars: { electrical_node_id: voltage_variable }
            - ground_node: The chosen ground node (for reference voltage)
        """
        voltage_vars = {}
        
        # Step 1: Find a ground node in circuit_components (if it exists)
        ground_node = None
        for comp_id, comp in self.circuit_components.items():
            if comp["type"] == "ground":  # Ground components must have only 1 terminal
                if len(comp["terminals"]) == 1:
                    ground_node = next(iter(comp["terminals"].values()))  # Get the electrical node ID
                    break  # Use the first found ground
        
        # Step 2: If no explicit ground found, default to the lowest node ID
        if ground_node is None:
            ground_node = min(self.electrical_nodes.keys())
        
        # Step 3: Assign voltage variables
        for node_id in self.electrical_nodes:
            if node_id == ground_node:
                voltage_vars[node_id] = 0  # Ground node has 0V
            else:
                voltage_vars[node_id] = sp.Symbol(f"V_{node_id}")  # Symbolic voltage variable
        
        return voltage_vars, ground_node
    
    def _assign_current_variables(self) -> Dict[str, sp.Symbol]:
        """
        Assigns current variables to circuit components based on the cleaned format.
        
        Returns:
            - current_vars: { component_id: current_variable }
        """
        current_vars = {}
        
        for comp_id, comp in self.circuit_components.items():
            comp_type = comp["type"]
            
            # Voltage sources enforce their own current, so we don't assign them
            if comp_type not in ["voltage-source", "ground"]:
                current_vars[comp_id] = sp.Symbol(f"I_{comp_id}")  # Assign symbolic current variable
        
        return current_vars
    
    def _extract_input_and_state_vars(self) -> Tuple[Dict[sp.Symbol, str], Dict[sp.Symbol, sp.Expr], Dict[sp.Symbol, str]]:
        """
        Extracts input and state variables, defining helper variables for clarity.
        
        Returns:
            - state_vars: Dictionary of helper state variables and their component IDs
            - state_derivatives: Dictionary of state derivatives
            - input_vars: Dictionary of helper input variables and their component IDs
        """
        state_vars = {}
        input_vars = {}
        state_derivatives = {}
        
        for comp_id, comp_data in self.circuit_components.items():
            comp_type = comp_data["type"]
            terminals = comp_data["terminals"]
            
            if comp_type == "capacitor":
                # Capacitor voltage state variable
                if len(terminals) == 2:
                    node_a, node_b = terminals.values()
                    v_a = self.voltage_vars.get(node_a, sp.Symbol(f"V_{node_a}"))
                    v_b = self.voltage_vars.get(node_b, sp.Symbol(f"V_{node_b}"))
                    
                    v_C = sp.Symbol(f"V_{comp_id}")  # Helper variable for capacitor voltage
                    i_C = self.current_vars.get(comp_id)  # Current through the capacitor
                    C_value = sp.Symbol(f"{comp_id}_value")  # Capacitance value as a symbol
                    state_vars[v_C] = comp_id  # Store component ID instead of expression
                    state_derivatives[sp.Derivative(v_C, 't')] = i_C/C_value  # dV/dt = i_C/C
            
            elif comp_type == "inductor":
                # Inductor current state variable
                if len(terminals) == 2:
                    node_a, node_b = terminals.values()
                    v_a = self.voltage_vars.get(node_a, sp.Symbol(f"V_{node_a}"))
                    v_b = self.voltage_vars.get(node_b, sp.Symbol(f"V_{node_b}"))
                    
                    if comp_id in self.current_vars:
                        i_L = self.current_vars[comp_id]  # Current through the inductor
                        L_value = sp.Symbol(f"{comp_id}_value")  # Inductance value as a symbol
                        state_vars[i_L] = comp_id  # Store component ID instead of expression
                        state_derivatives[sp.Derivative(i_L, 't')] = (v_a - v_b)/L_value  # di/dt = (V_A - V_B)/L
            
            elif comp_type == "voltage-source":
                # Voltage source input variable
                if len(terminals) == 2:
                    V_source = sp.Symbol(f"V_in_{comp_id}")  # Helper variable for voltage source
                    input_vars[V_source] = comp_id  # Store component ID instead of expression
            
            elif comp_type == "current-source":
                # Current source input variable
                if comp_id in self.current_vars:
                    I_source = sp.Symbol(f"I_in_{comp_id}")  # Helper variable for current source
                    input_vars[I_source] = comp_id  # Store component ID instead of expression
        
        return state_vars, state_derivatives, input_vars
    

    def _extract_power_switches(self) -> List[str]:
        """
        Extracts power switches from the circuit components.
        
        Returns:
            List[str]: A list containing the IDs of all power switches in the circuit
        """
        self.power_switches = [
            comp_id for comp_id, comp_data in self.circuit_components.items()
            if comp_data["type"] == "switch"
        ]
        return self.power_switches
    
    def _get_control_signals(self):
        """
        Get control signals from the circuit components.
        
        Returns:
            control_signals: list of functions of time
        """
        control_signals = []
        # switch times: {switch_id: [times]}
        switch_times = {}
        for switch_id in self.power_switches:
            switch_times[switch_id] = self.circuit_components[switch_id]["value"]
        logging.info("👾 Switch times: %s", switch_times)
        
        def create_control_signal(switch_id, switch_time):
            def control_signal(t):
                if t < switch_time:
                    return 0  # Switch is OFF
                else:
                    return 1  # Switch is ON
            return control_signal
            
        for switch_id in self.power_switches:
            control_signals.append(create_control_signal(switch_id, switch_times[switch_id]))
            
        return control_signals
    
    def _get_switch_positions(self):
        """
        Get the switch positions from the circuit components.

        Returns:
            number_of_piecewise_linear_models: int
            switch_positions: list of bitstrings
        """
        number_of_piecewise_linear_models = 2**len(self.power_switches)
        switch_positions = [bin(i)[2:].zfill(len(self.power_switches)) for i in range(number_of_piecewise_linear_models)]
        return number_of_piecewise_linear_models, switch_positions

    def create_model(self, switch_position):
        """
        Extract differential equations from the circuit and convert to state space form.
        
        Args:
            components: List of circuit components from JSON.
            
        Returns:
            Tuple containing:
            - A_substituted: State matrix with numerical values
            - B_substituted: Input matrix with numerical values
            - state_vars: Dictionary of state variables
            - input_vars: Dictionary of input variables
        """
        # Create ElectricalModel instance and build model
        model = ElectricalModel(self.electrical_nodes, self.circuit_components, self.voltage_vars, 
                              self.current_vars, self.state_vars, self.state_derivatives, 
                              self.input_vars, self.ground_node, self.power_switches, switch_position)
        solved_helpers, differential_equations = model.build_model()
        return model


    def extract_state_space_matrices(self, differential_equations) -> Tuple[sp.Matrix, sp.Matrix]:
        """
        Converts the state derivative dictionary into matrix form and computes Jacobians for A and B.
        
        Returns:
            - A: State matrix (Jacobian of dx/dt w.r.t. state variables).
            - B: Input matrix (Jacobian of dx/dt w.r.t. input variables).
        """
        # Convert dictionaries to lists of variables
        state_vars = list(self.state_vars.keys())  
        input_vars = list(self.input_vars.keys())  

        # Convert to matrix form
        dx_dt_sol = sp.Matrix(list(differential_equations.values()))

        # Compute Jacobians
        A = dx_dt_sol.jacobian(state_vars)  # Partial derivatives of dx/dt w.r.t. state variables
        B = dx_dt_sol.jacobian(input_vars)  # Partial derivatives of dx/dt w.r.t. input variables

        return A, B


    def substitute_component_values(self, expr, components):
        """
        Substitutes numerical values for component parameters in the symbolic equation.
        
        Args:
            expr: SymPy expression or matrix containing symbolic component parameters.
            components: List of circuit components from JSON.
            
        Returns:
            - expr_substituted: Expression with component values replaced.
        """
        subs_dict = {}

        for component in components:
            comp_id = component["id"]
            value = component["data"].get("value")  # Get numerical value
            if value is not None:
                subs_dict[sp.Symbol(f"{comp_id}_value")] = value  # Replace symbol with actual value

        return expr.subs(subs_dict)
        

    def create_input_function(self):
        """
        Creates a DC input function for the simulation that returns the values of the sources.
        
        Returns:
            - input_function: A callable function that takes time t and returns input values
        """
        # Create a mapping from input variables to their source values
        input_values = {}
        
        # Extract source values from circuit components
        for input_var, comp_id in self.input_vars.items():
            if comp_id in self.circuit_components:
                comp_data = self.circuit_components[comp_id]
                if "value" in comp_data:
                    input_values[input_var] = comp_data["value"]
                else:
                    # Default to 0 if no value is specified
                    input_values[input_var] = 0.0
            else:
                # Default to 0 if component not found
                input_values[input_var] = 0.0
        
        # Create a list of input variables in a consistent order
        input_vars_list = list(self.input_vars.keys())
        
        def DC_input_function(t):
            # Return the source values for t >= 0
            if t >= 0:
                return np.array([input_values[var] for var in input_vars_list])
            else:
                # Return zeros for t < 0
                return np.zeros(len(input_vars_list))
        
        return DC_input_function
        

    def run_solver(self, A, B, C, t_span, initial_conditions, input_function, control_signals):
        """
        Numerically solves the ODE system dx/dt = Ax + Bu using solve_ivp.

        Parameters:
        - A: State matrix (numpy array after substitution).
        - B: Input matrix (numpy array after substitution).
        - C: Output matrix (numpy array).
        - t_span: Tuple (t_start, t_end) defining the time range.
        - initial_conditions: Initial state vector (same size as state variables).
        - input_function: Function u(t) defining the input voltage/current.

        Returns:
        - t: Time points from simulation.
        - x: State variable trajectories over time.
        - y: Output trajectories over time.
        """
        # Convert A and B to numerical arrays (ensure float type)
        A_func = np.array(A).astype(float)
        B_func = np.array(B).astype(float)
        C_func = np.array(C).astype(float)

        # Define time points at fixed 0.1s intervals
        t_eval = np.arange(t_span[0], t_span[1], 0.1)  # Time points at 0.1s resolution

        # Define ODE system
        def state_space_ode(t, x):
            u = input_function(t)
            return (A_func @ x) + (B_func @ u)  # dx/dt = Ax + Bu

        # Define event functions for each switch
        switch_events = []
        for i, control_signal in enumerate(control_signals):
            def create_switch_event(switch_time):
                def switch_event(t, x):
                    return t - switch_time
                switch_event.terminal = True
                switch_event.direction = 0  # Detect both positive and negative crossings
                return switch_event
            
            # Get the switch time from the circuit components
            switch_id = self.power_switches[i]
            switch_time = self.circuit_components[switch_id]["value"]
            switch_events.append(create_switch_event(switch_time))

        # Solve ODE system
        sol = solve_ivp(state_space_ode, t_span, initial_conditions, method="RK45",
                       events=switch_events)

        # Compute output y = Cx
        y = C_func @ sol.y

        return sol.t, sol.y, y 