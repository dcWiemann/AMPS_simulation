import sympy as sp
import logging
from typing import Dict, Set, Tuple, List, Any
from amps_simulation.core.electrical_model import ElectricalModel

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
        self.voltage_vars = {}
        self.current_vars = {}
        self.state_vars = {}
        self.state_derivatives = {}
        self.input_vars = {}
        self.ground_node = None
        
    def assign_variables(self):
        """
        Assign all necessary variables for simulation.
        
        Returns:
            Tuple containing:
            - voltage_vars: Dictionary of voltage variables
            - current_vars: Dictionary of current variables
            - state_vars: Dictionary of state variables
            - state_derivatives: Dictionary of state derivatives
            - input_vars: Dictionary of input variables
            - ground_node: The ground node ID
        """
        # Assign voltage variables
        self.voltage_vars, self.ground_node = self._assign_voltage_variables()
        logging.info("✅ Voltage variables: %s", self.voltage_vars)
        logging.info("✅ Ground node: %s", self.ground_node)
        
        # Assign current variables
        self.current_vars = self._assign_current_variables()
        logging.info("✅ Current variables: %s", self.current_vars)
        
        # Extract input and state variables
        self.state_vars, self.state_derivatives, self.input_vars = self._extract_input_and_state_vars()
        logging.info("✅ State variables: %s", self.state_vars)
        logging.info("✅ State derivatives: %s", self.state_derivatives)
        logging.info("✅ Input variables: %s", self.input_vars)
        
        return (self.voltage_vars, self.current_vars, self.state_vars, 
                self.state_derivatives, self.input_vars, self.ground_node)
    
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
    
    def _extract_input_and_state_vars(self) -> Tuple[Dict[sp.Symbol, sp.Expr], Dict[sp.Symbol, sp.Expr], Dict[sp.Symbol, sp.Expr]]:
        """
        Extracts input and state variables, defining helper variables for clarity.
        
        Returns:
            - state_vars: Dictionary of helper state variables and their equations
            - state_derivatives: Dictionary of state derivatives
            - input_vars: Dictionary of helper input variables and their equations
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
                    state_vars[v_C] = v_a - v_b  # v_C = V_A - V_B
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
                        state_vars[i_L] = self.current_vars[comp_id]  # I_L = I_L
                        state_derivatives[sp.Derivative(i_L, 't')] = (v_a - v_b)/L_value  # di/dt = (V_A - V_B)/L
            
            elif comp_type == "voltage-source":
                # Voltage source input variable
                if len(terminals) == 2:
                    node_a, node_b = terminals.values()
                    v_a = self.voltage_vars.get(node_a, sp.Symbol(f"V_{node_a}"))
                    v_b = self.voltage_vars.get(node_b, sp.Symbol(f"V_{node_b}"))
                    
                    V_source = sp.Symbol(f"V_in_{comp_id}")  # Helper variable for voltage source
                    input_vars[V_source] = v_a - v_b  # v_in_V = V_A - V_B
            
            elif comp_type == "current-source":
                # Current source input variable
                if comp_id in self.current_vars:
                    I_source = sp.Symbol(f"I_in_{comp_id}")  # Helper variable for current source
                    input_vars[I_source] = self.current_vars[comp_id]  # i_in_I = I_I
        
        return state_vars, state_derivatives, input_vars 

    def extract_differential_equations(self, components):
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
                              self.input_vars, self.ground_node)
        A, B, solved_helpers, differential_equations = model.build_model()

        # Substitute numerical values into A and B
        A_substituted = self.substitute_component_values(A, components)
        B_substituted = self.substitute_component_values(B, components)

        logging.info("✅ Substituted state matrix A: %s", A_substituted)
        logging.info("✅ Substituted input matrix B: %s", B_substituted)

        return A_substituted, B_substituted, model.state_vars, model.input_vars
        
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