import sympy as sp
from collections import defaultdict
import logging
from typing import Dict, Set, Tuple, List, Any, Union

class ElectricalModel:
    """
    Class for handling electrical modeling of circuits.
    
    This class encapsulates the methods for writing KCL and KVL equations,
    finding loops, solving helper variables, and extracting state space matrices.
    """
    
    def __init__(self, electrical_nodes: Dict[int, Set[Tuple[str, str]]], 
                 circuit_components: Dict[str, Dict],
                 voltage_vars: Dict[int, sp.Symbol],
                 current_vars: Dict[str, sp.Symbol],
                 state_vars: Dict[sp.Symbol, sp.Expr],
                 state_derivatives: Dict[sp.Symbol, sp.Expr],
                 input_vars: Dict[sp.Symbol, sp.Expr],
                 ground_node: int):
        """
        Initialize the ElectricalModel class.
        
        Args:
            electrical_nodes: { electrical_node_id: set((component_id, terminal_id), ...) }
            circuit_components: { component_id: { "type": str, "value": float, "terminals": {terminal_id: electrical_node} } }
            voltage_vars: { electrical_node_id: voltage_variable }
            current_vars: { component_id: current_variable }
            state_vars: { state_variable: expression }
            state_derivatives: { state_variable: derivative_expression }
            input_vars: { input_variable: expression }
            ground_node: The ground node ID
        """
        self.electrical_nodes = electrical_nodes
        self.circuit_components = circuit_components
        self.voltage_vars = voltage_vars
        self.current_vars = current_vars
        self.state_vars = state_vars
        self.state_derivatives = state_derivatives
        self.input_vars = input_vars
        self.ground_node = ground_node
        
        # Initialize other attributes
        self.kcl_equations = []
        self.kvl_equations = []
        self.loops = []
        self.supernodes = {}
        self.solved_helpers = {}
        self.differential_equations = {}
        self.A = None
        self.B = None
        
    def build_model(self):
        """
        Build the complete electrical model by calling all necessary methods in sequence.
        
        Returns:
            Tuple containing:
            - A: State matrix
            - B: Input matrix
            - solved_helpers: Dictionary of solved helper variables
            - differential_equations: Dictionary of state derivatives
        """
        # Step 1: Write KCL equations
        self.kcl_equations, self.supernodes = self.write_kcl_equations()
        logging.info("✅ Supernodes: %s", self.supernodes)
        
        # Step 2: Find loops
        self.loops = self.find_loops()
        
        # Step 3: Write KVL equations
        self.kvl_equations = self.write_kvl_equations()
        
        logging.info("✅ KCL equations: %s", self.kcl_equations)
        logging.info("✅ KVL equations: %s", self.kvl_equations)
        
        # Step 4: Solve helper variables
        self.solved_helpers = self.solve_helper_variables()
        logging.info("✅ Solved helper variables: %s", self.solved_helpers)
        
        # Step 5: Solve for state derivatives
        self.differential_equations = self.solve_state_derivatives()
        logging.info("✅ State derivatives: %s", self.differential_equations)
        
        # Step 6: Extract state space matrices
        self.A, self.B = self.extract_state_space_matrices()
        logging.info("✅ Symbolic State matrix A: %s", self.A)
        logging.info("✅ Symbolic Input matrix B: %s", self.B)
        
        return self.A, self.B, self.solved_helpers, self.differential_equations
    
    def write_kcl_equations(self) -> Tuple[List[sp.Expr], Dict[str, Set[int]]]:
        """
        Generates Kirchhoff's Current Law (KCL) equations for electrical nodes.
        
        Returns:
            - kcl_equations: List of KCL equations in symbolic form
            - supernodes: Dictionary mapping voltage source IDs to sets of connected nodes
        """
        kcl_equations = []
        
        # Identify voltage sources in the cleaned components
        voltage_sources = {comp_id: comp for comp_id, comp in self.circuit_components.items() 
                          if comp["type"] == "voltage-source"}

        # Step 1: Identify supernodes (voltage sources connecting non-ground nodes)
        supernodes = {}  # Maps supernodes to merged electrical nodes
        for vs_id, vs in voltage_sources.items():
            connected_nodes = set(vs["terminals"].values())
            
            # Only create a supernode if the voltage source connects **two non-ground nodes**
            if self.ground_node not in connected_nodes:
                supernodes[vs_id] = connected_nodes
        
        logging.info("Supernodes detected: %s", supernodes)
        
        # Step 2: Write KCL for normal electrical nodes (excluding ground, supernodes, and voltage-source-connected-to-ground nodes)
        for node_id, terminals in self.electrical_nodes.items():
            if node_id == self.ground_node:
                continue  # Ignore ground node

            # Skip nodes that belong to a supernode
            if any(node_id in nodes for nodes in supernodes.values()):
                continue

            # Skip nodes connected to a voltage source where the other terminal is grounded
            skip_node = False
            for vs_id, vs in voltage_sources.items():
                connected_nodes = set(vs["terminals"].values())
                if node_id in connected_nodes and self.ground_node in connected_nodes:
                    skip_node = True
                    break

            if skip_node:
                continue

            equation = 0  # Initialize symbolic equation

            # Iterate over terminals in this electrical node
            for comp_id, terminal_id in terminals:
                if comp_id not in self.circuit_components:
                    continue

                comp_data = self.circuit_components[comp_id]
                comp_type = comp_data["type"]
                value = sp.Symbol(f"{comp_id}_value") if comp_data["value"] is not None else None

                # Express the current depending on the component type
                if comp_id in self.current_vars:
                    if terminal_id == "0":  # Convention: Current entering the node is positive
                        equation += self.current_vars[comp_id]
                    elif terminal_id == "1":  # Current leaving the node is negative
                        equation -= self.current_vars[comp_id]

            kcl_equations.append(equation)

        # Step 3: Write KCL for supernodes
        for supernode_id, nodes in supernodes.items():
            equation = 0  # Initialize symbolic equation
            logging.info("ℹ️ Processing supernode %s with nodes %s", supernode_id, nodes)
            for node in nodes:
                if node in self.electrical_nodes:
                    for comp_id, terminal_id in self.electrical_nodes[node]:
                        if comp_id not in self.circuit_components:
                            continue

                        comp_data = self.circuit_components[comp_id]
                        comp_type = comp_data["type"]
                        value = sp.Symbol(f"{comp_id}_value") if comp_data["value"] is not None else None

                        # Express current for each component type
                        if comp_id in self.current_vars:
                            equation += self.current_vars[comp_id]
                            logging.info("ℹ️ Current variable %s added to equation: %s", self.current_vars[comp_id], equation)

            logging.info("ℹ️ Supernode equation: %s = 0", equation)
            kcl_equations.append(equation)  # Store symbolic equation

        return kcl_equations, supernodes
    
    def find_loops(self) -> List[List[int]]:
        """
        Find closed loops in the circuit using Depth-First Search (DFS).
        
        Returns:
            - loops: A list of loops, where each loop is a list of electrical nodes forming a cycle.
        """
        # Step 1: Build adjacency list for electrical nodes
        graph = defaultdict(set)

        for node_id, terminals in self.electrical_nodes.items():
            # For each terminal, find its component and connected node
            for comp_id, terminal_id in terminals:
                comp_data = self.circuit_components.get(comp_id)
                if not comp_data:
                    continue  # Skip if component is not found

                # Get the other terminal of the component
                for other_terminal_id, other_node_id in comp_data["terminals"].items():
                    if other_node_id != node_id:  # Ensure it's a different node
                        graph[node_id].add(other_node_id)
                        graph[other_node_id].add(node_id)  # Bidirectional edge

        # Step 2: Find loops using DFS
        loops = []
        visited = set()

        def dfs(node, path, start_node):
            """Recursive DFS to find loops."""
            for neighbor in graph[node]:
                if neighbor == start_node and len(path) > 2:
                    # Found a valid loop, normalize order to prevent duplicates
                    loop = sorted(path[:])
                    if loop not in loops:
                        loops.append(loop)
                    continue

                if neighbor not in path:  # Prevent backtracking
                    dfs(neighbor, path + [neighbor], start_node)

        # Step 3: Start DFS from each node
        for node in graph:
            if node not in visited:
                dfs(node, [node], node)
                visited.add(node)

        return loops
    
    def write_kvl_equations(self) -> List[sp.Expr]:
        """
        Generate Kirchhoff's Voltage Law (KVL) equations using component voltage relations.
        
        Returns:
            - kvl_equations: List of symbolic KVL equations.
        """
        kvl_equations = []

        for loop in self.loops:
            equation = 0  # Initialize symbolic equation

            # Iterate through the loop, processing components
            for i in range(len(loop)):
                node_a = loop[i]
                node_b = loop[(i + 1) % len(loop)]  # Next node in the loop (wraps around)

                # Find the component connecting these two nodes
                component_id = None
                component = None
                terminal_a = None
                terminal_b = None

                for comp_id, comp_data in self.circuit_components.items():
                    terminals = comp_data["terminals"]
                    if set(terminals.values()) == {node_a, node_b}:  # Component connects these two nodes
                        component_id = comp_id
                        component = comp_data
                        # Identify which terminal corresponds to which node
                        for term_id, node in terminals.items():
                            if node == node_a:
                                terminal_a = term_id
                            elif node == node_b:
                                terminal_b = term_id
                        break

                if component is None:
                    continue  # No component directly connects these nodes

                comp_type = component["type"]
                value = sp.Symbol(f"{component_id}_value") if component["value"] is not None else None

                # Determine voltage drop direction based on terminal convention
                v_a = self.voltage_vars.get(node_a, sp.Symbol(f"V_{node_a}"))
                v_b = self.voltage_vars.get(node_b, sp.Symbol(f"V_{node_b}"))
                voltage_diff = v_a - v_b if terminal_a == "0" else v_b - v_a  # Polarity correction

                # Apply voltage relation based on component type
                if comp_type == "resistor":
                    equation += (value * self.current_vars[component_id]) if terminal_a == "0" else (-value * self.current_vars[component_id]) 

                elif comp_type == "capacitor":
                    equation += sp.Symbol(f"V_{component_id}") if terminal_a == "0" else -sp.Symbol(f"V_{component_id}")

                elif comp_type == "inductor":
                    equation += (value * sp.Symbol(f"d{self.current_vars[component_id]}_dt")) if terminal_a == "0" else (-value * sp.Symbol(f"d{self.current_vars[component_id]}_dt"))

                elif comp_type == "voltage-source":
                    equation += sp.Symbol(f"V_in_{component_id}") if terminal_a == "0" else -sp.Symbol(f"V_in_{component_id}")

                else:
                    logging.warning(f"⚠ Warning: Unknown component type '{comp_type}' for {component_id}, ignoring in KVL.")

            # Kirchhoff's Voltage Law states ΣV = 0
            kvl_equations.append(equation)

        return kvl_equations
    
    def solve_helper_variables(self) -> Dict[sp.Symbol, sp.Expr]:
        """
        Eliminates helper variables by solving for them in terms of state variables and input variables.
        
        Returns:
            - solved_helpers: Dictionary mapping helper variables to their expressions in terms of state and input variables.
        """
        # Step 1: Identify helper variables (present in voltage_vars or current_vars but not in state_vars or input_vars)
        helper_vars_current = (
            set(self.current_vars.values())
            - set(self.state_vars.keys())
            - set(self.input_vars.keys())
        )

        helper_vars_voltage = (
            set(self.voltage_vars.values())
            - set(self.state_vars.keys())
            - set(self.input_vars.keys())
        )
        helper_vars_voltage.discard(0)  # Remove zero if present

        unknown_vars_voltage = set(helper_vars_voltage) - set(self.state_vars.keys()) - set(self.input_vars.keys())
        unknown_vars_current = set(helper_vars_current) - set(self.state_vars.keys()) - set(self.input_vars.keys())
        unknown_vars = unknown_vars_voltage.union(unknown_vars_current)

        logging.info("ℹ️ Helper variables current: %s", helper_vars_current)
        logging.info("ℹ️ Helper variables voltage: %s", helper_vars_voltage)

        # Step 2: Generate equations for resistors and capacitors 
        helper_eqs = []
        for comp_id, comp_data in self.circuit_components.items():
            if comp_data["type"] == "resistor":
                r_value = sp.Symbol(f"{comp_id}_value")  # Symbolic resistance value
                i_r = self.current_vars[comp_id]  # Current through the resistor

                # Extract node voltages
                terminals = comp_data["terminals"]
                node_1 = terminals["0"]
                node_2 = terminals["1"]
                
                v_node_1 = self.voltage_vars.get(node_1, f"V_{node_1}")
                v_node_2 = self.voltage_vars.get(node_2, f"V_{node_2}")
                
                # Ohm's Law: (V1 - V2) = IR
                helper_eqs.append((v_node_1 - v_node_2) - r_value * i_r)

            elif comp_data["type"] == "capacitor":
                # Capacitor current equation: i = C dv/dt
                c_value = sp.Symbol(f"{comp_id}_value")  # Symbolic capacitance value
                i_c = self.current_vars[comp_id]  # Current through the capacitor
                
                # Extract terminal nodes
                terminals = comp_data["terminals"]
                node_1 = terminals["0"]
                node_2 = terminals["1"]

                # Get node voltage symbols
                v_node_1 = self.voltage_vars.get(node_1, f"V_{node_1}")
                v_node_2 = self.voltage_vars.get(node_2, f"V_{node_2}")

                # Voltage difference across the capacitor
                v_cap = v_node_1 - v_node_2

                helper_eqs.append(v_cap - sp.Symbol(f"V_{comp_id}"))  # Voltage across the capacitor

            elif comp_data["type"] == "voltage-source":
                # Voltage source equation: V = V_in
                terminals = comp_data["terminals"]
                node_1 = terminals["0"]
                node_2 = terminals["1"]

                v_node_1 = self.voltage_vars.get(node_1, f"V_{node_1}")
                v_node_2 = self.voltage_vars.get(node_2, f"V_{node_2}")

                # Voltage difference across the voltage source
                v_source = sp.Symbol(f"V_in_{comp_id}")
                helper_eqs.append(v_node_1 - v_node_2 - v_source)

        logging.info("ℹ️ Helper equation: %s", helper_eqs)
        logging.info("ℹ️ Unknown variables: %s", unknown_vars)
        logging.info("ℹ️ Number of unknown variables: %d", len(unknown_vars))

        # Combine all equations
        all_eqs = helper_eqs + self.kcl_equations
        logging.info("ℹ️ Number of equations: %d", len(all_eqs))

        solved_helpers = sp.solve(all_eqs, unknown_vars)
        logging.info("ℹ️ Number of solutions: %d", len(solved_helpers))
        logging.info("ℹ️ Solved helper variables: %s", solved_helpers)

        return solved_helpers
    
    def solve_state_derivatives(self) -> Dict[sp.Symbol, sp.Expr]:
        """
        Solves the system of equations for the time derivatives of state variables.
        
        Returns:
            - differential_equations: Dictionary mapping {state_variable: derivative_expression}
        """
        # Substitute solved helper variables into the state_derivatives
        differential_equations = {var: expr.subs(self.solved_helpers) for var, expr in self.state_derivatives.items()}

        return differential_equations
    
    def extract_state_space_matrices(self) -> Tuple[sp.Matrix, sp.Matrix]:
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
        dx_dt_sol = sp.Matrix(list(self.differential_equations.values()))

        # Compute Jacobians
        A = dx_dt_sol.jacobian(state_vars)  # Partial derivatives of dx/dt w.r.t. state variables
        B = dx_dt_sol.jacobian(input_vars)  # Partial derivatives of dx/dt w.r.t. input variables

        return A, B 