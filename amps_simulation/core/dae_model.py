from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import numpy as np
import networkx as nx
from amps_simulation.core.components import Resistor, PowerSwitch, Inductor, Capacitor, Source, Meter
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
    def __init__(self, graph: nx.Graph):
        super().__init__(graph)
        self.initialized = False

    def initialize(self):
        # Initialize the model
        
        # Find all circuit variables
        self.state_vars = self.find_state_vars()
        self.output_vars = self.find_output_vars()
        self.input_vars = self.find_input_vars()
        self.junction_voltage_var_list, self.component_current_var_list, self.component_voltage_var_list = self.variable_lists()
        # Find all equations that describe the circuit
        self.incidence_matrix = self.compute_incidence_matrix()
        self.kcl_eqs = self.compute_kcl_equations()
        self.kvl_eqs = self.compute_kvl_equations()
        self.static_eqs = self.compute_static_component_equations()
        self.switch_list = self.find_switches()
        self.switch_eqs = self.compute_switch_equations()
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
    
    def variable_lists(self) -> Tuple[List[Symbol], List[Symbol], List[Symbol]]:
        """Get the lists of variables for the circuit.
        
        Returns:
            Tuple[List[Symbol], List[Symbol], List[Symbol]]: Lists of node voltage variables, component current variables, and component voltage variables
        """
        nodelist = [node for node in self.graph.nodes()]
        edgelist = list(self.graph.edges(data=True))

        # Set ground voltage to 0
        junction_voltage_var_list = []
        for node in nodelist:
            voltage_var = self.graph.nodes[node]['junction'].voltage_var
            assert voltage_var is not None, "Voltage variable is None for component %s" % self.graph.nodes[node]['component'].comp_id
            junction_voltage_var_list.append(voltage_var)
                        
        component_current_var_list = [data['component'].current_var for _, _, data in edgelist]
        component_voltage_var_list = [data['component'].voltage_var for _, _, data in edgelist]

        return junction_voltage_var_list, component_current_var_list, component_voltage_var_list

    def compute_incidence_matrix(self) -> Tuple[Matrix, List[Symbol], List[Symbol], List[Symbol]]:
        """Compute the incidence matrix of the graph.
        
        Returns:
            Matrix: The incidence matrix of the graph
            List[Symbol]: The node voltage variables (corresponds to the rows of the incidence matrix)
            List[Symbol]: The component current variables (corresponds to the columns of the incidence matrix)
            List[Symbol]: The component voltage variables
        """
        nodelist = [node for node in self.graph.nodes()]
        edgelist = list(self.graph.edges(data=True))

        # Create incidence matrix manually to handle edge directions
        n_nodes = len(nodelist)
        n_edges = len(edgelist)
        incidence_matrix = np.zeros((n_nodes, n_edges))
        
        # Fill incidence matrix based on edge directions
        for edge_idx, (source, target, data) in enumerate(edgelist):
            source_idx = nodelist.index(source)
            target_idx = nodelist.index(target)
            
            # Set +1 for source node and -1 for target node based on edge direction
            incidence_matrix[source_idx, edge_idx] = 1
            incidence_matrix[target_idx, edge_idx] = -1
        
        # Convert to SymPy Matrix
        incidence_matrix = Matrix(incidence_matrix)
        
        return incidence_matrix
    
    def compute_kcl_equations(self) -> List[Symbol]:
        """Compute Kirchhoff's Current Law equations.
        
        Returns:
            List[Symbol]: List of KCL equations in symbolic form, excluding the ground node equation.
        """
        if self.initialized == False:
            _, comp_current_vars, _ = self.variable_lists()
            incidence_matrix = self.compute_incidence_matrix()
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
            junction_voltage_var_list, _, comp_voltage_vars = self.variable_lists()
            incidence_matrix = self.compute_incidence_matrix()
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
            if isinstance(component, (Resistor, Meter)):
                static_eqs.append(component.get_comp_eq())
        return static_eqs
    
    def find_switches(self) -> List[PowerSwitch]:
        """Find the switches in the graph.
        
        Returns:
            List[PowerSwitch]: List of switches
        """
        switch_list = []
        for _, _, data in self.graph.edges(data=True):
            if isinstance(data['component'], PowerSwitch):
                switch_list.append(data['component'])
        return switch_list
        
    def compute_switch_equations(self) -> List[Symbol]:
        """Compute the switch equations of the graph.
        
        Returns:
            List[Symbol]: The switch equations of the graph.
        """
        if self.initialized == False:
            switch_list = self.find_switches()
        else:
            switch_list = self.switch_list
        
        switch_eqs = []
        for switch in switch_list:
            switch_eqs.append(switch.get_comp_eq())

        return switch_eqs
    
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
            junction_voltage_var_list, component_current_var_list, component_voltage_var_list = self.variable_lists()
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
        
        logging.debug("input_vars: ", input_vars)
        logging.debug("output_vars: ", output_vars)
        logging.debug("state_vars: ", state_vars)

        # Combine all equations
        all_eqs = kcl_eqs + kvl_eqs + static_eqs + switch_eqs
        # Find vars to solve for:
        # Remove 0 (ground node) from junction_voltage_var_list
        junction_voltage_var_list_cleaned = [var for var in junction_voltage_var_list if var != 0]
        # Combine all variables
        combined_vars = junction_voltage_var_list_cleaned + component_current_var_list + component_voltage_var_list
        # Remove input_vars and state_vars
        excluded = set(input_vars) | set(state_vars)
        all_vars = [var for var in combined_vars if var not in excluded]

        logging.debug("all_eqs: ", all_eqs)
        logging.debug("all_vars: ", all_vars)

        number_of_equations = len(all_eqs)
        number_of_variables = len(all_vars)
        logging.debug("number of eqs: ", number_of_equations)
        logging.debug("number of vars: ", number_of_variables)

        if number_of_equations != number_of_variables:
            raise Warning("The number of equations and variables must be the same. (%d equations, %d variables)" % (number_of_equations, number_of_variables))
        
        # Solve the equations
        solution = solve(all_eqs, all_vars)
        number_of_solutions = len(solution)
        logging.debug("number of sols:", number_of_solutions)
        logging.debug("solutions: ", solution)

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
            switchmap = {switch.comp_id: switch.control_signal(t) for switch in self.switch_list}
        
        assert self.initialized == True, "Model must be initialized before updating switch states"
        self.switch_eqs = self.compute_switch_equations()
        self.circuit_eqs = self.compute_circuit_equations()
        self.derivatives = self.compute_derivatives()
        self.output_eqs = self.compute_output_equations()
        
        if t is not None:
            return self.circuit_eqs, self.derivatives, switchmap
        else:
            return self.circuit_eqs, self.derivatives