from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import numpy as np
import networkx as nx
from amps_simulation.core.components import Resistor, PowerSwitch, Inductor, Capacitor, Source, Meter
from sympy import Matrix, Symbol, sympify, solve

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
        self.outputs: Dict[str, float] = {}
    
    
    def get_derivatives(self) -> Dict[str, float]:
        """Get the current derivatives of the state variables.
        
        Returns:
            Dict[str, float]: Dictionary mapping state variable names to their derivatives
        """
        return self.derivatives
    
    def get_outputs(self) -> Dict[str, float]:
        """Get the current output values.
        
        Returns:
            Dict[str, float]: Dictionary mapping output variable names to their values
        """
        return self.outputs 
    

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
        self.res_eqs = self.compute_resistance_equations()
        self.switch_eqs = self.compute_switch_equations()
        self.circuit_vars = self.compute_circuit_vars()

    def find_state_vars(self) -> List[Symbol]:
        """Find the state variables in the graph.
        
        Returns:
            List[Symbol]: List of state variables
        """
        state_vars = []
        for edge in self.graph.edges(data=True):
            if isinstance(edge['component'], Inductor):
                state_vars.append(edge['component'].current_var)
            elif isinstance(edge['component'], Capacitor):
                state_vars.append(edge['component'].voltage_var)
        return state_vars
    
    def find_output_vars(self) -> List[Symbol]:
        """Find the output variables in the graph.
        """
        output_vars = []
        for edge in self.graph.edges(data=True):
            if isinstance(edge['component'], Meter):
                output_vars.append(edge['component'].output_var)
        return output_vars

    def find_input_vars(self) -> List[Symbol]:
        """Find the input variables in the graph.
        """
        input_vars = []
        for edge in self.graph.edges(data=True):
            if isinstance(edge['component'], Source):
                input_vars.append(edge['component'].input_var)
        return input_vars
    
    def variable_lists(self) -> Tuple[List[Symbol], List[Symbol], List[Symbol]]:
        """Get the lists of variables for the circuit.
        
        Returns:
            Tuple[List[Symbol], List[Symbol], List[Symbol]]: Lists of node voltage variables, component current variables, and component voltage variables
        """
        nodelist = [node for node in self.graph.nodes()]
        edgelist = list(self.graph.edges(data=True))

        # Convert variables to SymPy symbols, set ground voltage to 0
        junction_voltage_var_list = []
        for node in nodelist:
            voltage_var = self.graph.nodes[node]['junction'].voltage_var
            if voltage_var is None:
                junction_voltage_var_list.append(sympify(0))
            else:
                junction_voltage_var_list.append(sympify(voltage_var))
                        
        component_current_var_list = [sympify(data['component'].current_var) for _, _, data in edgelist]
        component_voltage_var_list = [sympify(data['component'].voltage_var) for _, _, data in edgelist]

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
        self.incidence_matrix = Matrix(incidence_matrix)
        
        return self.incidence_matrix
    
    def compute_kcl_equations(self) -> List[str]:
        """Compute Kirchhoff's Current Law equations.
        
        Returns:
            List[str]: List of KCL equations in symbolic form, excluding the ground node equation.
        """
        incidence_matrix, _, comp_current_vars, _ = self.compute_incidence_matrix()
        
        # Create current vector and multiply with incidence matrix
        current_vector = Matrix(comp_current_vars)
        kcl_equations = incidence_matrix * current_vector
        
        # Find the ground node index (where voltage_var is None)
        ground_node_idx = None
        for i, node in enumerate(self.graph.nodes()):
            if self.graph.nodes[node]['junction'].voltage_var is None:
                ground_node_idx = i
                break
        
        # Remove the ground node equation if found
        if ground_node_idx is not None:
            kcl_equations = [eq for i, eq in enumerate(kcl_equations) if i != ground_node_idx]
        
        # Convert equations to strings
        return [str(eq) for eq in kcl_equations]
    
    def compute_kvl_equations(self) -> List[str]:
        """Compute Kirchhoff's Voltage Law equations.
        
        Returns:
            List[str]: List of KVL equations in symbolic form.
        """
        incidence_matrix, junction_vars, _, comp_voltage_vars = self.compute_incidence_matrix()
        
        # Create voltage vector and multiply with transpose of incidence matrix
        voltage_vector = Matrix(junction_vars)
        kvl_equations = incidence_matrix.T * voltage_vector
        
        # Convert equations to strings and add component voltage variables
        return [f"{str(eq)} = {str(v_comp)}" for eq, v_comp in zip(kvl_equations, comp_voltage_vars)]
    
    def compute_resistance_equations(self) -> List[str]:
        """Compute the resistance equations of the graph.
        
        Returns:
            List[str]: The resistance equations of the graph.
        """
        R_eqs = []
        for _, _, data in self.graph.edges(data=True):
            if isinstance(data['component'], Resistor):
                R_eqs.append(data['component'].get_comp_eq())
        return R_eqs
    
    def compute_meter_equations(self) -> List[str]:
        """Compute the meter equations of the graph.
        
        Returns:
            List[str]: The meter equations of the graph.
        """
        meter_eqs = []
        for _, _, data in self.graph.edges(data=True):
            if isinstance(data['component'], Meter):
                meter_eqs.append(data['component'].get_comp_eq())
        return meter_eqs
        
    def compute_switch_equations(self) -> List[str]:
        """Compute the switch equations of the graph.
        
        Returns:
            List[str]: The switch equations of the graph.
        """
        switch_eqs = []
        for _, _, data in self.graph.edges(data=True):
            if isinstance(data['component'], PowerSwitch):
                switch_eqs.append(data['component'].get_comp_eq())
        return switch_eqs
    
    def compute_circuit_vars(self) -> List[str]:
        """Solve the circuit variables.
        
        Returns:
            List[str]: List of circuit variables in symbolic form.
        """
        # Combine all equations
        all_eqs = self.kcl_eqs + self.kvl_eqs + self.res_eqs + self.switch_eqs
        all_vars = self.junction_voltage_var_list + self.component_current_var_list + self.component_voltage_var_list

        # Solve the equations
        solution = solve(all_eqs, all_vars)
        return solution
    
    