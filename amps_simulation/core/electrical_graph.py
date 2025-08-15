from typing import Dict, List, Tuple
import numpy as np
import networkx as nx
from sympy import Matrix, Symbol
from .components import Resistor, PowerSwitch, Inductor, Capacitor, Source, Meter


class ElectricalGraph:
    """
    Class for managing electrical circuit graph structure and topology.
    
    This class handles the electrical circuit graph representation, including:
    - Circuit topology and incidence matrix
    - Component discovery and management (switches, etc.)
    - Variable lists for nodes and components
    - Graph-level operations and properties
    
    Separates graph structure concerns from mathematical DAE model computation.
    """
    
    def __init__(self, graph: nx.MultiDiGraph):
        """
        Initialize the ElectricalGraph.
        
        Args:
            graph: NetworkX MultiDiGraph representing the circuit
        """
        self.graph = graph
        
        # Graph topology properties
        self.incidence_matrix = None
        
        # Variable lists
        self.junction_voltage_var_list = None
        self.component_current_var_list = None
        self.component_voltage_var_list = None
        
        # Component lists
        self.switch_list = None
        
        # Initialize flag
        self.initialized = False
    
    def initialize(self) -> None:
        """
        Initialize all graph structure properties.
        
        This computes the incidence matrix, variable lists, and component lists.
        """
        # Compute graph topology
        self.incidence_matrix = self.compute_incidence_matrix()
        
        # Build variable lists
        self.junction_voltage_var_list, self.component_current_var_list, self.component_voltage_var_list = self.variable_lists()
        
        # Find components
        self.switch_list = self.find_switches()
        
        self.initialized = True
    
    def variable_lists(self) -> Tuple[List[Symbol], List[Symbol], List[Symbol]]:
        """Get the lists of variables for the circuit.
        
        Returns:
            Tuple[List[Symbol], List[Symbol], List[Symbol]]: Lists of node voltage variables, 
            component current variables, and component voltage variables
        """
        nodelist = [node for node in self.graph.nodes()]
        edgelist = list(self.graph.edges(data=True))

        # Set ground voltage to 0
        junction_voltage_var_list = []
        for node in nodelist:
            voltage_var = self.graph.nodes[node]['junction'].voltage_var
            assert voltage_var is not None, f"Voltage variable is None for node {node}"
            junction_voltage_var_list.append(voltage_var)
                        
        component_current_var_list = [data['component'].current_var for _, _, data in edgelist]
        component_voltage_var_list = [data['component'].voltage_var for _, _, data in edgelist]

        return junction_voltage_var_list, component_current_var_list, component_voltage_var_list

    def compute_incidence_matrix(self) -> Matrix:
        """Compute the incidence matrix of the graph.
        
        Returns:
            Matrix: The incidence matrix of the graph
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
        return Matrix(incidence_matrix)
    
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