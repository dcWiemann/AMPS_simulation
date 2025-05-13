from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np
import networkx as nx


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

    def compute_incidence_matrix(self) -> np.ndarray:
        """Compute the incidence matrix of the graph.
        
        Returns:
            np.ndarray: The incidence matrix of the graph
            list: The node voltage variables (corresponds to the rows of the incidence matrix)
            list: The component current variables (corresponds to the columns of the incidence matrix)
        """
        # remove ground node
        nodelist = [node for node in nodelist if node != 0]
        edgelist = list(self.graph.edges)

        # get node voltage and component current variables
        node_voltage_var_list = [node.voltage_var for node in nodelist]
        comp_current_var_list = [edge.current_var for edge in edgelist]
        return nx.incidence_matrix(self.graph, nodelist, edgelist).toarray(), node_voltage_var_list, comp_current_var_list
    
    def compute_kcl_equations(self) -> np.ndarray:
        """Compute the KCL equations of the graph.
        
        Returns:
            np.ndarray: The KCL equations of the graph.
        """
        incidence_matrix = self.compute_incidence_matrix()

