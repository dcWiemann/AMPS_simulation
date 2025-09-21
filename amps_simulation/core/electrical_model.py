from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import networkx as nx
from sympy import Matrix, Symbol
from .components import Resistor, PowerSwitch, Inductor, Capacitor, Source, Meter, Diode, Component, ElecJunction


class ElectricalModel:
    """
    Class for managing electrical circuit model structure and topology.

    This class handles the electrical circuit model representation, including:
    - Circuit topology and incidence matrix
    - Component discovery and management (switches, etc.)
    - Variable lists for nodes and components
    - Graph-level operations and properties
    
    Separates graph structure concerns from mathematical DAE model computation.
    """
    
    def __init__(self, graph: Optional[nx.MultiDiGraph] = None):
        """
        Initialize the ElectricalModel.

        Args:
            graph: NetworkX MultiDiGraph representing the circuit. If None, creates empty graph.
        """
        self.graph = graph if graph is not None else nx.MultiDiGraph()
        
        # Graph topology properties
        self.incidence_matrix = None
        
        # Variable lists
        self.junction_voltage_var_list = None
        self.component_current_var_list = None
        self.component_voltage_var_list = None
        
        # Component lists
        self.switch_list = None
        self.diode_list = None

        # Circuit variable lists
        self.state_vars = None
        self.input_vars = None
        self.output_vars = None

        # Initial state management
        self.initial_state_values = None
        
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
        self.diode_list = self.find_diodes()

        # Find circuit variables
        self.state_vars = self.find_state_vars()
        self.input_vars = self.find_input_vars()
        self.output_vars = self.find_output_vars()

        # Initialize default state values
        self.initial_state_values = self.compute_initial_state_values()
        
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
    
    def find_diodes(self) -> List[Diode]:
        """Find the diodes in the graph.
        
        Returns:
            List[Diode]: List of diodes
        """
        diode_list = []
        for _, _, data in self.graph.edges(data=True):
            if isinstance(data['component'], Diode):
                diode_list.append(data['component'])
        return diode_list
    
    def compute_initial_state_values(self) -> np.ndarray:
        """Compute default initial state values for the circuit.
        
        For now, this returns zeros for all state variables (inductors and capacitors).
        In the future, this could be extended to compute steady-state initial conditions.
        
        Returns:
            np.ndarray: Array of initial state values
        """
        # Count state variables (inductors and capacitors)
        state_var_count = 0
        for _, _, data in self.graph.edges(data=True):
            if isinstance(data['component'], (Inductor, Capacitor)):
                state_var_count += 1
        
        # Return zeros for now (could be extended for steady-state calculation)
        return np.zeros(state_var_count)

    def add_node(self, node_id: int, is_ground: bool = False) -> None:
        """
        Add an electrical junction node to the circuit.

        Args:
            node_id: Unique integer identifier for the node
            is_ground: Whether this node should be treated as ground reference
        """
        if node_id in self.graph.nodes:
            # Node already exists, update ground status if needed
            existing_junction = self.graph.nodes[node_id]['junction']
            if is_ground:
                existing_junction.is_ground = True
        else:
            # Create new junction and add node
            junction = ElecJunction(junction_id=node_id, is_ground=is_ground)
            self.graph.add_node(node_id, junction=junction)

    def add_component(self, component: Component, terminals: Union[List[int], Dict[str, int], None] = None, **kwargs) -> None:
        """
        Add a component to the circuit with specified terminal connections.

        Args:
            component: The component instance to add
            terminals: Terminal connections as list [node1, node2] or dict {"p": node1, "n": node2}
            **kwargs: Additional named terminal connections (e.g. p=1, n=0)
        """
        # Parse terminal connections
        if isinstance(terminals, list):
            # List format: [node1, node2]
            if len(terminals) != 2:
                raise ValueError(f"List format requires exactly 2 terminals, got {len(terminals)}")
            node1, node2 = terminals
        elif isinstance(terminals, dict):
            # Dict format: {"p": node1, "n": node2}
            terminal_dict = terminals.copy()
            terminal_dict.update(kwargs)  # Merge with any additional kwargs

            # For now, handle common 2-terminal case
            if "p" in terminal_dict and "n" in terminal_dict:
                node1, node2 = terminal_dict["p"], terminal_dict["n"]
            elif len(terminal_dict) == 2:
                # Take first two values for 2-terminal components
                nodes = list(terminal_dict.values())
                node1, node2 = nodes[0], nodes[1]
            else:
                raise ValueError(f"Unsupported terminal configuration: {terminal_dict}")
        elif terminals is None:
            # kwargs only (e.g. p=1, n=0)
            if len(kwargs) != 2:
                raise ValueError(f"Expected 2 terminal connections, got {len(kwargs)}")
            if "p" in kwargs and "n" in kwargs:
                node1, node2 = kwargs["p"], kwargs["n"]
            else:
                # Take first two values
                nodes = list(kwargs.values())
                node1, node2 = nodes[0], nodes[1]
        else:
            raise ValueError(f"Unsupported terminals parameter: {terminals}")

        # Auto-create nodes if they don't exist
        if node1 not in self.graph.nodes:
            self.add_node(node1)
        if node2 not in self.graph.nodes:
            self.add_node(node2)

        # Add component as edge between the two nodes
        self.graph.add_edge(node1, node2, component=component)

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
        for _, _, data in self.graph.edges(data=True):
            if isinstance(data['component'], Meter):
                output_vars.append(data['component'].output_var)
        return output_vars

    def find_input_vars(self) -> List[Symbol]:
        """Find the input variables in the graph.

        Returns:
            List[Symbol]: List of input variables
        """
        input_vars = []
        for _, _, data in self.graph.edges(data=True):
            if isinstance(data['component'], Source):
                input_vars.append(data['component'].input_var)
        return input_vars