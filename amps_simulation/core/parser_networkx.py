import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List
import networkx as nx
from .components import (
    Component, Resistor, Capacitor, Inductor,
    PowerSwitch, Diode, VoltageSource, CurrentSource, Ground
)

class CircuitParser(ABC):
    """Abstract base class for circuit parsers that produce NetworkX graphs."""
    
    @abstractmethod
    def parse(self, circuit_data: Any) -> nx.Graph:
        """
        Parse circuit data into a NetworkX graph structure.
        
        Args:
            circuit_data: Circuit description in the format supported by this parser
            
        Returns:
            nx.Graph: A NetworkX graph representing the circuit
        """
        pass


class ParserJson(CircuitParser):
    """Parser for JSON circuit descriptions that produces NetworkX graphs."""
    
    def __init__(self):
        """Initialize the parser with an empty directed graph."""
        self.graph = nx.MultiDiGraph()
        self.circuit_components = {}

    
    def parse(self, circuit_json: Dict[str, Any]) -> nx.Graph:
        """
        Parse JSON circuit description into a NetworkX graph structure.
        
        Args:
            circuit_json: Dictionary containing circuit description with 'nodes' and 'edges'
            
        Returns:
            nx.Graph: A NetworkX graph representing the circuit
        """
        # Extract components and connections
        components = circuit_json["nodes"]
        connections = circuit_json["edges"]
        
        self.circuit_components = self._create_circuit_components(components)
        self._create_electrical_graph(connections, components)
        
        return self.graph
    

    def _create_circuit_components(self, components: list) -> List[Component]:
        """
        Create component objects from a list of component dictionaries (from JSON nodes).
        Returns a list of component objects in the same order as input.
        """
        component_map = {
            "resistor": Resistor,
            "capacitor": Capacitor,
            "inductor": Inductor,
            "switch": PowerSwitch,
            "diode": Diode,
            "voltage-source": VoltageSource,
            "current-source": CurrentSource,
            "ground": Ground,
        }
        component_list = []
        for comp in components:
            comp_id = comp["id"]
            data = comp["data"]
            ctype = data.get("componentType")
            cls = component_map.get(ctype)
            if not cls:
                logging.warning(f"Unknown component type: {ctype} for id {comp_id}")
                continue
            kwargs = {"comp_id": comp_id}
            if ctype == "resistor":
                kwargs["resistance"] = data.get("value")
            elif ctype == "capacitor":
                kwargs["capacitance"] = data.get("value")
            elif ctype == "inductor":
                kwargs["inductance"] = data.get("value")
            elif ctype == "voltage-source":
                kwargs["voltage"] = data.get("value")
            elif ctype == "current-source":
                kwargs["current"] = data.get("value")
            # Add more mappings as needed for other types
            component_list.append(cls(**kwargs))
        return component_list

    def _create_electrical_graph(self, connections: list, components: list) -> None:
        """
        Create an electrical graph from the circuit connections.
        In this graph:
        - Nodes represent electrical junctions (points where components connect)
        - Edges represent components, with direction from terminal 0 to terminal 1
        
        Args:
            connections: List of connection dictionaries from JSON edges
            components: List of component dictionaries from JSON nodes
        """
        # Create a mapping of component-terminal pairs to electrical nodes
        node_mapping = {}  # Maps (comp_id, terminal) to electrical node number
        next_node_number = 1
        
        # First pass: identify all electrical nodes by finding connected terminals
        for conn in connections:
            source_comp_id = conn["source"]
            target_comp_id = conn["target"]
            source_terminal = conn.get("sourceHandle")
            target_terminal = conn.get("targetHandle")
            
            # Get the keys for both terminals
            source_key = (source_comp_id, source_terminal)
            target_key = (target_comp_id, target_terminal)
            
            # If either terminal is already mapped, use that node number
            # This ensures connected terminals share the same node
            if source_key in node_mapping and target_key in node_mapping:
                # If both terminals are mapped but to different nodes, merge them
                source_node = node_mapping[source_key]
                target_node = node_mapping[target_key]
                if source_node != target_node:
                    # Update all references to target_node to use source_node
                    for key, node in list(node_mapping.items()):
                        if node == target_node:
                            node_mapping[key] = source_node
            elif source_key in node_mapping:
                node_mapping[target_key] = node_mapping[source_key]
            elif target_key in node_mapping:
                node_mapping[source_key] = node_mapping[target_key]
            else:
                # If neither terminal is mapped, create a new node
                node_mapping[source_key] = next_node_number
                node_mapping[target_key] = next_node_number
                next_node_number += 1
        
        # Second pass: create nodes for unconnected terminals
        for comp in self.circuit_components:
            comp_id = comp.comp_id
            
            # Find all connections for this component
            comp_connections = [c for c in connections 
                              if c["source"] == comp_id or c["target"] == comp_id]
            
            # Get all terminals for this component
            terminals = {"0", "1"} if not isinstance(comp, Ground) else {"0"}
            
            # For each terminal, if it's not connected, create a new node
            for terminal in terminals:
                key = (comp_id, terminal)
                # Check if this terminal appears in any connection
                is_connected = False
                for conn in comp_connections:
                    if (conn["source"] == comp_id and conn["sourceHandle"] == terminal) or \
                       (conn["target"] == comp_id and conn["targetHandle"] == terminal):
                        is_connected = True
                        break
                
                # If terminal is not connected, create a new node
                if not is_connected:
                    node_mapping[key] = next_node_number
                    next_node_number += 1
        
        # Third pass: create the graph with components as edges
        for comp in self.circuit_components:
            comp_id = comp.comp_id
            
            # Get the terminals and their corresponding electrical nodes
            terminals = {}
            for terminal in ("0", "1"):
                key = (comp_id, terminal)
                if key in node_mapping:
                    terminals[terminal] = node_mapping[key]
            
            # Create edges for each component
            # For components with two terminals, create an edge from terminal 0 to terminal 1
            if "0" in terminals and "1" in terminals:
                source_node = str(terminals["0"])
                target_node = str(terminals["1"])
                
                # Add nodes if they don't exist
                if not self.graph.has_node(source_node):
                    self.graph.add_node(source_node, type="electrical_node")
                if not self.graph.has_node(target_node):
                    self.graph.add_node(target_node, type="electrical_node")
                
                # Add edge with component information
                self.graph.add_edge(
                    source_node,
                    target_node,
                    component=comp
                )
            
            # For components with one terminal (like ground), just add the node
            elif len(terminals) == 1:
                node = str(list(terminals.values())[0])
                if not self.graph.has_node(node):
                    self.graph.add_node(node, type="electrical_node")
        