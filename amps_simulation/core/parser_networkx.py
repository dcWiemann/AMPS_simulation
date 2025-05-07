import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List, Optional
import networkx as nx
from .components import (
    Component, Resistor, Capacitor, Inductor,
    PowerSwitch, Diode, VoltageSource, CurrentSource, Ground, ElecJunction
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
        self.components_list = []

    
    def parse(self, circuit_json: Dict[str, Any]) -> nx.Graph:
        """
        Parse JSON circuit description into a NetworkX graph structure.
        
        Args:
            circuit_json: Dictionary containing circuit description with 'nodes' and 'edges'
            
        Returns:
            nx.Graph: A NetworkX graph representing the circuit
        """
        # Clear registries to avoid duplicate ID issues
        Component.clear_registry()
        ElecJunction.clear_registry()

        # Extract components and connections
        components = circuit_json["nodes"]
        connections = circuit_json["edges"]
        
        self.components_list = self._create_circuit_components(components)
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

    def _get_component(self, comp_id: str) -> Optional[Component]:
        """Get a component by its ID using the Component registry."""
        return Component.get_component(comp_id)

    def _identify_electrical_nodes(self, connections: list) -> Tuple[Dict[Tuple[str, str], int], int, Optional[int]]:
        """
        Identify and map electrical nodes based on the given connections.

        This function processes the connections between components to determine
        which terminals are electrically connected, assigning a unique node number
        to each distinct electrical junction. It also handles merging of ground nodes.

        Args:
            connections (list): A list of connection dictionaries, each specifying
                the source and target components and their respective terminals.

        Returns:
            Tuple[Dict[Tuple[str, str], int], int, Optional[int]]: A tuple containing:
                - A dictionary mapping (component ID, terminal) pairs to node numbers.
                - The next available node number for new nodes.
                - The node number for the merged ground node, if any.
        """
        node_mapping = {}
        next_node_number = 1
        ground_node = None

        # First pass: identify ground nodes and their connections
        for conn in connections:
            source_comp_id = conn["source"]
            target_comp_id = conn["target"]
            
            # If either component is a ground, use the ground node number
            if any(isinstance(self._get_component(comp_id), Ground) 
                   for comp_id in [source_comp_id, target_comp_id]):
                if ground_node is None:
                    ground_node = next_node_number
                    next_node_number += 1
                
                source_key = (source_comp_id, conn.get("sourceHandle"))
                target_key = (target_comp_id, conn.get("targetHandle"))
                node_mapping[source_key] = ground_node
                node_mapping[target_key] = ground_node
                continue

            # Regular node handling for non-ground connections
            source_key = (source_comp_id, conn.get("sourceHandle"))
            target_key = (target_comp_id, conn.get("targetHandle"))
            
            if source_key in node_mapping and target_key in node_mapping:
                source_node = node_mapping[source_key]
                target_node = node_mapping[target_key]
                if source_node != target_node:
                    # Merge nodes, preferring non-ground nodes
                    merge_to = source_node if source_node != ground_node else target_node
                    for key, node in list(node_mapping.items()):
                        if node == (target_node if merge_to == source_node else source_node):
                            node_mapping[key] = merge_to
            elif source_key in node_mapping:
                node_mapping[target_key] = node_mapping[source_key]
            elif target_key in node_mapping:
                node_mapping[source_key] = node_mapping[target_key]
            else:
                node_mapping[source_key] = next_node_number
                node_mapping[target_key] = next_node_number
                next_node_number += 1

        # Create nodes for unconnected terminals (excluding ground)
        for comp in self.components_list:
            if not isinstance(comp, Ground):
                comp_id = comp.comp_id
                terminals = {"0", "1"}
                for terminal in terminals:
                    key = (comp_id, terminal)
                    if key not in node_mapping:
                        node_mapping[key] = next_node_number
                        next_node_number += 1

        return node_mapping, next_node_number, ground_node

    def _create_junctions(self, node_mapping: Dict[Tuple[str, str], int]) -> Dict[int, ElecJunction]:
        """
        Create ElecJunction objects for each unique electrical node.

        Args:
            node_mapping (Dict[Tuple[str, str], int]): A dictionary mapping
                (component ID, terminal) pairs to node numbers.

        Returns:
            Dict[int, ElecJunction]: A dictionary mapping node numbers to their
            corresponding ElecJunction objects.
        """
        used_numbers = sorted(set(node_mapping.values()))
        number_map = {old: i + 1 for i, old in enumerate(used_numbers)}
        for key, value in node_mapping.items():
            node_mapping[key] = number_map[value]

        junctions = {node_number: ElecJunction(junction_id=node_number) for node_number in number_map.values()}
        return junctions

    def _add_nodes_and_edges(self, node_mapping: Dict[Tuple[str, str], int], junctions: Dict[int, ElecJunction]) -> None:
        """
        Add nodes and edges to the graph using the identified nodes and junctions.

        This function iterates over the circuit components, adding nodes to the
        graph for each junction and creating edges between nodes to represent
        the components. Ground components are skipped as they do not form edges.

        Args:
            junctions (Dict[int, ElecJunction]): A dictionary mapping node numbers
                to ElecJunction objects, representing the electrical junctions.
        """
        for comp in self.components_list:
            comp_id = comp.comp_id

            if isinstance(comp, Ground):
                continue

            terminals = {}
            for terminal in ("0", "1"):
                key = (comp_id, terminal)
                if key in node_mapping:
                    terminals[terminal] = node_mapping[key]

            if "0" in terminals and "1" in terminals:
                source_node = str(terminals["0"])
                target_node = str(terminals["1"])

                if not self.graph.has_node(source_node):
                    elec_junction = junctions[int(source_node)]
                    self.graph.add_node(source_node, junction=elec_junction)
                if not self.graph.has_node(target_node):
                    elec_junction = junctions[int(target_node)]
                    self.graph.add_node(target_node, junction=elec_junction)

                self.graph.add_edge(
                    source_node,
                    target_node,
                    component=comp
                )

    def _create_electrical_graph(self, connections: list, components: list) -> None:
        """
        Create an electrical graph from the circuit connections and components.

        This function orchestrates the process of building a graph representation
        of the circuit by identifying electrical nodes, creating junctions, and
        adding nodes and edges to the graph. It ensures that all components and
        their connections are accurately represented in the graph structure.

        Args:
            connections (list): A list of connection dictionaries from JSON edges.
            components (list): A list of component dictionaries from JSON nodes.
        """
        node_mapping, next_node_number, ground_node = self._identify_electrical_nodes(connections)
        junctions = self._create_junctions(node_mapping)
        self._add_nodes_and_edges(node_mapping, junctions)
        