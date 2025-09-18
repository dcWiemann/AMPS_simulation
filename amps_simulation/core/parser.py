import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List, Optional
import networkx as nx
from .components import (
    Component, Resistor, Capacitor, Inductor,
    PowerSwitch, Diode, VoltageSource, CurrentSource, Ground, ElecJunction, Ammeter, Voltmeter
)
from .control_orchestrator import ControlGraph, ControlSignal
from .control_port import ControlPort

class CircuitParser(ABC):
    """Abstract base class for circuit parsers that produce NetworkX graphs."""
    
    @abstractmethod
    def parse(self, circuit_data: Any) -> Tuple[nx.Graph, ControlGraph]:
        """
        Parse circuit data into NetworkX graph and control graph structures.
        
        Args:
            circuit_data: Circuit description in the format supported by this parser
            
        Returns:
            Tuple[nx.Graph, ControlGraph]: Electrical graph and control graph
        """
        pass


class ParserJson(CircuitParser):
    """Parser for JSON circuit descriptions that produces NetworkX graphs."""
    
    def __init__(self):
        """Initialize the parser with an empty directed graph."""
        self.graph = nx.MultiDiGraph()
        self.control_graph = ControlGraph()
        self.components_list = []

    
    def parse(self, circuit_json: Dict[str, Any]) -> Tuple[nx.Graph, ControlGraph]:
        """
        Parse JSON circuit description into NetworkX graph and control graph structures.
        
        Args:
            circuit_json: Dictionary containing circuit description with 'nodes' and 'edges'
            
        Returns:
            Tuple[nx.Graph, ControlGraph]: Electrical graph and control graph
        """
        # Clear registries to avoid duplicate ID issues
        Component.clear_registry()
        ElecJunction.clear_registry()
        ControlPort.clear_registry()

        # Extract components and connections
        components = circuit_json["nodes"]
        connections = circuit_json["edges"]
        
        self.components_list = self._create_circuit_components(components)
        self._create_electrical_model(connections, components)
        self._create_control_graph(components)
        
        return self.graph, self.control_graph
    

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
            "ammeter": Ammeter,
            "voltmeter": Voltmeter,
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
                # Use "voltage" field if present, otherwise "value" field, otherwise default to 0
                value = data.get("voltage", data.get("value", 0.0))
                # If value is a string, it's a control signal, use 0.0 as placeholder for the component
                kwargs["voltage"] = 0.0 if isinstance(value, str) else value
            elif ctype == "current-source":
                # Use "current" field if present, otherwise "value" field, otherwise default to 0  
                value = data.get("current", data.get("value", 0.0))
                # If value is a string, it's a control signal, use 0.0 as placeholder for the component
                kwargs["current"] = 0.0 if isinstance(value, str) else value
            elif ctype == "switch":
                kwargs["switch_time"] = data.get("value")
                kwargs["is_on"] = False # default is off
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

        # Process all connections in two passes
        # First pass: handle all non-ground connections
        for conn in connections:
            source_comp_id = conn["source"]
            target_comp_id = conn["target"]
            
            # Skip ground connections for first pass
            if any(isinstance(self._get_component(comp_id), Ground) 
                   for comp_id in [source_comp_id, target_comp_id]):
                continue

            source_key = (source_comp_id, conn.get("sourceHandle"))
            target_key = (target_comp_id, conn.get("targetHandle"))
            
            if source_key in node_mapping and target_key in node_mapping:
                # Both terminals already mapped - merge if different
                source_node = node_mapping[source_key]
                target_node = node_mapping[target_key]
                if source_node != target_node:
                    # Merge target_node into source_node
                    for key, node in list(node_mapping.items()):
                        if node == target_node:
                            node_mapping[key] = source_node
            elif source_key in node_mapping:
                # Source mapped, assign target to same node
                node_mapping[target_key] = node_mapping[source_key]
            elif target_key in node_mapping:
                # Target mapped, assign source to same node
                node_mapping[source_key] = node_mapping[target_key]
            else:
                # Neither mapped, create new node for both
                node_mapping[source_key] = next_node_number
                node_mapping[target_key] = next_node_number
                next_node_number += 1

        # Second pass: handle ground connections and merge with existing nodes
        for conn in connections:
            source_comp_id = conn["source"]
            target_comp_id = conn["target"]
            
            # Only process ground connections
            if not any(isinstance(self._get_component(comp_id), Ground) 
                       for comp_id in [source_comp_id, target_comp_id]):
                continue

            source_key = (source_comp_id, conn.get("sourceHandle"))
            target_key = (target_comp_id, conn.get("targetHandle"))
            
            # Find which terminal is already mapped (non-ground terminal)
            existing_node = None
            if source_key in node_mapping:
                existing_node = node_mapping[source_key]
            elif target_key in node_mapping:
                existing_node = node_mapping[target_key]
            
            if existing_node is not None:
                # If we already have a ground node, merge the existing node with it
                if ground_node is not None:
                    # Merge existing_node into ground_node
                    for key, node in list(node_mapping.items()):
                        if node == existing_node:
                            node_mapping[key] = ground_node
                else:
                    # Use the existing node as the ground node
                    ground_node = existing_node
            else:
                # Neither terminal is mapped yet, create new ground node
                if ground_node is None:
                    ground_node = next_node_number
                    next_node_number += 1
            
            # Assign both terminals to ground node
            node_mapping[source_key] = ground_node
            node_mapping[target_key] = ground_node

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

    def _create_junctions(self, node_mapping: Dict[Tuple[str, str], int], ground_node: Optional[int]) -> Dict[int, ElecJunction]:
        """
        Create ElecJunction objects for each unique electrical node.

        Args:
            node_mapping (Dict[Tuple[str, str], int]): A dictionary mapping
                (component ID, terminal) pairs to node numbers.
            ground_node (Optional[int]): The node number of the ground node, if any.

        Returns:
            Dict[int, ElecJunction]: A dictionary mapping node numbers to their
            corresponding ElecJunction objects.
        """
        used_numbers = sorted(set(node_mapping.values()))
        number_map = {old: i + 1 for i, old in enumerate(used_numbers)}
        for key, value in node_mapping.items():
            node_mapping[key] = number_map[value]

        # Update ground_node to new numbering if it exists
        if ground_node is not None:
            ground_node = number_map[ground_node]

        junctions = {}
        for node_number in number_map.values():
            is_ground = (node_number == ground_node) if ground_node is not None else False
            junctions[node_number] = ElecJunction(junction_id=node_number, is_ground=is_ground)

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

    def _create_electrical_model(self, connections: list, components: list) -> None:
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
        junctions = self._create_junctions(node_mapping, ground_node)
        self._add_nodes_and_edges(node_mapping, junctions)

        # Add ground node to the graph
        if ground_node is None:
            # Find the node with the most components (edges) connected to it
            node_degrees = self.graph.degree()
            max_degree_node = max(node_degrees, key=lambda x: x[1])[0]
            ground_node = int(max_degree_node)
            # Assign is_ground = True to the node with the most connections
            junctions[ground_node].is_ground = True

    def _create_control_graph(self, components: List[Dict[str, Any]]) -> None:
        """
        Create control graph from source components with 'value' fields.
        
        Args:
            components: List of component dictionaries from JSON nodes
        """
        for comp_dict in components:
            comp_id = comp_dict["id"]
            data = comp_dict["data"]
            comp_type = data.get("componentType")
            
            # Only process voltage and current sources
            if comp_type not in ["voltage-source", "current-source"]:
                continue
                
            value = data.get("value")
            if value is None:
                continue
                
            # Create control signal from value
            signal_id = f"{comp_id}_signal"
            signal = ControlSignal(signal_id, value)
            self.control_graph.add_signal(signal)
            
            # Create control port for the source
            port_name = f"{comp_id}_port"
            # Get the component to access its input_var
            component = Component.get_component(comp_id)
            if component and hasattr(component, 'input_var'):
                # Assign port name to the component
                component.control_port_name = port_name
                
                port = ControlPort(name=port_name, variable=component.input_var, port_type="source")
                self.control_graph.add_port(port)
                
                # Connect signal to port (1:1 mapping for now)
                self.control_graph.connect_signal_to_port(signal_id, port_name)