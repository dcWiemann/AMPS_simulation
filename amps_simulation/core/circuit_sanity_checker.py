"""
Circuit sanity checker for detecting common electrical circuit topology issues.

This module provides comprehensive sanity checks for electrical circuits represented
as NetworkX graphs, using efficient NetworkX subgraph filtering methods.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
import networkx as nx
from .components import (
    Component, VoltageSource, CurrentSource, Inductor, Capacitor, 
    Ground, ElecJunction
)

class CircuitTopologyError(Exception):
    """Exception raised for circuit topology errors."""
    pass

def has_short_circuit_path(G: nx.MultiDiGraph, s, t, exclude_components: Set[Component] = None):
    """Check if there is a short circuit path between two nodes."""
    if exclude_components is None:
        exclude_components = set()
    
    def edge_filter(u, v, k):
        component = G[u][v][k].get('component')
        return (component is not None and 
                component not in exclude_components and 
                component.is_short_circuit)
    
    H = nx.subgraph_view(G, filter_edge=edge_filter).to_undirected()
    return nx.has_path(H, s, t)

def has_current_path(G: nx.MultiDiGraph, s, t, exclude_components: Set[Component] = None):
    """Check if there's a current path between two nodes (not blocked by open circuits)."""
    if exclude_components is None:
        exclude_components = set()
    
    def edge_filter(u, v, k):
        component = G[u][v][k].get('component')
        return (component is not None and 
                component not in exclude_components and 
                not component.is_open_circuit)
    
    H = nx.subgraph_view(G, filter_edge=edge_filter).to_undirected()
    return nx.has_path(H, s, t)

class CircuitSanityChecker:
    """
    Performs comprehensive sanity checks on electrical circuit graphs.
    Uses NetworkX subgraph filtering for efficient path analysis.
    """
    
    def __init__(self, graph: nx.MultiDiGraph):
        """
        Initialize the sanity checker with a circuit graph.
        
        Args:
            graph: NetworkX MultiDiGraph representing the circuit
        """
        self.graph = graph
        self.warnings = []
        self.errors = []
        
    def check_all(self, raise_on_error: bool = True) -> Dict[str, List[str]]:
        """
        Run all sanity checks on the circuit.
        
        Args:
            raise_on_error: If True, raise exception on first error found
            
        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        self.warnings.clear()
        self.errors.clear()
        
        # Run all checks
        self._check_short_circuited_voltage_sources()
        self._check_open_circuit_current_sources()
        self._check_parallel_voltage_sources()
        self._check_series_current_sources()
        self._check_floating_nodes()
        self._check_open_circuit_inductors()
        self._check_short_circuit_capacitors()
        
        result = {
            'errors': self.errors.copy(),
            'warnings': self.warnings.copy()
        }
        
        if self.errors and raise_on_error:
            error_msg = "Circuit topology errors found:\n" + "\n".join(self.errors)
            raise CircuitTopologyError(error_msg)
            
        return result
    
    def _get_components_by_type(self, component_type: type) -> List[Tuple[str, str, Component]]:
        """Get all components of a specific type from the graph."""
        components = []
        for source, target, edge_data in self.graph.edges(data=True):
            component = edge_data.get('component')
            if component and isinstance(component, component_type):
                components.append((source, target, component))
        return components
    
    def _check_short_circuited_voltage_sources(self):
        """Check for voltage sources that are short-circuited by other components."""
        voltage_sources = self._get_components_by_type(VoltageSource)
        
        for source_node, target_node, vs_component in voltage_sources:
            # Check if voltage source terminals are short-circuited by other components
            if has_short_circuit_path(self.graph, source_node, target_node, {vs_component}):
                self.errors.append(
                    f"Voltage source '{vs_component.comp_id}' is short-circuited "
                    f"(nodes {source_node}-{target_node} connected by short circuit path)"
                )
    
    def _check_open_circuit_current_sources(self):
        """Check for current sources in open circuits."""
        current_sources = self._get_components_by_type(CurrentSource)
        
        for source_node, target_node, cs_component in current_sources:
            # Check if there's no current path excluding the current source itself
            if not has_current_path(self.graph, source_node, target_node, {cs_component}):
                self.errors.append(
                    f"Current source '{cs_component.comp_id}' is in open circuit "
                    f"(no current path between nodes {source_node}-{target_node})"
                )
    
    def _check_parallel_voltage_sources(self):
        """Check for voltage sources connected in parallel with different voltages."""
        voltage_sources = self._get_components_by_type(VoltageSource)
        
        # Group voltage sources by their terminal node pairs (direct parallel)
        node_pairs = {}
        for source_node, target_node, vs_component in voltage_sources:
            pair = tuple(sorted([source_node, target_node]))
            if pair not in node_pairs:
                node_pairs[pair] = []
            node_pairs[pair].append(vs_component)
        
        # Check direct parallel connections - always an error
        for pair, components in node_pairs.items():
            if len(components) > 1:
                comp_ids = [f"{comp.comp_id}({comp.voltage}V)" for comp in components]
                self.errors.append(
                    f"Voltage sources connected in parallel: "
                    f"{comp_ids} on nodes {pair[0]}-{pair[1]} "
                    f"(parallel voltage sources are not allowed regardless of voltage values)"
                )
        
        # Check for voltage sources connected through short circuit paths
        # Only check if BOTH terminals of each source are short-circuited to the other
        for i, (s1_src, s1_tgt, vs1) in enumerate(voltage_sources):
            for j, (s2_src, s2_tgt, vs2) in enumerate(voltage_sources[i+1:], i+1):
                # Skip if already found as direct parallel connection
                pair1 = tuple(sorted([s1_src, s1_tgt]))
                pair2 = tuple(sorted([s2_src, s2_tgt]))
                if pair1 == pair2:
                    continue
                
                # For true parallel connection, both terminal pairs must be short-circuited:
                # s1_src must be connected to s2_src AND s1_tgt must be connected to s2_tgt
                # OR s1_src must be connected to s2_tgt AND s1_tgt must be connected to s2_src
                
                # Check parallel configuration 1: s1_src<->s2_src AND s1_tgt<->s2_tgt
                parallel_config1 = (
                    has_short_circuit_path(self.graph, s1_src, s2_src, {vs1, vs2}) and
                    has_short_circuit_path(self.graph, s1_tgt, s2_tgt, {vs1, vs2})
                )
                
                # Check parallel configuration 2: s1_src<->s2_tgt AND s1_tgt<->s2_src  
                parallel_config2 = (
                    has_short_circuit_path(self.graph, s1_src, s2_tgt, {vs1, vs2}) and
                    has_short_circuit_path(self.graph, s1_tgt, s2_src, {vs1, vs2})
                )
                
                if parallel_config1 or parallel_config2:
                    self.errors.append(
                        f"Voltage sources '{vs1.comp_id}'({vs1.voltage}V) and "
                        f"'{vs2.comp_id}'({vs2.voltage}V) connected in parallel through short circuit paths "
                        f"(parallel voltage sources are not allowed regardless of voltage values)"
                    )
    
    def _check_series_current_sources(self):
        """Check for current sources connected in series."""
        current_sources = self._get_components_by_type(CurrentSource)
        
        for i, (s1_src, s1_tgt, cs1) in enumerate(current_sources):
            for j, (s2_src, s2_tgt, cs2) in enumerate(current_sources[i+1:], i+1):
                shared_nodes = set([s1_src, s1_tgt]) & set([s2_src, s2_tgt])
                
                if shared_nodes:
                    shared_node = shared_nodes.pop()
                    
                    # Check if shared node has only these two current sources connected
                    # and no other current paths (check both incoming and outgoing edges)
                    current_sources_at_node = []
                    other_current_paths = 0
                    
                    # Check outgoing edges from shared_node
                    for neighbor in self.graph.neighbors(shared_node):
                        for edge_data in self.graph[shared_node][neighbor].values():
                            component = edge_data.get('component')
                            if isinstance(component, CurrentSource):
                                current_sources_at_node.append(component)
                            elif component and not component.is_open_circuit:
                                other_current_paths += 1
                    
                    # Check incoming edges to shared_node
                    for predecessor in self.graph.predecessors(shared_node):
                        for edge_data in self.graph[predecessor][shared_node].values():
                            component = edge_data.get('component')
                            if isinstance(component, CurrentSource):
                                current_sources_at_node.append(component)
                            elif component and not component.is_open_circuit:
                                other_current_paths += 1
                    
                    # Series connection if only these two current sources and no other current paths
                    if (len(current_sources_at_node) == 2 and 
                        set(current_sources_at_node) == {cs1, cs2} and
                        other_current_paths == 0):
                        
                        self.errors.append(
                            f"Current sources '{cs1.comp_id}'({cs1.current}A) and "
                            f"'{cs2.comp_id}'({cs2.current}A) connected in series at node {shared_node} "
                            f"(series current sources are not allowed regardless of current values)"
                        )
    
    def _check_floating_nodes(self):
        """Check for nodes that are not connected to ground."""
        ground_nodes = set()
        
        # Find all ground nodes
        for node, node_data in self.graph.nodes(data=True):
            junction = node_data.get('junction')
            if junction and isinstance(junction, ElecJunction) and junction.is_ground:
                ground_nodes.add(node)
        
        if not ground_nodes:
            self.warnings.append("No ground reference found in circuit")
            return
        
        # Create subgraph with only current-carrying components
        def edge_filter(u, v, k):
            component = self.graph[u][v][k].get('component')
            return component is not None and not component.is_open_circuit
        
        current_graph = nx.subgraph_view(self.graph, filter_edge=edge_filter).to_undirected()
        
        # Find all nodes connected to ground using NetworkX connected_components
        connected_to_ground = set()
        for ground_node in ground_nodes:
            if ground_node in current_graph:
                component_nodes = nx.node_connected_component(current_graph, ground_node)
                connected_to_ground.update(component_nodes)
        
        # Find floating nodes
        all_nodes = set(self.graph.nodes())
        floating_nodes = all_nodes - connected_to_ground
        
        if floating_nodes:
            self.errors.append(
                f"Floating nodes detected (not connected to ground): {sorted(floating_nodes)}"
            )
    
    def _check_open_circuit_inductors(self):
        """Check for inductors in open circuits requiring i_L=0 and v_L=0 constraints."""
        inductors = self._get_components_by_type(Inductor)
        
        for source_node, target_node, inductor in inductors:
            # Check if inductor is the only current path between circuit sections
            if not has_current_path(self.graph, source_node, target_node, {inductor}):
                self.warnings.append(
                    f"Inductor '{inductor.comp_id}' is in open circuit. "
                    f"Constraints i_L = 0 and v_L = 0 must be added to equation system. "
                    f"Check for current discontinuities in initial conditions."
                )
    
    def _check_short_circuit_capacitors(self):
        """Check for capacitors in short circuits requiring v_C=0 constraint."""
        capacitors = self._get_components_by_type(Capacitor)
        
        for source_node, target_node, capacitor in capacitors:
            # Check if capacitor terminals are short-circuited by other components
            if has_short_circuit_path(self.graph, source_node, target_node, {capacitor}):
                self.warnings.append(
                    f"Capacitor '{capacitor.comp_id}' is in short circuit "
                    f"(nodes {source_node}-{target_node} connected by short circuit path). "
                    f"Constraint v_C = 0 must be added to equation system."
                )
    
    def get_constraint_modifications(self) -> Dict[str, List[str]]:
        """
        Get required equation system modifications based on detected issues.
        
        Returns:
            Dictionary with component IDs requiring constraint equations
        """
        constraints = {
            'zero_current_inductors': [],   # Inductors requiring i_L = 0
            'zero_voltage_inductors': [],   # Inductors requiring v_L = 0  
            'zero_voltage_capacitors': [],  # Capacitors requiring v_C = 0
        }
        
        # Check open circuit inductors
        inductors = self._get_components_by_type(Inductor)
        for source_node, target_node, inductor in inductors:
            if not has_current_path(self.graph, source_node, target_node, {inductor}):
                constraints['zero_current_inductors'].append(inductor.comp_id)
                constraints['zero_voltage_inductors'].append(inductor.comp_id)
        
        # Check short circuit capacitors  
        capacitors = self._get_components_by_type(Capacitor)
        for source_node, target_node, capacitor in capacitors:
            if has_short_circuit_path(self.graph, source_node, target_node, {capacitor}):
                constraints['zero_voltage_capacitors'].append(capacitor.comp_id)
        
        return constraints
    
    def log_results(self):
        """Log the sanity check results."""
        if self.errors:
            logging.error("Circuit topology errors:")
            for error in self.errors:
                logging.error(f"  - {error}")
        
        if self.warnings:
            logging.warning("Circuit topology warnings:")  
            for warning in self.warnings:
                logging.warning(f"  - {warning}")
        
        if not self.errors and not self.warnings:
            logging.info("Circuit topology checks passed successfully")