import sympy as sp
import logging
from typing import Dict, Set, Tuple, List, Any
import networkx as nx
from scipy.integrate import solve_ivp
import numpy as np
from .components import Component, PowerSwitch, Capacitor, Inductor, VoltageSource, CurrentSource

class Engine:
    """
    Class for handling circuit simulation using a NetworkX graph structure.
    
    This class takes a NetworkX graph and handles the simulation of the circuit.
    """
    
    def __init__(self, graph: nx.MultiDiGraph):
        """
        Initialize the Engine class.
        
        Args:
            graph: NetworkX MultiDiGraph representing the circuit
        """
        self.graph = graph
        
        # Initialize simulation variables
        self.components_list = []
        self.state_vars = {}  # Dictionary of state variables
        self.state_derivatives = {}  # Dictionary of state derivatives
        self.input_vars = {}  # Dictionary of input variables
        self.power_switches = ()  # Tuple of power switches
        
        # Initialize voltage and current variables
        self.voltage_vars = {}  # Dictionary of voltage variables
        self.current_vars = {}  # Dictionary of current variables
        
        # Ground node reference
        self.ground_node = None

    def initialize(self) -> None:
        """
        Initialize all variables needed for simulation.
        This method should be called before running any simulation.
        """
        # Create components list from edge data
        self.components_list = []
        for _, _, edge_data in self.graph.edges(data=True):
            component = edge_data.get('component')
            if component is not None:
                self.components_list.append(component)
        
        # Set up all necessary variables
        self._set_state_vars()
        self._set_input_vars()
        self._set_power_switches()
        
        logging.info(f"✅ State variables: {self.state_vars}")
        logging.info(f"✅ Input variables: {self.input_vars}")
        logging.info(f"✅ Power switches: {self.power_switches}")

    def _set_state_vars(self) -> None:
        """
        Set state variables for capacitors and inductors and their derivatives.
        
        State variables are:
        - Capacitor voltages (v_C)
        - Inductor currents (i_L)
        
        Derivative equations are:
        - Capacitor: dv/dt = i/C
        - Inductor: di/dt = v/L
        """
        for component in self.components_list:
            if isinstance(component, Capacitor):
                # For capacitors, the state variable is the voltage
                v_C = sp.Symbol(component.voltage_var)
                i_C = sp.Symbol(component.current_var)
                C_value = component.capacitance
                
                # Store state variable
                self.state_vars[v_C] = component.comp_id
                
                # Create and store derivative equation: dv/dt = i/C
                self.state_derivatives[sp.Derivative(v_C, 't')] = i_C/C_value
                
            elif isinstance(component, Inductor):
                # For inductors, the state variable is the current
                i_L = sp.Symbol(component.current_var)
                v_L = sp.Symbol(component.voltage_var)
                L_value = component.inductance
                
                # Store state variable
                self.state_vars[i_L] = component.comp_id
                
                # Create and store derivative equation: di/dt = v/L
                self.state_derivatives[sp.Derivative(i_L, 't')] = v_L/L_value

    def _set_input_vars(self) -> None:
        """
        Set input variables for voltage and current sources.
        
        Input variables are:
        - Voltage source voltages using component.voltage_var
        - Current source currents using component.current_var
        """
        for component in self.components_list:
            if isinstance(component, VoltageSource):
                # For voltage sources, the input variable is the voltage
                V_source = sp.Symbol(component.voltage_var)
                self.input_vars[V_source] = component.comp_id
                
            elif isinstance(component, CurrentSource):
                # For current sources, the input variable is the current
                I_source = sp.Symbol(component.current_var)
                self.input_vars[I_source] = component.comp_id

    def _set_power_switches(self) -> None:
        """
        Extract power switches from the circuit components.
        
        Returns:
            Tuple[str, ...]: A tuple containing the IDs of all power switches in the circuit
        """
        self.power_switches = tuple(
            comp.comp_id for comp in self.components_list
            if isinstance(comp, PowerSwitch)
        )