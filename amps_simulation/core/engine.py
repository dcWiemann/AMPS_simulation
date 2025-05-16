import sympy as sp
import logging
from typing import Dict, Set, Tuple, List, Any
import networkx as nx
from scipy.integrate import solve_ivp
import numpy as np
from .components import Component, PowerSwitch, Capacitor, Inductor, VoltageSource, CurrentSource
import itertools

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
        # self.switch_control_signals = None  # Function to get switch control signals
        self.switch_events = None  # List of switch events
        
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
        self._get_state_vars()
        self._get_input_vars()
        self._get_power_switches()
        if self.power_switches:
            # self.switch_control_signals = self._get_switch_control_signals()
            self.switch_events = self._get_switch_events()
            # self.possible_switch_positions = self._get_possible_switch_positions()
        else:
            self.switch_control_signals = None
            self.switch_events = None
            # self.possible_switch_positions = (0,) # Default to 0 if no power switches

        logging.debug(f"✅ State variables: {self.state_vars}")
        logging.debug(f"✅ Input variables: {self.input_vars}")
        logging.debug(f"✅ Power switches: {self.power_switches}")
        logging.debug(f"✅ Switch control signals: {self.switch_control_signals}")
        logging.debug(f"✅ Switch events: {self.switch_events}")


    def _get_state_vars(self) -> None:
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

    def _get_input_vars(self) -> None:
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

    def _get_power_switches(self) -> None:
        """
        Extract power switches from the circuit components.
        
        Returns:
            Tuple[str, ...]: A tuple containing the IDs of all power switches in the circuit
        """
        self.power_switches = tuple(
            comp.comp_id for comp in self.components_list
            if isinstance(comp, PowerSwitch)
        )

    def _get_switch_control_signals(self):
        """
        Creates a callable function that returns the switch states at any given time t.
        
        Returns:
            callable: A function switch_control_signals(t) that returns a tuple of 0s and 1s
                     representing the state of each switch at time t. The order of the tuple
                     corresponds exactly to the order of switches in self.power_switches.
        """
        # Get switch times from components
        switch_times = {}
        for comp in self.components_list:
            if isinstance(comp, PowerSwitch):
                switch_times[comp.comp_id] = comp.switch_time

        def switch_control_signals(t):
            """
            Returns the state of all switches at time t.
            
            Args:
                t: Time at which to evaluate switch states
                
            Returns:
                tuple: A tuple of 0s and 1s representing switch states (0=OFF, 1=ON)
            """
            # Create tuple of switch states in the same order as self.power_switches
            return tuple(1 if t >= switch_times[switch_id] else 0 
                        for switch_id in self.power_switches)
            
        return switch_control_signals

    def _get_switch_events(self):
        """
        Creates event functions for each switch in the circuit.
        
        Returns:
            List[callable]: A list of event functions that return 0 when a switch changes state.
        """
        # Get switch times from components
        switch_times = {}
        for comp in self.components_list:
            if isinstance(comp, PowerSwitch):
                switch_times[comp.comp_id] = comp.switch_time

        def create_event_function(switch_time):
            def event(t, x):
                return t - switch_time
            event.terminal = True  # Stop the integration when the event occurs
            event.direction = 0  # Detect both positive and negative crossings
            return event

        # Create a list of event functions for each switch
        switch_events = [create_event_function(switch_times[switch_id]) for switch_id in self.power_switches]
        
        return switch_events
    
    # def _get_possible_switch_positions(self):
    #     """
    #     Get all possible switch positions for the circuit.
        
    #     Returns:
    #         List[Tuple[int, ...]]: A list of tuples representing the possible switch positions
    #     """
    #     return list(itertools.product([0, 1], repeat=len(self.power_switches)))
