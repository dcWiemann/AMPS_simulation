"""
Comprehensive test suite for diode mode detection using LCP solver.

This test suite verifies diode conducting/blocking state detection for:
1. Voltage source + resistor + diode circuit (V-R-D)
2. Capacitor + diode circuit (C-D)
3. Two diodes in series circuit (V-D-D-R)
4. Bridge rectifier circuit (4 diodes with AC input and DC filtering)

All circuits are built programmatically using ElectricalModel.add_node() and
add_component() methods. The Engine class initializes the model and provides
access to the DAE model for diode state detection.

Diode states are represented as boolean values:
- True = CONDUCTING (diode acts as short circuit, v_D = 0)
- False = BLOCKING (diode acts as open circuit, i_D = 0)
"""

import pytest
import numpy as np
import os
import sys

# Add project to path for test environment
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from amps_simulation.core.electrical_model import ElectricalModel
from amps_simulation.core.components import (
    VoltageSource,
    Resistor,
    Diode,
    Capacitor,
    Component,
    ElecJunction
)
from amps_simulation.core.engine import Engine


class TestDiodeIntegration:
    """Comprehensive test suite for diode mode detection in various circuits."""

    def setup_method(self):
        """Clear component and junction registries before each test."""
        Component.clear_registry()
        ElecJunction.clear_registry()

    def teardown_method(self):
        """Clear component and junction registries after each test."""
        Component.clear_registry()
        ElecJunction.clear_registry()

    # ========================================================================
    # CIRCUIT 1: Voltage Source + Resistor + Diode
    # ========================================================================

    def build_voltage_resistor_diode_circuit(self, v_input: float) -> tuple:
        """
        Build V-R-D circuit: V1(+) ---R1(1Ω)--- ---D1|>--- GND

        Args:
            v_input: Input voltage value (V)

        Returns:
            tuple: (engine, state_values, input_values)
        """
        model = ElectricalModel()

        # Add nodes (0 = ground)
        model.add_node(0, is_ground=True)
        model.add_node(1)  # Voltage source positive terminal
        model.add_node(2)  # Node between resistor and diode

        # Create components
        v1 = VoltageSource(comp_id="V1", voltage=v_input)
        r1 = Resistor(comp_id="R1", resistance=1.0)
        d1 = Diode(comp_id="D1")

        # Add components to circuit
        model.add_component(v1, [1, 0])  # Voltage source: GND to node 1
        model.add_component(r1, [1, 2])  # Resistor: node 1 to node 2
        model.add_component(d1, [2, 0])  # Diode: node 2 to GND (anode at node 2)

        model.initialize()

        # Initialize engine
        engine = Engine(model)
        engine.initialize()

        # Prepare state and input arrays
        n_states = len(engine.state_vars)
        state_values = np.zeros(n_states)
        input_values = np.array([v_input])

        return engine, state_values, input_values

    def test_vrd_positive_voltage(self):
        """
        Test V-R-D circuit with V_in = +5V.

        Expected: Diode should be CONDUCTING (v_D = 0).
        With positive voltage, current flows through R1 and forward-biases D1.
        """
        engine, state_values, input_values = self.build_voltage_resistor_diode_circuit(5.0)

        # Detect diode states
        dae_system = engine.electrical_dae_system
        diode_states = dae_system.detect_diode_states(state_values, input_values)

        # Verify results
        assert len(diode_states) == 1, f"Expected 1 diode, found {len(diode_states)}"
        assert diode_states[0] == True, "Diode D1 should be CONDUCTING with +5V input"

        print(f"\n[V-R-D +5V] Diode D1 state: {'CONDUCTING (v_D=0)' if diode_states[0] else 'BLOCKING (i_D=0)'}")

    def test_vrd_negative_voltage(self):
        """
        Test V-R-D circuit with V_in = -5V.

        Expected: Diode should be BLOCKING (i_D = 0).
        With negative voltage, the diode is reverse-biased and blocks current.
        """
        engine, state_values, input_values = self.build_voltage_resistor_diode_circuit(-5.0)

        # Detect diode states
        dae_system = engine.electrical_dae_system
        diode_states = dae_system.detect_diode_states(state_values, input_values)

        # Verify results
        assert len(diode_states) == 1, f"Expected 1 diode, found {len(diode_states)}"
        assert diode_states[0] == False, "Diode D1 should be BLOCKING with -5V input"

        print(f"\n[V-R-D -5V] Diode D1 state: {'CONDUCTING (v_D=0)' if diode_states[0] else 'BLOCKING (i_D=0)'}")

    def test_vrd_zero_voltage(self):
        """
        Test V-R-D circuit with V_in = 0V.

        Expected: Diode should be BLOCKING (i_D = 0).
        With zero voltage, there is no forward bias and the diode blocks.
        """
        engine, state_values, input_values = self.build_voltage_resistor_diode_circuit(0.0)

        # Detect diode states
        dae_system = engine.electrical_dae_system
        diode_states = dae_system.detect_diode_states(state_values, input_values)

        # Verify results
        assert len(diode_states) == 1, f"Expected 1 diode, found {len(diode_states)}"
        assert diode_states[0] == False, "Diode D1 should be BLOCKING with 0V input"

        print(f"\n[V-R-D 0V] Diode D1 state: {'CONDUCTING (v_D=0)' if diode_states[0] else 'BLOCKING (i_D=0)'}")

    # ========================================================================
    # CIRCUIT 2: Capacitor + Diode
    # ========================================================================

    def build_capacitor_diode_circuit(self, v_cap: float) -> tuple:
        """
        Build C-D-R circuit: C1 ---D1|>--- R1(1Ω) --- GND

        Args:
            v_cap: Initial capacitor voltage (V)

        Returns:
            tuple: (engine, state_values, input_values)
        """
        model = ElectricalModel()

        # Add nodes (0 = ground)
        model.add_node(0, is_ground=True)
        model.add_node(1)  # Capacitor positive terminal
        model.add_node(2)  # Node between diode and resistor

        # Create components
        c1 = Capacitor(comp_id="C1", capacitance=1e-3)  # 1 mF
        d1 = Diode(comp_id="D1")
        r1 = Resistor(comp_id="R1", resistance=1.0)

        # Add components to circuit
        model.add_component(c1, [1, 0])  # Capacitor: GND to node 1
        model.add_component(d1, [1, 2])  # Diode: node 1 to node 2 (anode at node 1)
        model.add_component(r1, [2, 0])  # Resistor: node 2 to GND

        model.initialize()

        # Initialize engine
        engine = Engine(model)
        engine.initialize()

        # Set capacitor voltage as state variable
        n_states = len(engine.state_vars)
        assert n_states == 1, f"Expected 1 state variable (capacitor), found {n_states}"
        state_values = np.array([v_cap])

        # No voltage sources in this circuit
        n_inputs = len(engine.input_vars)
        input_values = np.zeros(n_inputs) if n_inputs > 0 else np.array([])

        return engine, state_values, input_values

    def test_capacitor_diode_positive_voltage(self):
        """
        Test C-D-R circuit with v_C = +5V.

        Expected: Diode mode depends on circuit dynamics.
        With positive capacitor voltage, the diode is forward-biased and should conduct,
        allowing the capacitor to discharge through R1.
        """
        engine, state_values, input_values = self.build_capacitor_diode_circuit(5.0)

        # Detect diode states
        dae_system = engine.electrical_dae_system
        diode_states = dae_system.detect_diode_states(state_values, input_values)

        # Verify results
        assert len(diode_states) == 1, f"Expected 1 diode, found {len(diode_states)}"
        assert diode_states[0] == True, "Diode D1 should be CONDUCTING with +5V capacitor voltage"

        print(f"\n[C-D v_C=+5V] Diode D1 state: {'CONDUCTING (v_D=0)' if diode_states[0] else 'BLOCKING (i_D=0)'}")
        print(f"  Expected: CONDUCTING (capacitor discharges through diode and resistor)")

    def test_capacitor_diode_negative_voltage(self):
        """
        Test C-D-R circuit with v_C = -5V.

        Expected: Diode should be BLOCKING.
        With negative capacitor voltage, the diode is reverse-biased and blocks current.
        """
        engine, state_values, input_values = self.build_capacitor_diode_circuit(-5.0)

        # Detect diode states
        dae_system = engine.electrical_dae_system
        diode_states = dae_system.detect_diode_states(state_values, input_values)

        # Verify results
        assert len(diode_states) == 1, f"Expected 1 diode, found {len(diode_states)}"
        assert diode_states[0] == False, "Diode D1 should be BLOCKING with -5V capacitor voltage"

        print(f"\n[C-D v_C=-5V] Diode D1 state: {'CONDUCTING (v_D=0)' if diode_states[0] else 'BLOCKING (i_D=0)'}")
        print(f"  Expected: BLOCKING (diode is reverse-biased)")

    def test_capacitor_diode_zero_voltage(self):
        """
        Test C-D-R circuit with v_C = 0V.

        Expected: Diode should be BLOCKING.
        With zero capacitor voltage, there is no forward bias and the diode blocks.
        """
        engine, state_values, input_values = self.build_capacitor_diode_circuit(0.0)

        # Detect diode states
        dae_system = engine.electrical_dae_system
        diode_states = dae_system.detect_diode_states(state_values, input_values)

        # Verify results
        assert len(diode_states) == 1, f"Expected 1 diode, found {len(diode_states)}"
        assert diode_states[0] == False, "Diode D1 should be BLOCKING with 0V capacitor voltage"

        print(f"\n[C-D v_C=0V] Diode D1 state: {'CONDUCTING (v_D=0)' if diode_states[0] else 'BLOCKING (i_D=0)'}")
        print(f"  Expected: BLOCKING (no forward bias)")

    # ========================================================================
    # CIRCUIT 3: Two Diodes in Series
    # ========================================================================

    def build_two_diodes_circuit(self, v_input: float) -> tuple:
        """
        Build V-D-D-R circuit: V1(+) ---D1|>--- ---D2|>--- R1(1Ω) --- GND

        Args:
            v_input: Input voltage value (V)

        Returns:
            tuple: (engine, state_values, input_values)
        """
        model = ElectricalModel()

        # Add nodes (0 = ground)
        model.add_node(0, is_ground=True)
        model.add_node(1)  # Voltage source positive terminal
        model.add_node(2)  # Node between D1 and D2
        model.add_node(3)  # Node between D2 and R1

        # Create components
        v1 = VoltageSource(comp_id="V1", voltage=v_input)
        d1 = Diode(comp_id="D1")
        d2 = Diode(comp_id="D2")
        r1 = Resistor(comp_id="R1", resistance=1.0)

        # Add components to circuit
        model.add_component(v1, [1, 0])  # Voltage source: node 1 to GND
        model.add_component(d1, [1, 2])  # Diode D1: node 1 to node 2
        model.add_component(d2, [2, 3])  # Diode D2: node 2 to node 3
        model.add_component(r1, [3, 0])  # Resistor: node 3 to GND

        model.initialize()

        # Initialize engine
        engine = Engine(model)

        # Prepare state and input arrays (no state variables in this circuit)
        state_values = np.array([])
        input_values = np.array([v_input])

        engine.initialize(initial_conditions=None, initial_inputs=input_values)

        return engine, state_values, input_values

    def test_two_diodes_positive_voltage(self):
        """
        Test two diodes in series with V_in = +5V.

        Expected: Both diodes should be CONDUCTING.
        With positive voltage, current flows through both diodes in series,
        and both are forward-biased.
        """
        engine, state_values, input_values = self.build_two_diodes_circuit(5.0)

        # Detect diode states
        dae_system = engine.electrical_dae_system
        diode_states = dae_system.detect_diode_states(state_values, input_values)

        # Verify results
        assert len(diode_states) == 2, f"Expected 2 diodes, found {len(diode_states)}"
        assert diode_states[0] == True, "Diode D1 should be CONDUCTING with +5V input"
        assert diode_states[1] == True, "Diode D2 should be CONDUCTING with +5V input"

        print(f"\n[Two Diodes +5V] D1={'CONDUCTING' if diode_states[0] else 'BLOCKING'}, "
              f"D2={'CONDUCTING' if diode_states[1] else 'BLOCKING'}")

    def test_two_diodes_negative_voltage(self):
        """
        Test two diodes in series with V_in = -5V.

        Expected: Both diodes should be BLOCKING.
        With negative voltage, both diodes are reverse-biased and block current.
        """
        engine, state_values, input_values = self.build_two_diodes_circuit(-5.0)

        # Detect diode states
        dae_system = engine.electrical_dae_system
        diode_states = dae_system.detect_diode_states(state_values, input_values)

        # Verify results
        assert len(diode_states) == 2, f"Expected 2 diodes, found {len(diode_states)}"
        assert diode_states[0] == False, "Diode D1 should be BLOCKING with -5V input"
        assert diode_states[1] == False, "Diode D2 should be BLOCKING with -5V input"

        print(f"\n[Two Diodes -5V] D1={'CONDUCTING' if diode_states[0] else 'BLOCKING'}, "
              f"D2={'CONDUCTING' if diode_states[1] else 'BLOCKING'}")

    # ========================================================================
    # CIRCUIT 4: Bridge Rectifier
    # ========================================================================

    def build_bridge_rectifier_circuit(self, v_input: float, v_cap_initial: float = 0.0) -> tuple:
        """
        Build bridge rectifier circuit with AC input and DC filtering.

        AC side: AC+ ---V1(AC)--- AC_GND
        DC side: DC+ ---R_load(10Ω)--- DC_GND
                 DC+ ---C_filter(100μF)--- DC_GND

        When V_in > 0: D1 and D4 should conduct (current flows AC+ -> D1 -> DC+ -> Load -> DC_GND -> D4 -> AC_GND)
        When V_in < 0: D2 and D3 should conduct (current flows AC_GND -> D2 -> DC+ -> Load -> DC_GND -> D3 -> AC+)
        When V_in = 0: All diodes should block

        Args:
            v_input: AC input voltage value (V)
            v_cap_initial: Initial capacitor voltage (V)

        Returns:
            tuple: (engine, state_values, input_values)
        """
        model = ElectricalModel()

        # Add nodes
        model.add_node(0, is_ground=True)  # AC ground
        model.add_node(1)  # AC + terminal
        model.add_node(2)  # DC + resistor
        model.add_node(3)  # between resistor and capacitor
        model.add_node(4)  # DC ground bus

        # Create components
        v1 = VoltageSource(comp_id="V1", voltage=v_input)
        d1 = Diode(comp_id="D1")  # AC+ to DC+
        d2 = Diode(comp_id="D2")  # DC+ to AC_GND
        d3 = Diode(comp_id="D3")  # AC+ to DC_GND
        d4 = Diode(comp_id="D4")  # DC_GND to AC_GND
        r_load = Resistor(comp_id="R_load", resistance=0.1) # 0.1 Ω Vorwiderstand
        c_filter = Capacitor(comp_id="C_filter", capacitance=100e-6)  # 100 μF

        # Add AC source
        model.add_component(v1, [1, 0])  # Voltage source: AC+ to AC_GND

        # Add bridge diodes
        # When V_in > 0: Current path is AC+ (node 1) -> D1 -> DC+ (node 2) -> Load -> DC_GND (node 3) -> D4 -> AC_GND (node 0)
        # When V_in < 0: Current path is AC_GND (node 0) -> D2 -> DC+ (node 2) -> Load -> DC_GND (node 3) -> D3 -> AC+ (node 1)
        model.add_component(d1, [1, 2])  # D1: AC+ to DC+ (anode at AC+)
        model.add_component(d2, [0, 2])  # D2: AC_GND to DC+ (anode at AC_GND)
        model.add_component(d3, [4, 1])  # D3: DC_GND to AC+ (anode at DC_GND)
        model.add_component(d4, [4, 0])  # D4: DC_GND to AC_GND (anode at DC_GND)

        # Add DC side load and filter
        model.add_component(r_load, [2, 3])     # Load resistor: DC+ to DC_GND
        model.add_component(c_filter, [3, 4])   # Filter capacitor: DC+ to DC_GND

        model.initialize()

        # Initialize engine
        engine = Engine(model)
        engine.initialize(initial_conditions=np.array([v_cap_initial]), initial_inputs=np.array([v_input]))

        # Set capacitor voltage as state variable
        n_states = len(engine.state_vars)
        assert n_states == 1, f"Expected 1 state variable (filter capacitor), found {n_states}"
        state_values = np.array([v_cap_initial])

        # Set input voltage
        input_values = np.array([v_input])

        return engine, state_values, input_values

    def test_bridge_rectifier_positive_voltage(self):
        """
        Test bridge rectifier with V_in = +5V and v_C = 0V.

        Expected: D1 and D4 should be CONDUCTING, D2 and D3 should be BLOCKING.
        Current path: AC+ -> D1 -> DC+ -> Load -> DC_GND -> D4 -> AC_GND
        """
        engine, state_values, input_values = self.build_bridge_rectifier_circuit(5.0, 0.0)

        # Detect diode states
        dae_system = engine.electrical_dae_system
        diode_states = dae_system.detect_diode_states(state_values, input_values)

        # Verify results
        assert len(diode_states) == 4, f"Expected 4 diodes in bridge, found {len(diode_states)}"

        # Count conducting diodes
        num_conducting = sum(diode_states)

        print(f"\n[Bridge +5V] D1={'ON' if diode_states[0] else 'OFF'}, "
              f"D2={'ON' if diode_states[1] else 'OFF'}, "
              f"D3={'ON' if diode_states[2] else 'OFF'}, "
              f"D4={'ON' if diode_states[3] else 'OFF'}")
        print(f"  Expected: D1=ON, D2=OFF, D3=OFF, D4=ON (2 diodes conducting)")

        # Exactly 2 diodes should be conducting
        assert num_conducting == 2, f"Expected 2 conducting diodes, found {num_conducting}"

        # Specifically, D1 and D4 should conduct for positive input
        # assert diode_states[0] == True, "D1 should conduct with positive input"
        # assert diode_states[3] == True, "D4 should conduct with positive input"
        # assert diode_states[1] == False, "D2 should block with positive input"
        # assert diode_states[2] == False, "D3 should block with positive input"

        # Check the ordering of diodes
        diode_list = engine.electrical_model.diode_list
        # find index of D1, D2, D3, D4 in diode_list
        d1_index = diode_list.index(next(d for d in diode_list if d.comp_id == "D1"))
        d2_index = diode_list.index(next(d for d in diode_list if d.comp_id == "D2"))
        d3_index = diode_list.index(next(d for d in diode_list if d.comp_id == "D3"))
        d4_index = diode_list.index(next(d for d in diode_list if d.comp_id == "D4"))
        assert diode_states[d1_index] == True, "D1 should conduct with positive input"
        assert diode_states[d2_index] == False, "D2 should block with positive input"
        assert diode_states[d3_index] == False, "D3 should block with positive input"
        assert diode_states[d4_index] == True, "D4 should conduct with positive input"

    def test_bridge_rectifier_negative_voltage(self):
        """
        Test bridge rectifier with V_in = -5V and v_C = 0V.

        Expected: D2 and D3 should be CONDUCTING, D1 and D4 should be BLOCKING.
        Current path: AC_GND -> D2 -> DC+ -> Load -> DC_GND -> D3 -> AC+
        """
        engine, state_values, input_values = self.build_bridge_rectifier_circuit(-5.0, 0.0)

        # Detect diode states
        dae_system = engine.electrical_dae_system
        diode_states = dae_system.detect_diode_states(state_values, input_values)

        # Verify results
        assert len(diode_states) == 4, f"Expected 4 diodes in bridge, found {len(diode_states)}"

        # Count conducting diodes
        num_conducting = sum(diode_states)

        print(f"\n[Bridge -5V] D1={'ON' if diode_states[0] else 'OFF'}, "
              f"D2={'ON' if diode_states[1] else 'OFF'}, "
              f"D3={'ON' if diode_states[2] else 'OFF'}, "
              f"D4={'ON' if diode_states[3] else 'OFF'}")
        print(f"  Expected: D1=OFF, D2=ON, D3=ON, D4=OFF (2 diodes conducting)")

        # Exactly 2 diodes should be conducting
        assert num_conducting == 2, f"Expected 2 conducting diodes, found {num_conducting}"

        # Check the ordering of diodes
        diode_list = engine.electrical_model.diode_list

        # find index of D1, D2, D3, D4 in diode_list
        d1_index = diode_list.index(next(d for d in diode_list if d.comp_id == "D1"))
        d2_index = diode_list.index(next(d for d in diode_list if d.comp_id == "D2"))
        d3_index = diode_list.index(next(d for d in diode_list if d.comp_id == "D3"))
        d4_index = diode_list.index(next(d for d in diode_list if d.comp_id == "D4"))
        assert diode_states[d1_index] == False, "D1 should block with negative input"
        assert diode_states[d2_index] == True, "D2 should conduct with negative input"
        assert diode_states[d3_index] == True, "D3 should conduct with negative input"
        assert diode_states[d4_index] == False, "D4 should block with negative input"


    def test_bridge_rectifier_zero_voltage(self):
        """
        Test bridge rectifier with V_in = 0V and v_C = 0V.

        Expected: All diodes should be BLOCKING.
        With zero input and zero capacitor voltage, there is no forward bias
        on any diode.
        """
        engine, state_values, input_values = self.build_bridge_rectifier_circuit(0.0, 0.0)

        # Detect diode states
        dae_system = engine.electrical_dae_system
        diode_states = dae_system.detect_diode_states(state_values, input_values)

        # Verify results
        assert len(diode_states) == 4, f"Expected 4 diodes in bridge, found {len(diode_states)}"

        # Count conducting diodes
        num_conducting = sum(diode_states)

        print(f"\n[Bridge 0V] D1={'ON' if diode_states[0] else 'OFF'}, "
              f"D2={'ON' if diode_states[1] else 'OFF'}, "
              f"D3={'ON' if diode_states[2] else 'OFF'}, "
              f"D4={'ON' if diode_states[3] else 'OFF'}")
        print(f"  Expected: All diodes OFF (0 diodes conducting)")

        # All diodes should be blocking
        assert num_conducting == 0, f"Expected 0 conducting diodes, found {num_conducting}"
        assert all(state == False for state in diode_states), "All diodes should be blocking with 0V input"


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v", "-s"])
