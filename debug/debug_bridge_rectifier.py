#!/usr/bin/env python3
"""
Debug script to investigate bridge rectifier initialization issues.

This script creates a bridge rectifier circuit step-by-step and analyzes
why the DAE system initialization is failing.
"""

import sys
import os
import logging

# Add the parent directory to the path so we can import amps_simulation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from amps_simulation.core.electrical_model import ElectricalModel
from amps_simulation.core.dae_system import ElectricalDaeSystem
from amps_simulation.core.components import Resistor, Diode, VoltageSource
import numpy as np

# Set up logging with more verbose output
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def create_bridge_rectifier_minimal():
    """Create a minimal bridge rectifier circuit step by step."""
    print("Creating minimal bridge rectifier circuit...")

    # Clear registries
    from amps_simulation.core.components import Component, ElecJunction
    Component.clear_registry()
    ElecJunction.clear_registry()

    # Create empty electrical model
    model = ElectricalModel()

    # Add nodes (using integer IDs)
    print("Adding nodes...")
    model.add_node(1, is_ground=False)  # ac_pos
    model.add_node(2, is_ground=False)  # ac_neg
    model.add_node(3, is_ground=False)  # dc_pos
    model.add_node(0, is_ground=True)   # dc_neg (ground)

    print(f"Added {len(model.graph.nodes())} nodes: {list(model.graph.nodes())}")

    # Create components
    print("Creating components...")
    v_ac = VoltageSource(comp_id="V_ac", voltage=15.0)
    diode_1 = Diode(comp_id="D1", is_on=False)
    diode_2 = Diode(comp_id="D2", is_on=False)
    diode_3 = Diode(comp_id="D3", is_on=False)
    diode_4 = Diode(comp_id="D4", is_on=False)
    r_load = Resistor(comp_id="R_load", resistance=1000.0)

    print(f"Created components: V_ac, D1, D2, D3, D4, R_load")

    # Add components with terminal connections
    print("Adding components to circuit...")
    model.add_component(v_ac, p=1, n=2)        # V_ac: ac_pos to ac_neg
    model.add_component(diode_1, p=1, n=3)     # D1: ac_pos to dc_pos
    model.add_component(diode_2, p=2, n=3)     # D2: ac_neg to dc_pos
    model.add_component(diode_3, p=0, n=1)     # D3: dc_neg to ac_pos
    model.add_component(diode_4, p=0, n=2)     # D4: dc_neg to ac_neg
    model.add_component(r_load, p=3, n=0)      # R_load: dc_pos to dc_neg

    print(f"Circuit has {model.graph.number_of_edges()} edges")

    # Print circuit topology
    print("\nCircuit topology:")
    for source, target, edge_data in model.graph.edges(data=True):
        component = edge_data.get('component')
        if component:
            comp_type = component.__class__.__name__
            comp_id = component.comp_id
            print(f"  {comp_id} ({comp_type}): {source} -> {target}")

    return model

def debug_electrical_model_initialization():
    """Debug the electrical model initialization step by step."""
    print("\n" + "="*60)
    print("DEBUGGING ELECTRICAL MODEL INITIALIZATION")
    print("="*60)

    model = create_bridge_rectifier_minimal()

    print("\nInitializing electrical model...")
    try:
        model.initialize()
        print("✓ Electrical model initialization PASSED")

        # Print discovered variables
        junction_vars, current_vars, voltage_vars = model.variable_lists()
        print(f"\nVariable counts:")
        print(f"  Junction voltage vars: {len(junction_vars)} - {junction_vars}")
        print(f"  Component current vars: {len(current_vars)} - {current_vars}")
        print(f"  Component voltage vars: {len(voltage_vars)} - {voltage_vars}")

        print(f"\nState vars: {len(model.state_vars)} - {model.state_vars}")
        print(f"Input vars: {len(model.input_vars)} - {model.input_vars}")
        print(f"Output vars: {len(model.output_vars)} - {model.output_vars}")
        print(f"Diode list: {len(model.diode_list)} - {[d.comp_id for d in model.diode_list]}")

        return model

    except Exception as e:
        print(f"✗ Electrical model initialization FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_dae_system_creation(electrical_model):
    """Debug the DAE system creation step."""
    print("\n" + "="*60)
    print("DEBUGGING DAE SYSTEM CREATION")
    print("="*60)

    if electrical_model is None:
        print("✗ Cannot create DAE system - electrical model is None")
        return None

    print("Creating ElectricalDaeSystem...")
    try:
        dae_system = ElectricalDaeSystem(electrical_model)
        print("✓ DAE system creation PASSED")
        return dae_system
    except Exception as e:
        print(f"✗ DAE system creation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_individual_equations(dae_system):
    """Debug computation of individual equation sets."""
    print("\n" + "="*60)
    print("DEBUGGING INDIVIDUAL EQUATION COMPUTATION")
    print("="*60)

    if dae_system is None:
        print("✗ Cannot compute equations - DAE system is None")
        return False

    try:
        # Test KCL equations
        print("\n1. Testing KCL equations...")
        kcl_eqs = dae_system.compute_kcl_equations()
        print(f"✓ KCL equations: {len(kcl_eqs)} equations")
        for i, eq in enumerate(kcl_eqs):
            print(f"    KCL[{i}]: {eq} = 0")

    except Exception as e:
        print(f"✗ KCL equations FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        # Test KVL equations
        print("\n2. Testing KVL equations...")
        kvl_eqs = dae_system.compute_kvl_equations()
        print(f"✓ KVL equations: {len(kvl_eqs)} equations")
        for i, eq in enumerate(kvl_eqs):
            print(f"    KVL[{i}]: {eq} = 0")

    except Exception as e:
        print(f"✗ KVL equations FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        # Test static component equations
        print("\n3. Testing static component equations...")
        static_eqs = dae_system.compute_static_component_equations()
        print(f"✓ Static equations: {len(static_eqs)} equations")
        for i, eq in enumerate(static_eqs):
            print(f"    Static[{i}]: {eq} = 0")

    except Exception as e:
        print(f"✗ Static equations FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        # Test switch equations
        print("\n4. Testing switch equations...")
        switch_eqs = dae_system.compute_switch_equations()
        print(f"✓ Switch equations: {len(switch_eqs)} equations")
        for i, eq in enumerate(switch_eqs):
            print(f"    Switch[{i}]: {eq} = 0")

    except Exception as e:
        print(f"✗ Switch equations FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        # Test diode equations
        print("\n5. Testing diode equations...")
        diode_eqs = dae_system.compute_diode_equations()
        print(f"✓ Diode equations: {len(diode_eqs)} equations")
        for i, eq in enumerate(diode_eqs):
            print(f"    Diode[{i}]: {eq} = 0")

    except Exception as e:
        print(f"✗ Diode equations FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def debug_circuit_equation_solving(dae_system):
    """Debug the circuit equation solving step that's failing."""
    print("\n" + "="*60)
    print("DEBUGGING CIRCUIT EQUATION SOLVING")
    print("="*60)

    if dae_system is None:
        print("✗ Cannot solve equations - DAE system is None")
        return False

    try:
        # Get all equation types
        print("Computing all equation sets...")
        kcl_eqs = dae_system.compute_kcl_equations()
        kvl_eqs = dae_system.compute_kvl_equations()
        static_eqs = dae_system.compute_static_component_equations()
        switch_eqs = dae_system.compute_switch_equations()
        diode_eqs = dae_system.compute_diode_equations()

        print(f"Equation counts: KCL={len(kcl_eqs)}, KVL={len(kvl_eqs)}, Static={len(static_eqs)}, Switch={len(switch_eqs)}, Diode={len(diode_eqs)}")

        # Combine all equations
        all_equations = kcl_eqs + kvl_eqs + static_eqs + switch_eqs + diode_eqs
        print(f"Total equations: {len(all_equations)}")

        # Get variable lists
        junction_voltage_var_list, component_current_var_list, component_voltage_var_list = dae_system.electrical_model.variable_lists()
        junction_voltage_var_list_cleaned = [var for var in junction_voltage_var_list if var != 0]
        combined_vars = junction_voltage_var_list_cleaned + component_current_var_list + component_voltage_var_list

        print(f"Variable counts: Junction={len(junction_voltage_var_list_cleaned)}, Current={len(component_current_var_list)}, Voltage={len(component_voltage_var_list)}")
        print(f"Total combined variables: {len(combined_vars)}")

        # Get input and state variables to exclude
        if dae_system.electrical_model.initialized:
            input_vars = dae_system.electrical_model.input_vars
            state_vars = dae_system.electrical_model.state_vars
        else:
            input_vars = dae_system.electrical_model.find_input_vars()
            state_vars = dae_system.electrical_model.find_state_vars()

        print(f"Input vars to exclude: {input_vars}")
        print(f"State vars to exclude: {state_vars}")

        excluded = set(input_vars) | set(state_vars)
        vars_to_solve = [var for var in combined_vars if var not in excluded]

        print(f"Variables to solve: {len(vars_to_solve)}")
        print(f"Equation/variable balance: {len(all_equations)} equations, {len(vars_to_solve)} variables")

        if len(all_equations) != len(vars_to_solve):
            print("✗ EQUATION/VARIABLE MISMATCH!")
            print("\nDetailed analysis:")
            print("All equations:")
            for i, eq in enumerate(all_equations):
                print(f"  [{i:2d}] {eq} = 0")
            print("\nAll variables to solve:")
            for i, var in enumerate(vars_to_solve):
                print(f"  [{i:2d}] {var}")
            print("\nExcluded variables:")
            for var in excluded:
                print(f"  - {var}")
            return False

        # Try to solve the system
        print("\nAttempting to solve the equation system...")
        solution = dae_system._solve_circuit_equations_safe(all_equations, vars_to_solve)

        if solution is None:
            print("✗ Equation system has no solution")

            # Try with smaller subsets to isolate the problem
            print("\nTrying smaller equation subsets...")

            # Try just KCL + KVL
            subset_eqs = kcl_eqs + kvl_eqs
            subset_vars = vars_to_solve[:len(subset_eqs)] if len(subset_eqs) <= len(vars_to_solve) else vars_to_solve
            print(f"Testing KCL+KVL: {len(subset_eqs)} equations, {len(subset_vars)} variables")
            subset_solution = dae_system._solve_circuit_equations_safe(subset_eqs, subset_vars)
            if subset_solution:
                print("✓ KCL+KVL subset solvable")
            else:
                print("✗ KCL+KVL subset not solvable")

            return False
        else:
            print(f"✓ Found solution with {len(solution)} variables")
            return True

    except Exception as e:
        print(f"✗ Circuit equation solving FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_lcp_formulation_without_init(dae_system):
    """Test LCP matrix computation without DAE system initialization."""
    print("\n" + "="*60)
    print("DEBUGGING LCP FORMULATION WITHOUT INITIALIZATION")
    print("="*60)

    if dae_system is None:
        print("✗ Cannot test LCP - DAE system is None")
        return False

    print("Testing compute_diode_lcp_matrices() directly (bypasses initialization)...")

    try:
        # Set up test conditions for bridge rectifier
        state_values = np.array([])  # No inductors or capacitors
        input_values = np.array([15.0])  # V_ac = 15V

        print(f"Test conditions:")
        print(f"  State values: {state_values}")
        print(f"  Input values: {input_values}")
        print(f"  Diodes in circuit: {len(dae_system.electrical_model.diode_list)}")

        # Try to compute LCP matrices directly
        print("\nAttempting to compute LCP matrices...")
        M_matrix, q_vector = dae_system.compute_diode_lcp_matrices(state_values, input_values)

        print(f"\nLCP Results:")
        print(f"  M matrix shape: {M_matrix.shape}")
        print(f"  q vector shape: {q_vector.shape}")

        if M_matrix.shape[0] > 0:
            print(f"✓ LCP matrices successfully computed!")

            # Convert to numpy for analysis
            M_np = np.array(M_matrix.tolist(), dtype=float)
            q_np = np.array(q_vector.tolist(), dtype=float).flatten()

            print(f"\nMatrix properties:")
            print(f"  M matrix:\n{M_np}")
            print(f"  q vector: {q_np}")

            # Check if square matrix
            if M_np.shape[0] == M_np.shape[1]:
                det_M = np.linalg.det(M_np)
                cond_M = np.linalg.cond(M_np)
                print(f"  Determinant: {det_M:.2e}")
                print(f"  Condition number: {cond_M:.2e}")

                if abs(det_M) > 1e-10:
                    print("  ✓ M matrix is well-conditioned for LCP solving")

                    # Try to use the LCP solver to get diode states
                    print("\nTesting LCP solver...")
                    try:
                        diode_names = [diode.comp_id for diode in dae_system.electrical_model.diode_list]
                        conducting_states, info = dae_system.lcp_solver.detect_diode_states(M_np, q_np, diode_names)

                        print(f"LCP solver results:")
                        print(f"  Converged: {info['converged']}")
                        print(f"  Pivots: {info['pivots']}")
                        if info['converged']:
                            print(f"  Complementarity: {info['complementarity']:.2e}")
                            print(f"  Diode states: {conducting_states}")
                            for diode, state in zip(dae_system.electrical_model.diode_list, conducting_states):
                                print(f"    {diode.comp_id}: {'CONDUCTING' if state else 'BLOCKING'}")

                            # Test the complete diode state detection method
                            print("\nTesting complete diode state detection method...")
                            try:
                                detected_states = dae_system._detect_diode_states_lcp(state_values, input_values)
                                print(f"✓ Complete LCP detection successful!")
                                print(f"  Detected states: {detected_states}")
                                for i, (diode, state) in enumerate(zip(dae_system.electrical_model.diode_list, detected_states)):
                                    print(f"    {diode.comp_id}: {'CONDUCTING' if state else 'BLOCKING'}")

                                return True
                            except Exception as lcp_detect_error:
                                print(f"✗ Complete LCP detection failed: {lcp_detect_error}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print("  ⚠ LCP solver did not converge")
                    except Exception as lcp_solve_error:
                        print(f"✗ LCP solver failed: {lcp_solve_error}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("  ⚠ M matrix is singular or ill-conditioned")
            else:
                print(f"  ⚠ M matrix is not square ({M_np.shape[0]}x{M_np.shape[1]})")
        else:
            print("✗ Empty LCP matrices returned")

        return False

    except Exception as e:
        print(f"✗ LCP formulation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_dae_system_initialization(dae_system):
    """Debug the full DAE system initialization that's failing."""
    print("\n" + "="*60)
    print("DEBUGGING DAE SYSTEM INITIALIZATION")
    print("="*60)

    if dae_system is None:
        print("✗ Cannot initialize - DAE system is None")
        return False

    print("Attempting DAE system initialization...")
    try:
        dae_system.initialize()
        print("✓ DAE system initialization PASSED")
        return True
    except Exception as e:
        print(f"✗ DAE system initialization FAILED: {e}")
        print("This is expected - we've identified the root cause in equation solving")
        return False

def main():
    """Main debug function to run all tests step by step."""

    print("BRIDGE RECTIFIER DEBUG SCRIPT")
    print("="*60)
    print("Investigating bridge rectifier initialization failure step by step")

    # Step 1: Debug electrical model initialization
    electrical_model = debug_electrical_model_initialization()
    if electrical_model is None:
        print("\n✗ CRITICAL: Electrical model initialization failed - cannot continue")
        return

    # Step 2: Debug DAE system creation
    dae_system = debug_dae_system_creation(electrical_model)
    if dae_system is None:
        print("\n✗ CRITICAL: DAE system creation failed - cannot continue")
        return

    # Step 3: Debug individual equation computation
    equations_ok = debug_individual_equations(dae_system)
    if not equations_ok:
        print("\n✗ CRITICAL: Individual equation computation failed - cannot continue")
        return

    # Step 4: Debug circuit equation solving (this is likely where it fails)
    solving_ok = debug_circuit_equation_solving(dae_system)
    if not solving_ok:
        print("\n✗ CRITICAL: Circuit equation solving failed - this is the root cause")
        print("But let's test if LCP formulation can work independently...")

        # Step 4.5: Test LCP formulation without initialization
        lcp_ok = debug_lcp_formulation_without_init(dae_system)
        if lcp_ok:
            print("\n✓ SUCCESS: LCP formulation works independently of initialization!")
            print("The LCP approach bypasses the initialization problem")
        else:
            print("\n✗ LCP formulation also fails")

    # Step 5: Debug full DAE system initialization (expected to fail)
    print("\nTesting full DAE system initialization (expected to fail)...")
    init_ok = debug_dae_system_initialization(dae_system)

    print("\n" + "="*60)
    print("DEBUG SUMMARY")
    print("="*60)
    print("✓ Electrical model initialization: WORKS")
    print("✓ DAE system creation: WORKS")
    print("✓ Individual equation computation: WORKS")
    print("✗ Circuit equation solving: FAILS (root cause)")
    print("? LCP formulation: TEST RESULTS ABOVE")
    print("✗ Full DAE initialization: FAILS (expected)")
    print("="*60)

if __name__ == "__main__":
    main()