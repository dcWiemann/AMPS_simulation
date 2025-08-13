#!/usr/bin/env python3
"""
Test the full simulation workflow on 4 identical circuit variants.
Tests complete pipeline: parsing, engine initialization, simulation, and output validation.
Verifies that all circuits produce identical results with proper final values.
"""

import pytest
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from amps_simulation.core.parser import ParserJson
from amps_simulation.core.engine import Engine
from amps_simulation.core.components import Component


class TestFullWorkflow:
    """Test complete simulation workflow on circuit variants."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear component registry before each test."""
        Component.clear_registry()
    
    def load_circuit(self, filename):
        """Load circuit from JSON file."""
        with open(f"test_data/{filename}", 'r') as f:
            return json.load(f)
    
    def run_circuit_simulation(self, circuit_filename):
        """Run complete simulation workflow for a circuit file."""
        # Load and parse circuit
        circuit_data = self.load_circuit(circuit_filename)
        parser = ParserJson()
        graph, control_graph = parser.parse(circuit_data)
        
        # Create and initialize engine
        engine = Engine(graph, control_graph)
        engine.initialize()
        
        # Run simulation for 5 seconds to reach steady state
        t_span = (0.0, 5.0)
        initial_conditions = np.zeros(len(engine.state_vars))
        
        result = engine.run_simulation(
            t_span=t_span,
            initial_conditions=initial_conditions,
            method='RK45',
            max_step=0.01
        )
        
        return {
            'engine': engine,
            'result': result,
            'circuit_name': circuit_filename.replace('.json', '')
        }
    
    def save_simulation_plot(self, sim_data, results_dir):
        """Save simulation results as PNG plot."""
        engine = sim_data['engine']
        result = sim_data['result']
        circuit_name = sim_data['circuit_name']
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Circuit Simulation Results: {circuit_name}', fontsize=16)
        
        # Plot state variables (inductor current and capacitor voltage)
        if len(engine.state_vars) >= 2:
            # Inductor current
            axes[0, 0].plot(result['t'], result['y'][0, :], 'b-', linewidth=2)
            axes[0, 0].set_title(f'Inductor Current: {engine.state_vars[0]}')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Current (A)')
            axes[0, 0].grid(True)
            axes[0, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Switch Event')
            axes[0, 0].legend()
            
            # Capacitor voltage
            axes[0, 1].plot(result['t'], result['y'][1, :], 'g-', linewidth=2)
            axes[0, 1].set_title(f'Capacitor Voltage: {engine.state_vars[1]}')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Voltage (V)')
            axes[0, 1].grid(True)
            axes[0, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Switch Event')
            axes[0, 1].legend()
        
        # Plot output variables (ammeters and voltmeter)
        if 'out' in result and len(result['out']) > 0:
            output_data = result['out']
            
            # First ammeter
            if len(output_data) >= 1:
                axes[1, 0].plot(result['t'], output_data[0, :], 'm-', linewidth=2)
                axes[1, 0].set_title(f'Ammeter 1: {engine.output_vars[0]}')
                axes[1, 0].set_xlabel('Time (s)')
                axes[1, 0].set_ylabel('Current (A)')
                axes[1, 0].grid(True)
                axes[1, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Switch Event')
                axes[1, 0].legend()
            
            # Second ammeter/voltmeter
            if len(output_data) >= 2:
                axes[1, 1].plot(result['t'], output_data[1, :], 'c-', linewidth=2)
                axes[1, 1].set_title(f'Output 2: {engine.output_vars[1]}')
                axes[1, 1].set_xlabel('Time (s)')
                if 'VM' in str(engine.output_vars[1]):
                    axes[1, 1].set_ylabel('Voltage (V)')
                else:
                    axes[1, 1].set_ylabel('Current (A)')
                axes[1, 1].grid(True)
                axes[1, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Switch Event')
                axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(results_dir, f'{circuit_name}_simulation.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def test_full_workflow_all_circuits(self):
        """Test complete workflow on all 4 circuit variants."""
        circuit_files = [
            'full_var0.json',
            'full_var1.json',
            'full_var2.json', 
            'full_var3.json'
        ]
        
        results_dir = 'results'
        simulation_results = []
        
        # Run simulations for all circuits
        for circuit_file in circuit_files:
            print(f"Running simulation for {circuit_file}...")
            sim_data = self.run_circuit_simulation(circuit_file)
            simulation_results.append(sim_data)
            
            # Save plot
            plot_path = self.save_simulation_plot(sim_data, results_dir)
            print(f"Saved plot: {plot_path}")
        
        # Validate results
        self.validate_simulation_results(simulation_results)
        
        print("All circuit simulations completed successfully!")
    
    def validate_simulation_results(self, simulation_results):
        """Validate that all circuits produce expected and identical results."""
        # Expected tolerances
        voltage_tolerance = 0.1  # ±0.1V
        current_tolerance = 0.1  # ±0.1A
        
        # Expected final values
        expected_final_voltage = 5.0  # V
        expected_final_current = 2.5  # A
        
        final_voltages = []
        final_currents = []
        
        for i, sim_data in enumerate(simulation_results):
            engine = sim_data['engine']
            result = sim_data['result']
            circuit_name = sim_data['circuit_name']
            
            print(f"\nValidating {circuit_name}:")
            print(f"  State variables: {engine.state_vars}")
            print(f"  Output variables: {engine.output_vars}")
            print(f"  Final time: {result['t'][-1]:.3f}s")
            
            # Get final state values
            final_state = result['y'][:, -1]
            print(f"  Final states: {final_state}")
            
            # Find voltmeter output (should be ~5V)
            voltmeter_idx = None
            for j, output_var in enumerate(engine.output_vars):
                if 'VM' in str(output_var):
                    voltmeter_idx = j
                    break
            
            if voltmeter_idx is not None and 'out' in result:
                final_voltage = abs(result['out'][voltmeter_idx, -1])
                final_voltages.append(final_voltage)
                print(f"  Final voltage (VM): {final_voltage:.4f}V")
                
                # Validate voltage
                assert abs(final_voltage - expected_final_voltage) <= voltage_tolerance, \
                    f"{circuit_name}: Final voltage {final_voltage:.4f}V not within {expected_final_voltage} ± {voltage_tolerance}V"
            
            # Find ammeter outputs (should be ~2.5A)
            for j, output_var in enumerate(engine.output_vars):
                if 'AM' in str(output_var) and 'out' in result:
                    final_current = abs(result['out'][j, -1])
                    final_currents.append(final_current)
                    print(f"  Final current ({output_var}): {final_current:.4f}A")
                    
                    # Validate current
                    assert abs(final_current - expected_final_current) <= current_tolerance, \
                        f"{circuit_name}: Final current {final_current:.4f}A not within {expected_final_current} ± {current_tolerance}A"
        
        # Verify all circuits produce identical results
        if len(final_voltages) > 1:
            voltage_std = np.std(final_voltages)
            print(f"\nVoltage consistency across circuits: std = {voltage_std:.6f}V")
            assert voltage_std < voltage_tolerance/2, \
                f"Voltage results not consistent across circuits: std={voltage_std:.6f}V > {voltage_tolerance/2}V"
        
        if len(final_currents) > 1:
            current_std = np.std(final_currents)
            print(f"Current consistency across circuits: std = {current_std:.6f}A")
            assert current_std < current_tolerance/2, \
                f"Current results not consistent across circuits: std={current_std:.6f}A > {current_tolerance/2}A"
        
        print(f"\n[+] All validations passed!")
        print(f"[+] Final voltages: {[f'{v:.4f}V' for v in final_voltages]}")
        print(f"[+] Final currents: {[f'{c:.4f}A' for c in final_currents]}")


if __name__ == "__main__":
    # Run the test directly
    test_instance = TestFullWorkflow()
    test_instance.setup()
    test_instance.test_full_workflow_all_circuits()