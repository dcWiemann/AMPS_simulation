#!/usr/bin/env python3
"""
Debug script for running complete AMPS simulation workflow from JSON to plots.

This script provides comprehensive debugging and visualization capabilities:
- Loads circuit from JSON file
- Runs complete simulation pipeline with error handling
- Generates detailed plots of all simulation results
- Performs circuit topology sanity checks
- Exports simulation data and metadata
- Provides performance timing information
"""

import sys
import json
import time
import logging
import argparse
import traceback
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from amps_simulation.core.parser import ParserJson
from amps_simulation.core.engine import Engine
from amps_simulation.core.electrical_model import ElectricalModel
from amps_simulation.core.components import Component
from amps_simulation.core.circuit_sanity_checker import CircuitSanityChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug/simulation_debug.log')
    ]
)

class SimulationDebugger:
    """Comprehensive debugging tool for AMPS circuit simulation workflow."""
    
    def __init__(self, json_file: str, output_dir: str = "debug/results"):
        """
        Initialize the simulation debugger.
        
        Args:
            json_file: Path to circuit JSON file
            output_dir: Directory for output files and plots
        """
        self.json_file = Path(json_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.circuit_name = self.json_file.stem
        self.timing_data = {}
        self.simulation_data = {}
        
        logging.info(f"Initialized debugger for circuit: {self.circuit_name}")
    
    def load_circuit(self) -> Dict[str, Any]:
        """Load circuit from JSON file with error handling."""
        start_time = time.time()
        
        try:
            with open(self.json_file, 'r') as f:
                circuit_data = json.load(f)
            
            self.timing_data['load_circuit'] = time.time() - start_time
            logging.info(f"Circuit loaded successfully from {self.json_file}")
            logging.info(f"Circuit contains {len(circuit_data.get('nodes', []))} nodes and {len(circuit_data.get('edges', []))} edges")
            
            return circuit_data
            
        except Exception as e:
            logging.error(f"Failed to load circuit from {self.json_file}: {e}")
            raise
    
    def run_sanity_checks(self, graph) -> Dict[str, Any]:
        """Run circuit topology sanity checks and return results."""
        start_time = time.time()
        logging.info("Running circuit topology sanity checks...")
        
        try:
            checker = CircuitSanityChecker(graph)
            sanity_results = checker.check_all(raise_on_error=False)
            
            self.timing_data['sanity_checks'] = time.time() - start_time
            
            # Log results
            if sanity_results['errors']:
                logging.warning(f"Found {len(sanity_results['errors'])} topology errors:")
                for error in sanity_results['errors']:
                    logging.warning(f"  ERROR: {error}")
            
            if sanity_results['warnings']:
                logging.info(f"Found {len(sanity_results['warnings'])} topology warnings:")
                for warning in sanity_results['warnings']:
                    logging.info(f"  WARNING: {warning}")
            
            if not sanity_results['errors'] and not sanity_results['warnings']:
                logging.info("All topology sanity checks passed!")
                
            # Get constraint requirements
            constraints = checker.get_constraint_modifications()
            if any(constraints.values()):
                logging.info("Circuit constraint requirements:")
                for constraint_type, components in constraints.items():
                    if components:
                        logging.info(f"  {constraint_type}: {components}")
            
            return sanity_results
            
        except Exception as e:
            logging.error(f"Sanity checks failed: {e}")
            return {'errors': [f"Sanity check failure: {e}"], 'warnings': []}
    
    def parse_circuit(self, circuit_data: Dict[str, Any]) -> tuple:
        """Parse circuit data into graph structures."""
        start_time = time.time()
        logging.info("Parsing circuit data...")
        
        try:
            # Clear component registry
            Component.clear_registry()
            
            parser = ParserJson()
            graph, control_graph = parser.parse(circuit_data)
            
            self.timing_data['parse_circuit'] = time.time() - start_time
            
            # Log graph information
            logging.info(f"Created electrical graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            if control_graph.signals:
                logging.info(f"Control graph has {len(control_graph.signals)} signals and {len(control_graph.ports)} ports")
            
            return graph, control_graph
            
        except Exception as e:
            logging.error(f"Circuit parsing failed: {e}")
            raise
    
    def initialize_engine(self, graph, control_graph) -> Engine:
        """Initialize simulation engine with error handling."""
        start_time = time.time()
        logging.info("Initializing simulation engine...")

        try:
            electrical_model = ElectricalModel(graph)
            engine = Engine(electrical_model, control_graph)
            engine.initialize()
            
            self.timing_data['initialize_engine'] = time.time() - start_time
            
            # Log engine information
            logging.info(f"Engine initialized successfully")
            logging.info(f"State variables ({len(engine.state_vars)}): {list(engine.state_vars)}")
            logging.info(f"Input variables ({len(engine.input_vars)}): {list(engine.input_vars)}")
            logging.info(f"Output variables ({len(engine.output_vars)}): {list(engine.output_vars)}")
            
            if engine.switch_list:
                logging.info(f"Switch components ({len(engine.switch_list)}): {[s.comp_id for s in engine.switch_list]}")
            
            return engine
            
        except Exception as e:
            logging.error(f"Engine initialization failed: {e}")
            raise
    
    def run_simulation(self, engine: Engine, 
                      t_span: tuple = (0.0, 5.0), 
                      initial_conditions: Optional[np.ndarray] = None,
                      method: str = 'RK45',
                      **kwargs) -> Dict[str, Any]:
        """Run circuit simulation with comprehensive error handling."""
        start_time = time.time()
        logging.info(f"Starting simulation for time span {t_span} using method {method}")
        
        try:
            # Set default initial conditions
            if initial_conditions is None:
                initial_conditions = np.zeros(len(engine.state_vars))
                logging.info(f"Using zero initial conditions: {initial_conditions}")
            else:
                logging.info(f"Using provided initial conditions: {initial_conditions}")
            
            # Default simulation parameters
            sim_params = {
                'max_step': 0.01,
                'rtol': 1e-8,
                'atol': 1e-10
            }
            sim_params.update(kwargs)
            
            # Run simulation
            result = engine.run_simulation(
                t_span=t_span,
                initial_conditions=initial_conditions,
                method=method,
                **sim_params
            )
            
            self.timing_data['run_simulation'] = time.time() - start_time
            
            # Log simulation results
            if result['success']:
                logging.info(f"Simulation completed successfully in {self.timing_data['run_simulation']:.3f}s")
                logging.info(f"Integration steps: {len(result['t'])}")
                logging.info(f"Function evaluations: {result.get('nfev', 'N/A')}")
                logging.info(f"Final time: {result['t'][-1]:.6f}s")
                
                if len(engine.state_vars) > 0:
                    final_states = result['y'][:, -1]
                    logging.info(f"Final state values: {dict(zip(engine.state_vars, final_states))}")
                
                if result.get('out') is not None and len(engine.output_vars) > 0:
                    final_outputs = result['out'][:, -1]
                    logging.info(f"Final output values: {dict(zip(engine.output_vars, final_outputs))}")
                
                if result.get('switchmap_size', 0) > 0:
                    logging.info(f"Switch combinations cached: {result['switchmap_size']}")
                
            else:
                logging.error(f"Simulation failed: {result['message']}")
            
            return result
            
        except Exception as e:
            logging.error(f"Simulation execution failed: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def create_comprehensive_plots(self, engine: Engine, result: Dict[str, Any]) -> List[str]:
        """Create comprehensive plots of simulation results."""
        start_time = time.time()
        logging.info("Generating comprehensive plots...")
        
        plot_files = []
        
        try:
            # Set up plotting style
            plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
            
            # 1. Main results plot
            plot_files.append(self._create_main_results_plot(engine, result))
            
            # 2. State variables detailed plot
            if len(engine.state_vars) > 0:
                plot_files.append(self._create_state_variables_plot(engine, result))
            
            # 3. Output variables detailed plot
            if result.get('out') is not None and len(engine.output_vars) > 0:
                plot_files.append(self._create_output_variables_plot(engine, result))
            
            # 4. Phase plots if multiple state variables
            if len(engine.state_vars) >= 2:
                phase_plot = self._create_phase_plots(engine, result)
                if phase_plot:
                    plot_files.append(phase_plot)
            
            # 5. Performance and timing plot
            plot_files.append(self._create_performance_plot(result))
            
            self.timing_data['create_plots'] = time.time() - start_time
            logging.info(f"Generated {len(plot_files)} plots in {self.timing_data['create_plots']:.3f}s")
            
            return plot_files
            
        except Exception as e:
            logging.error(f"Plot generation failed: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return plot_files
    
    def _create_main_results_plot(self, engine: Engine, result: Dict[str, Any]) -> str:
        """Create main results overview plot."""
        fig_size = (16, 12)
        n_plots = len(engine.state_vars) + (len(engine.output_vars) if result.get('out') is not None else 0)
        n_cols = min(3, max(2, n_plots))
        n_rows = max(2, (n_plots + n_cols - 1) // n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
        # Always ensure axes is a 1D array of Axes objects
        if n_rows == 1 and n_cols == 1:
            axes = [axes]  # Single subplot case
        else:
            axes = axes.flatten()  # Multi-subplot case
        
        fig.suptitle(f'AMPS Simulation Results: {self.circuit_name}', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # Plot state variables
        for i, state_var in enumerate(engine.state_vars):
            if plot_idx < len(axes):
                ax = axes[plot_idx]
                ax.plot(result['t'], result['y'][i, :], color=colors[i % 10], linewidth=2, label=str(state_var))
                ax.set_title(f'State: {state_var}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Time (s)')
                
                # Determine unit based on variable name
                if 'i_' in str(state_var).lower():
                    ax.set_ylabel('Current (A)')
                elif 'v_' in str(state_var).lower():
                    ax.set_ylabel('Voltage (V)')
                else:
                    ax.set_ylabel('Value')
                
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Add switch event markers if applicable
                if engine.switch_list:
                    for switch in engine.switch_list:
                        if hasattr(switch, 'switch_time') and switch.switch_time > 0:
                            ax.axvline(x=switch.switch_time, color='red', linestyle='--', 
                                     alpha=0.7, label=f'Switch {switch.comp_id}')
                
                plot_idx += 1
        
        # Plot output variables
        if result.get('out') is not None:
            for i, output_var in enumerate(engine.output_vars):
                if plot_idx < len(axes):
                    ax = axes[plot_idx]
                    ax.plot(result['t'], result['out'][i, :], color=colors[(i + len(engine.state_vars)) % 10], 
                           linewidth=2, label=str(output_var))
                    ax.set_title(f'Output: {output_var}', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Time (s)')
                    
                    # Determine unit based on variable name
                    if 'AM' in str(output_var) or 'i_' in str(output_var).lower():
                        ax.set_ylabel('Current (A)')
                    elif 'VM' in str(output_var) or 'v_' in str(output_var).lower():
                        ax.set_ylabel('Voltage (V)')
                    else:
                        ax.set_ylabel('Value')
                    
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    # Add switch event markers
                    if engine.switch_list:
                        for switch in engine.switch_list:
                            if hasattr(switch, 'switch_time') and switch.switch_time > 0:
                                ax.axvline(x=switch.switch_time, color='red', linestyle='--', 
                                         alpha=0.7, label=f'Switch {switch.comp_id}')
                    
                    plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{self.circuit_name}_main_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(plot_path)
    
    def _create_state_variables_plot(self, engine: Engine, result: Dict[str, Any]) -> str:
        """Create detailed state variables plot."""
        fig, axes = plt.subplots(len(engine.state_vars), 1, figsize=(12, 4*len(engine.state_vars)))
        if len(engine.state_vars) == 1:
            axes = [axes]
        
        fig.suptitle(f'State Variables: {self.circuit_name}', fontsize=16, fontweight='bold')
        
        for i, (state_var, ax) in enumerate(zip(engine.state_vars, axes)):
            ax.plot(result['t'], result['y'][i, :], linewidth=2, color=f'C{i}')
            ax.set_title(f'{state_var}', fontsize=12)
            ax.set_xlabel('Time (s)')
            
            # Add statistics text box
            data = result['y'][i, :]
            stats_text = f'Min: {np.min(data):.4f}\nMax: {np.max(data):.4f}\nFinal: {data[-1]:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Determine unit
            if 'i_' in str(state_var).lower():
                ax.set_ylabel('Current (A)')
            elif 'v_' in str(state_var).lower():
                ax.set_ylabel('Voltage (V)')
            else:
                ax.set_ylabel('Value')
                
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{self.circuit_name}_state_variables.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(plot_path)
    
    def _create_output_variables_plot(self, engine: Engine, result: Dict[str, Any]) -> str:
        """Create detailed output variables plot."""
        output_data = result['out']
        
        fig, axes = plt.subplots(len(engine.output_vars), 1, figsize=(12, 4*len(engine.output_vars)))
        if len(engine.output_vars) == 1:
            axes = [axes]
        
        fig.suptitle(f'Output Variables: {self.circuit_name}', fontsize=16, fontweight='bold')
        
        for i, (output_var, ax) in enumerate(zip(engine.output_vars, axes)):
            ax.plot(result['t'], output_data[i, :], linewidth=2, color=f'C{i}')
            ax.set_title(f'{output_var}', fontsize=12)
            ax.set_xlabel('Time (s)')
            
            # Add statistics text box
            data = output_data[i, :]
            stats_text = f'Min: {np.min(data):.4f}\nMax: {np.max(data):.4f}\nFinal: {data[-1]:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Determine unit
            if 'AM' in str(output_var) or 'i_' in str(output_var).lower():
                ax.set_ylabel('Current (A)')
            elif 'VM' in str(output_var) or 'v_' in str(output_var).lower():
                ax.set_ylabel('Voltage (V)')
            else:
                ax.set_ylabel('Value')
                
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{self.circuit_name}_output_variables.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(plot_path)
    
    def _create_phase_plots(self, engine: Engine, result: Dict[str, Any]) -> str:
        """Create phase space plots for state variables."""
        n_states = len(engine.state_vars)
        n_phase_plots = min(3, n_states//2)
        if n_phase_plots == 0:
            return None  # Skip if no phase plots possible
        
        fig, axes = plt.subplots(1, n_phase_plots, figsize=(15, 5))
        if n_phase_plots == 1:
            axes = [axes]
        
        fig.suptitle(f'Phase Space Plots: {self.circuit_name}', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        for i in range(0, n_states-1, 2):
            if plot_idx < len(axes):
                ax = axes[plot_idx]
                ax.plot(result['y'][i, :], result['y'][i+1, :], linewidth=1.5, alpha=0.8)
                ax.set_xlabel(f'{engine.state_vars[i]}')
                ax.set_ylabel(f'{engine.state_vars[i+1]}')
                ax.set_title(f'Phase: {engine.state_vars[i]} vs {engine.state_vars[i+1]}')
                ax.grid(True, alpha=0.3)
                
                # Mark start and end points
                ax.plot(result['y'][i, 0], result['y'][i+1, 0], 'go', markersize=8, label='Start')
                ax.plot(result['y'][i, -1], result['y'][i+1, -1], 'ro', markersize=8, label='End')
                ax.legend()
                
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{self.circuit_name}_phase_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(plot_path)
    
    def _create_performance_plot(self, result: Dict[str, Any]) -> str:
        """Create performance and timing analysis plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        fig.suptitle(f'Performance Analysis: {self.circuit_name}', fontsize=16, fontweight='bold')
        
        # Timing breakdown
        ax = axes[0, 0]
        timing_labels = list(self.timing_data.keys())
        timing_values = list(self.timing_data.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(timing_labels)))
        
        bars = ax.bar(timing_labels, timing_values, color=colors)
        ax.set_title('Execution Time Breakdown')
        ax.set_ylabel('Time (s)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, timing_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{value:.3f}s', ha='center', va='bottom')
        
        # Integration statistics
        ax = axes[0, 1]
        stats = ['Steps', 'Function Evals', 'Success', 'Final Time']
        values = [
            len(result.get('t', [])),
            result.get('nfev', 0),
            1 if result.get('success', False) else 0,
            result.get('t', [0])[-1] if result.get('t') is not None else 0
        ]
        
        bars = ax.bar(stats, values, color=['skyblue', 'lightgreen', 'gold', 'salmon'])
        ax.set_title('Integration Statistics')
        ax.set_ylabel('Count / Value')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   f'{value}', ha='center', va='bottom')
        
        # Step size analysis
        if result.get('t') is not None and len(result['t']) > 1:
            ax = axes[1, 0]
            step_sizes = np.diff(result['t'])
            ax.plot(result['t'][1:], step_sizes, linewidth=1, alpha=0.7)
            ax.set_title('Integration Step Sizes')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Step Size (s)')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f'Mean: {np.mean(step_sizes):.2e}\nStd: {np.std(step_sizes):.2e}\nMin: {np.min(step_sizes):.2e}\nMax: {np.max(step_sizes):.2e}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Memory/efficiency metrics
        ax = axes[1, 1]
        efficiency_metrics = ['Total Time (s)', 'Steps/Second', 'Switchmap Size']
        total_time = sum(self.timing_data.values())
        steps_per_sec = len(result.get('t', [])) / max(total_time, 1e-6)
        switchmap_size = result.get('switchmap_size', 0)
        
        efficiency_values = [total_time, steps_per_sec, switchmap_size]
        bars = ax.bar(efficiency_metrics, efficiency_values, color=['mediumpurple', 'orange', 'teal'])
        ax.set_title('Efficiency Metrics')
        ax.set_ylabel('Value')
        
        for bar, value in zip(bars, efficiency_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiency_values)*0.01,
                   f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{self.circuit_name}_performance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(plot_path)
    
    def export_simulation_data(self, engine: Engine, result: Dict[str, Any], 
                              sanity_results: Dict[str, Any]) -> str:
        """Export comprehensive simulation data to files."""
        start_time = time.time()
        logging.info("Exporting simulation data...")
        
        # Prepare comprehensive data structure
        export_data = {
            'circuit_info': {
                'name': self.circuit_name,
                'source_file': str(self.json_file),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'engine_info': {
                'state_variables': [str(var) for var in engine.state_vars],
                'input_variables': [str(var) for var in engine.input_vars],
                'output_variables': [str(var) for var in engine.output_vars],
                'num_components': len(engine.components_list),
                'has_switches': bool(engine.switch_list),
                'switch_count': len(engine.switch_list) if engine.switch_list else 0
            },
            'simulation_results': {
                'success': result.get('success', False),
                'message': result.get('message', ''),
                'final_time': float(result['t'][-1]) if result.get('t') is not None else 0,
                'num_steps': len(result.get('t', [])),
                'function_evaluations': result.get('nfev', 0),
                'switchmap_size': result.get('switchmap_size', 0)
            },
            'sanity_check_results': sanity_results,
            'timing_data': self.timing_data,
            'performance_metrics': {
                'total_execution_time': sum(self.timing_data.values()),
                'steps_per_second': len(result.get('t', [])) / max(sum(self.timing_data.values()), 1e-6)
            }
        }
        
        # Add final values
        if result.get('success', False):
            if result.get('y') is not None and len(engine.state_vars) > 0:
                final_states = result['y'][:, -1].tolist()
                export_data['final_values'] = {
                    'states': dict(zip([str(var) for var in engine.state_vars], final_states))
                }
                
            if result.get('out') is not None and len(engine.output_vars) > 0:
                final_outputs = result['out'][:, -1].tolist()
                export_data['final_values']['outputs'] = dict(zip([str(var) for var in engine.output_vars], final_outputs))
        
        # Export JSON metadata
        json_path = self.output_dir / f'{self.circuit_name}_simulation_data.json'
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        # Export CSV data if simulation was successful
        if result.get('success', False) and result.get('t') is not None:
            csv_data = {'time': result['t']}
            
            # Add state variables
            for i, var in enumerate(engine.state_vars):
                csv_data[f'state_{var}'] = result['y'][i, :]
            
            # Add output variables
            if result.get('out') is not None:
                for i, var in enumerate(engine.output_vars):
                    csv_data[f'output_{var}'] = result['out'][i, :]
            
            df = pd.DataFrame(csv_data)
            csv_path = self.output_dir / f'{self.circuit_name}_time_series.csv'
            df.to_csv(csv_path, index=False)
            
            logging.info(f"Exported time series data to {csv_path}")
        
        self.timing_data['export_data'] = time.time() - start_time
        logging.info(f"Exported simulation data to {json_path}")
        
        return str(json_path)
    
    def run_complete_workflow(self, **simulation_kwargs) -> Dict[str, Any]:
        """Run the complete simulation workflow with comprehensive debugging."""
        workflow_start_time = time.time()
        
        logging.info(f"{'='*60}")
        logging.info(f"Starting complete AMPS simulation workflow")
        logging.info(f"Circuit: {self.circuit_name}")
        logging.info(f"Output directory: {self.output_dir}")
        logging.info(f"{'='*60}")
        
        try:
            # 1. Load circuit
            circuit_data = self.load_circuit()
            
            # 2. Parse circuit
            graph, control_graph = self.parse_circuit(circuit_data)
            
            # 3. Run sanity checks
            sanity_results = self.run_sanity_checks(graph)
            
            # Stop if critical errors found
            if sanity_results['errors']:
                logging.error("Critical topology errors found - stopping workflow")
                return {
                    'success': False,
                    'error': 'Critical topology errors',
                    'sanity_results': sanity_results
                }
            
            # 4. Initialize engine
            engine = self.initialize_engine(graph, control_graph)
            
            # 5. Run simulation
            result = self.run_simulation(engine, **simulation_kwargs)
            
            if not result.get('success', False):
                logging.error("Simulation failed - stopping workflow")
                return {
                    'success': False,
                    'error': 'Simulation failed',
                    'simulation_result': result
                }
            
            # 6. Generate plots
            plot_files = self.create_comprehensive_plots(engine, result)
            
            # 7. Export data
            data_file = self.export_simulation_data(engine, result, sanity_results)
            
            # Calculate total workflow time
            total_workflow_time = time.time() - workflow_start_time
            
            # Final summary
            logging.info(f"{'='*60}")
            logging.info(f"WORKFLOW COMPLETED SUCCESSFULLY")
            logging.info(f"Total execution time: {total_workflow_time:.3f}s")
            logging.info(f"Generated plots: {len(plot_files)}")
            logging.info(f"Output directory: {self.output_dir}")
            logging.info(f"{'='*60}")
            
            return {
                'success': True,
                'total_time': total_workflow_time,
                'timing_breakdown': self.timing_data,
                'plot_files': plot_files,
                'data_file': data_file,
                'simulation_result': result,
                'sanity_results': sanity_results,
                'engine': engine
            }
            
        except Exception as e:
            logging.error(f"Workflow failed with exception: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'partial_timing': self.timing_data
            }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Debug script for AMPS circuit simulation workflow',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('json_file', help='Path to circuit JSON file')
    parser.add_argument('--output-dir', default='debug/results', 
                       help='Output directory for plots and data')
    parser.add_argument('--t-end', type=float, default=5.0,
                       help='Simulation end time (seconds)')
    parser.add_argument('--method', default='RK45',
                       help='Integration method (RK45, DOP853, etc.)')
    parser.add_argument('--max-step', type=float, default=0.01,
                       help='Maximum integration step size')
    parser.add_argument('--rtol', type=float, default=1e-8,
                       help='Relative tolerance')
    parser.add_argument('--atol', type=float, default=1e-10,
                       help='Absolute tolerance')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if file exists
    if not Path(args.json_file).exists():
        logging.error(f"Circuit file not found: {args.json_file}")
        sys.exit(1)
    
    # Create debugger and run workflow
    debugger = SimulationDebugger(args.json_file, args.output_dir)
    
    simulation_params = {
        't_span': (0.0, args.t_end),
        'method': args.method,
        'max_step': args.max_step,
        'rtol': args.rtol,
        'atol': args.atol
    }
    
    workflow_result = debugger.run_complete_workflow(**simulation_params)
    
    if workflow_result['success']:
        print(f"\nWorkflow completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Generated {len(workflow_result['plot_files'])} plots")
        print(f"Total time: {workflow_result['total_time']:.3f}s")
    else:
        print(f"\nWorkflow failed: {workflow_result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()