#!/usr/bin/env python3
"""
Debug script to check what output values are actually generated.
"""

import json
import numpy as np
from amps_simulation.core.parser import ParserJson
from amps_simulation.core.engine import Engine
from amps_simulation.core.components import Component

# Clear registry
Component.clear_registry()

# Load and run full_var0 to see actual outputs
with open("test_data/full_var0.json", 'r') as f:
    circuit_data = json.load(f)

parser = ParserJson()
graph, control_graph = parser.parse(circuit_data)

engine = Engine(graph, control_graph)
engine.initialize()

t_span = (0.0, 5.0)
initial_conditions = np.zeros(len(engine.state_vars))

result = engine.run_simulation(
    t_span=t_span,
    initial_conditions=initial_conditions,
    method='RK45',
    max_step=0.01
)

print("Engine state variables:", engine.state_vars)
print("Engine output variables:", engine.output_vars)
print("Final state values:", result['y'][:, -1])

if 'outputs' in result:
    print("Final output values:", result['outputs'][:, -1])
    print("Output shape:", result['outputs'].shape)
    
    for i, output_var in enumerate(engine.output_vars):
        final_val = result['outputs'][i, -1]
        print(f"  {output_var}: {final_val:.6f}")
else:
    print("No 'outputs' key in result!")

if 'out' in result:
    print("'out' key found!")
    print("Out shape:", result['out'].shape)
    print("Final out values:", result['out'][:, -1])
    
    for i, output_var in enumerate(engine.output_vars):
        final_val = result['out'][i, -1]
        print(f"  {output_var}: {final_val:.6f}")
    
print("Result keys:", list(result.keys()))