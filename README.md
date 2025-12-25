# Analysis and Simulation of Power Systems

**amps_simulation** is a lightweight and extensible circuit simulation tool written in Python.

It represents a circuit as a graph, builds a (mode-dependent) DAE/state-space model, and simulates it with SciPy's `solve_ivp`.

Supported components:

- **Sources**
  - `VoltageSource`
  - `CurrentSource`
- **Passive Components**
  - `Resistor`
  - `Capacitor`
  - `Inductor`
- **Active Components**
  - `PowerSwitch`
  - `Diode`
- **Meters**
  - `Voltmeter`
  - `Ammeter`

## Installation

### From source

```bash
python3 -m venv .venv
source .venv/bin/activate

# editable install
python -m pip install -e .  # or: python -m pip install -e ".[test]"

# runtime/test dependencies not currently declared in packaging metadata
python -m pip install "pydantic>=2" networkx
```

### Run tests

```bash
pytest
```

## Basic API usage

### Load a circuit from JSON

The test circuits in `test_data/` are valid inputs for the JSON parser:

```python
import json

from amps_simulation.core.engine import Engine
from amps_simulation.core.electrical_model import ElectricalModel
from amps_simulation.core.control_model import ControlModel
from amps_simulation.core.parser import ParserJson

with open("test_data/test_rc.json") as f:
    circuit_json = json.load(f)

graph, control_graph = ParserJson().parse(circuit_json)

electrical_model = ElectricalModel(graph)
control_model = ControlModel(control_graph)

engine = Engine(electrical_model, control_model)
engine.initialize()

result = engine.run_simulation(t_span=(0.0, 0.1), method="RK45", max_step=1e-3)
```

There is also a convenience wrapper that parses + runs + (optionally) plots:

```python
from amps_simulation.run_simulation import run_simulation

result = run_simulation(circuit_json, t_span=(0.0, 0.1), plot_results=False, max_step=1e-3)
```

### Build a circuit programmatically (RC low-pass + voltmeter)

This example builds a simple RC low-pass driven by a voltage source, and measures the capacitor voltage with an ideal voltmeter placed across the capacitor.

```python
import numpy as np

from amps_simulation.core.components import (
    Component,
    ElecJunction,
    VoltageSource,
    Resistor,
    Capacitor,
    Voltmeter,
)
from amps_simulation.core.electrical_model import ElectricalModel
from amps_simulation.core.engine import Engine

# If you're building multiple circuits in one Python process, clear registries
Component.clear_registry()
ElecJunction.clear_registry()

model = ElectricalModel()
model.add_node(0, is_ground=True)  # ground reference
model.add_node(1)  # vin
model.add_node(2)  # vout

V1 = VoltageSource("V1", 5.0)
R1 = Resistor("R1", 1_000.0)
C1 = Capacitor("C1", 1e-3)
VM1 = Voltmeter(comp_id="VM1")

# RC low-pass: vin -- R1 -- vout -- C1 -- gnd
model.add_component(V1, p=1, n=0)
model.add_component(R1, p=1, n=2)
model.add_component(C1, p=2, n=0)

# Voltmeter across capacitor (p=vout, n=gnd => positive reading)
model.add_component(VM1, p=2, n=0)

engine = Engine(model)
engine.initialize()

# Provide the source waveform u(t). The vector order matches `engine.input_vars`.
engine.control_input_function = lambda t: np.array([5.0])

result = engine.run_simulation(t_span=(0.0, 0.25), max_step=1e-3)

t = result["t"]
v_c = result["out"][engine.output_vars.index(VM1.output_var)]
```

### Results and variable ordering

- `engine.state_vars`: ordered symbolic state variables (capacitor voltages `v_C*(t)`, inductor currents `i_L*(t)`).
- `engine.input_vars`: ordered symbolic inputs (source voltages/currents).
- `engine.output_vars`: ordered symbolic outputs (meters).
- `result["t"]`: time vector.
- `result["y"]`: state trajectories with shape `(len(engine.state_vars), len(t))`.
- `result["out"]`: output trajectories with shape `(len(engine.output_vars), len(t))` (or `None` if no meters).

Solver settings are passed through to SciPy via `Engine.run_simulation(..., **kwargs)` (e.g. `max_step`, `rtol`, `atol`, `method`).

## License

MIT License. See `LICENSE` for details.
