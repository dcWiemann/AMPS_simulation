## Architecture Overview (planned)

This section captures the planned structure that will move into the README once stabilized.

### Core concepts
- **Model (composite)**: Container with its own `name`, an `ElectricalModel`, a `ControlModel`, optional `subsystems` (nested Models), and external ports. Enforces unique, clean names (no double underscores) among components, ports, and subsystems.
- **ElectricalModel**: Pure topology and components; no simulation variables. Adding a component uses `add_component(component, name, nodes=[...])`. Enforces uniqueness of component names within the model.
- **ControlModel**: Control blocks and control ports. Enforces unique block/port names within the model.
- **ControlPorts & ElectricalPorts**: Interfaces to the outer world. ControlPorts inherit from ControlBlocks. ElectricalPorts expose external terminals.
- **VarInfo**: Immutable mapping of `path` (tuple of names from root to element) to `variables` (e.g., voltage/current/signal symbols) plus optional `value`. Path naming uses `model__subsystem__component` segments to guarantee global uniqueness.

### Engine responsibilities
- Consumes the top-level Model, flattens nested Models into single ElectricalModel and ControlModel while preserving full paths for traceability.
- Owns stable ordering of nodes/edges/ports; builds a topology snapshot (incidence matrix, VarInfos, index maps) once and reuses it across switch/diode modes.
- Creates symbols from paths, attaches VarInfo to each node/edge, and invokes components with provided symbols and control/state values (`cp_value`).
- Caches mode-specific results (e.g., A/B/C/D or equation sets) keyed by switch/diode states; the topology snapshot is reused.

### Component API (simulation-facing)
- `get_comp_eq(voltage_var, current_var, cp_value=None)`
- `is_short_circuit_state(cp_value=None)`
- `is_open_circuit_state(cp_value=None)`
- Class attributes: `n_control_port`, `n_states` (`n_states=1` for capacitor/inductor, else `0`; switches have `n_control_port=1`).
- Components are simulation-stateless; all runtime info is provided via arguments.

### Control/Electrical interface
- Components needing control use `cp_value` supplied by Engine from ControlModel outputs.
- Meters and sources expose ControlPorts (InPorts for meters, OutPorts for sources/switches) to wire control/electrical domains.

### Naming and paths
- Variable names derive from the full path: `model__subsystem__component__varname` (e.g., `mdl__ss1__R1__v`).
- Path validation: non-empty segments, no double underscores.

### Testing and evolution
- Tests will be updated to the new APIs (no backward compatibility). A separate markdown report will summarize test changes.
