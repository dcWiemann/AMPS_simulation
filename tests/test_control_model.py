import networkx as nx
import pytest

import control
import numpy as np

from amps_simulation.core.control_block import ControlPort, ControlSource, Gain, InPort, LinearControlBlock, OutPort, StateSpaceModel, Sum
from amps_simulation.core.control_model import ControlModel
from amps_simulation.core.control_signal import ControlSignal


def _build_model_with_two_sources() -> ControlModel:
    """
    Build a small ControlModel with two sources (V5=5.0, I1=1.0).

    Intentionally uses only the ControlModel API (no parser, no JSON files).
    """
    model = ControlModel()

    for comp_id, value in [("V5", 5.0), ("I1", 1.0)]:
        signal_source = ControlSource(name=f"{comp_id}Source", output_names=[f"{comp_id}Out"])
        port_name = f"{comp_id}Port"
        port_block = ControlPort(
            name=port_name,
            input_names=[f"{port_name}In"],
            output_names=[f"{port_name}Out"],
        )
        model.add_block([signal_source, port_block])
        model.connect(signal_source, 0, port_block, 0, signal=ControlSignal(f"{comp_id}_signal", value))

    return model


def test_control_model_initialize_populates_block_lists():
    model = ControlModel()
    in_port = InPort(name="in1")
    out_port = OutPort(name="out1")
    linear = LinearControlBlock(name="G", input_names=["u"], output_names=["y"])
    generic = ControlPort(name="Vsrc", output_names=["VsrcOut"])

    model.add_block([in_port, out_port, linear, generic])
    model.initialize()

    assert [b.name for b in model.list_all_blocks] == ["in1", "out1", "G", "Vsrc"]
    assert model.list_linear_blocks == [linear]
    assert model.list_input_ports == [in_port]
    assert model.list_output_ports == [out_port]


def test_control_model_construction_defaults():
    model = ControlModel()
    assert isinstance(model.graph, nx.MultiDiGraph)
    assert len(model.graph.nodes) == 0
    assert len(model.graph.edges) == 0
    assert model.initialized is False


def test_control_model_add_block_and_get_block_errors():
    model = ControlModel()
    port = ControlPort(name="P", output_names=["POut"])
    model.add_block(port)

    assert model.get_block("P") is port
    assert model.block_list == [port]

    with pytest.raises(ValueError, match="already exists"):
        model.add_block(ControlPort(name="P", output_names=["POut"]))

    with pytest.raises(KeyError, match="not found"):
        model.get_block("missing")

    model.graph.add_node("junk", block="not a block")
    with pytest.raises(TypeError, match="does not contain a ControlBlock"):
        model.get_block("junk")


def test_control_model_connect_creates_edge_and_signal_metadata():
    model = ControlModel()
    src = ControlSource(name="V1Source", output_names=["V1Out"])
    dst = ControlPort(
        name="V1Port",
        input_names=["V1PortIn"],
        output_names=["V1PortOut"],
    )
    model.add_block([src, dst])

    signal = model.connect(src, 0, dst, 0, signal=ControlSignal("V1Signal", 1.0))
    assert signal.name == "V1Signal"
    assert signal.src_block_name == "V1Source"
    assert signal.dst_block_name == "V1Port"
    assert signal.src_port_idx == 0
    assert signal.dst_port_idx == 0

    edge_data = model.graph.get_edge_data("V1Source", "V1Port")
    assert edge_data is not None
    assert signal.name in edge_data
    assert edge_data[signal.name]["signal"] is signal


def test_control_model_initialize_ignores_nodes_without_control_blocks():
    model = ControlModel()
    model.graph.add_node("junk", block="not a block")
    model.add_block(ControlPort(name="P", output_names=["POut"]))
    model.initialize()

    assert [b.name for b in model.list_all_blocks] == ["P"]


def test_control_model_initialize_lists_for_source_ports():
    model = _build_model_with_two_sources()
    model.initialize()

    names = {b.name for b in model.list_all_blocks}
    assert names == {"I1Source", "I1Port", "V5Source", "V5Port"}
    assert {b.name for b in model.list_linear_blocks} == set()
    assert {b.name for b in model.list_input_ports} == set()
    assert {b.name for b in model.list_output_ports} == set()

    assert set(model.port_blocks().keys()) == {"I1Port", "V5Port"}


def test_control_model_compile_input_function_builds_u_vector_from_signals():
    model = _build_model_with_two_sources()

    model.compile_input_function(["V5Port", "I1Port"])
    assert model.get_port_order() == ["V5Port", "I1Port"]

    u0 = model.get_input_vector(0.0)
    u1 = model.get_input_vector(123.0)

    assert u0.shape == (2,)
    assert u0.tolist() == [5.0, 1.0]
    assert u1.tolist() == [5.0, 1.0]


def test_control_model_compile_input_function_rejects_multiple_drivers():
    model = ControlModel()
    dst = ControlPort(
        name="P",
        input_names=["PIn"],
        output_names=["POut"],
    )
    s1 = ControlSource(name="S1", output_names=["S1Out"])
    s2 = ControlSource(name="S2", output_names=["S2Out"])
    model.add_block([dst, s1, s2])

    model.connect(s1, 0, dst, 0, signal=ControlSignal("sig1", 1.0))
    model.connect(s2, 0, dst, 0, signal=ControlSignal("sig2", 2.0))

    with pytest.raises(ValueError, match="multiple driving signals"):
        model.compile_input_function(["P"])


def test_control_model_get_input_vector_requires_compile():
    model = ControlModel()
    with pytest.raises(RuntimeError, match="compile_input_function"):
        model.get_input_vector(0.0)


def test_control_model_api_feedback_system_with_pi_and_plant_state_space():
    model = ControlModel()

    ref = InPort(name="Ref")
    error_sum = Sum(name="ErrorSum", signs=[1, -1], n_inputs=2)

    kp = 2.0
    ki = 3.0
    pi = StateSpaceModel(
        name="PI",
        A=np.array([[0.0]]),
        B=np.array([[1.0]]),
        C=np.array([[ki]]),
        D=np.array([[kp]]),
    )

    a = 4.0
    b = 1.5
    c = 1.0
    plant = StateSpaceModel(
        name="Plant",
        A=np.array([[-a]]),
        B=np.array([[b]]),
        C=np.array([[c]]),
        D=np.array([[0.0]]),
    )

    kfb = Gain(name="Kfb", gain=0.7)
    y = OutPort(name="Y")

    model.add_block([ref, error_sum, pi, plant, kfb, y])

    model.connect(ref, 0, error_sum, 0, signal=ControlSignal("RefToSum", 0.0))
    model.connect(plant, 0, kfb, 0, signal=ControlSignal("PlantToKfb", 0.0))
    model.connect(kfb, 0, error_sum, 1, signal=ControlSignal("KfbToSum", 0.0))
    model.connect(error_sum, 0, pi, 0, signal=ControlSignal("ErrorToPI", 0.0))
    model.connect(pi, 0, plant, 0, signal=ControlSignal("PIToPlant", 0.0))
    model.connect(plant, 0, y, 0, signal=ControlSignal("PlantToY", 0.0))

    model.initialize()

    assert model.list_input_ports == [ref]
    assert model.list_output_ports == [y]
    assert set(b.name for b in model.list_linear_blocks) == {"ErrorSum", "PI", "Plant", "Kfb"}

    assert error_sum.signs == [1, -1]
    assert error_sum.D.tolist() == [[1.0, -1.0]]

    assert isinstance(pi.state_space, control.StateSpace)
    assert isinstance(plant.state_space, control.StateSpace)
    assert isinstance(kfb.state_space, control.StateSpace)

    cycles = list(nx.simple_cycles(nx.DiGraph(model.graph)))
    assert any({"ErrorSum", "PI", "Plant", "Kfb"}.issubset(set(cycle)) for cycle in cycles)
