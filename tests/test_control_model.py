import networkx as nx
import pytest

from amps_simulation.core.control_block import ControlPort, ControlSource, InPort, LinearControlBlock, OutPort
from amps_simulation.core.control_model import ControlModel
from amps_simulation.core.control_signal import ControlSignal


def _build_model_with_two_sources() -> ControlModel:
    """
    Build a small ControlModel with two sources (V5=5.0, I1=1.0).

    Intentionally uses only the ControlModel API (no parser, no JSON files).
    """
    model = ControlModel()

    for comp_id, value in [("V5", 5.0), ("I1", 1.0)]:
        signal_source = ControlSource(name=f"{comp_id}Source", outport_names=[f"{comp_id}Out"])
        port_name = f"{comp_id}Port"
        port_block = ControlPort(
            name=port_name,
            port_type="source",
            inport_names=[f"{port_name}In"],
            outport_names=[f"{port_name}Out"],
        )
        model.add_block([signal_source, port_block])
        model.connect(signal_source, 0, port_block, 0, signal=ControlSignal(f"{comp_id}_signal", value))

    return model


def test_control_model_initialize_populates_block_lists():
    model = ControlModel()
    in_port = InPort(name="in1")
    out_port = OutPort(name="out1")
    linear = LinearControlBlock(name="G", inport_names=["u"], outport_names=["y"])
    generic = ControlPort(name="Vsrc", port_type="source", outport_names=["VsrcOut"])

    model.add_block([in_port, out_port, linear, generic])
    model.initialize()

    assert [b.name for b in model.list_all_blocks] == ["in1", "out1", "G", "Vsrc"]
    assert model.list_linear_blocks == [linear]
    assert model.list_inports == [in_port]
    assert model.list_outports == [out_port]


def test_control_model_construction_defaults():
    model = ControlModel()
    assert isinstance(model.graph, nx.MultiDiGraph)
    assert len(model.graph.nodes) == 0
    assert len(model.graph.edges) == 0
    assert model.initialized is False


def test_control_model_add_block_and_get_block_errors():
    model = ControlModel()
    port = ControlPort(name="P", port_type="source", outport_names=["POut"])
    model.add_block(port)

    assert model.get_block("P") is port
    assert model.block_list == [port]

    with pytest.raises(ValueError, match="already exists"):
        model.add_block(ControlPort(name="P", port_type="source", outport_names=["POut"]))

    with pytest.raises(KeyError, match="not found"):
        model.get_block("missing")

    model.graph.add_node("junk", block="not a block")
    with pytest.raises(TypeError, match="does not contain a ControlBlock"):
        model.get_block("junk")


def test_control_model_connect_creates_edge_and_signal_metadata():
    model = ControlModel()
    src = ControlSource(name="V1Source", outport_names=["V1Out"])
    dst = ControlPort(
        name="V1Port",
        port_type="source",
        inport_names=["V1PortIn"],
        outport_names=["V1PortOut"],
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
    model.add_block(ControlPort(name="P", port_type="source", outport_names=["POut"]))
    model.initialize()

    assert [b.name for b in model.list_all_blocks] == ["P"]


def test_control_model_initialize_lists_for_source_ports():
    model = _build_model_with_two_sources()
    model.initialize()

    names = {b.name for b in model.list_all_blocks}
    assert names == {"I1Source", "I1Port", "V5Source", "V5Port"}
    assert {b.name for b in model.list_linear_blocks} == set()
    assert {b.name for b in model.list_inports} == set()
    assert {b.name for b in model.list_outports} == set()

    assert set(model.port_blocks(port_type="source").keys()) == {"I1Port", "V5Port"}


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
        port_type="source",
        inport_names=["PIn"],
        outport_names=["POut"],
    )
    s1 = ControlSource(name="S1", outport_names=["S1Out"])
    s2 = ControlSource(name="S2", outport_names=["S2Out"])
    model.add_block([dst, s1, s2])

    model.connect(s1, 0, dst, 0, signal=ControlSignal("sig1", 1.0))
    model.connect(s2, 0, dst, 0, signal=ControlSignal("sig2", 2.0))

    with pytest.raises(ValueError, match="multiple driving signals"):
        model.compile_input_function(["P"])


def test_control_model_get_input_vector_requires_compile():
    model = ControlModel()
    with pytest.raises(RuntimeError, match="compile_input_function"):
        model.get_input_vector(0.0)
