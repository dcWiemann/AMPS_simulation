from amps_simulation.core.control_block import ControlPort, InPort, OutPort, LinearControlBlock
from amps_simulation.core.control_model import ControlModel

def test_control_port_evaluate_pass_through():
    port = ControlPort(name="portA", variable=5, port_type="source", inport_names=["u"], outport_names=["y"])
    assert port.name == "portA"
    assert port.variable == 5
    assert port.port_type == "source"
    assert port.evaluate(t=0.0, u=[123.0]) == 123.0
    assert port.evaluate(t=0.0, u=[]) is None

def test_control_port_subclasses_inport_outport_metadata():
    in_port = InPort(name="in1")
    out_port = OutPort(name="out1")

    assert in_port.port_type == "input"
    assert in_port.inport_names == ["in1__in"]
    assert in_port.outport_names == []

    assert out_port.port_type == "output"
    assert out_port.inport_names == []
    assert out_port.outport_names == ["out1__out"]


def test_control_model_port_blocks_filters_by_type():
    model = ControlModel()
    model.add_block(ControlPort(name="Vsrc", port_type="source", outport_names=["Vsrc__out"]))
    model.add_block(ControlPort(name="Sw", port_type="switch", outport_names=["Sw__out"]))
    model.add_block(LinearControlBlock(name="G", inport_names=["u"], outport_names=["y"]))

    assert set(model.port_blocks().keys()) == {"Vsrc", "Sw"}
    assert set(model.port_blocks(port_type="source").keys()) == {"Vsrc"}
    assert set(model.port_blocks(port_type="switch").keys()) == {"Sw"}
