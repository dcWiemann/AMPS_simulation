from amps_simulation.core.control_block import ControlPort, InPort, OutPort, LinearControlBlock
from amps_simulation.core.control_model import ControlModel

def test_control_port_evaluate_pass_through():
    port = ControlPort(name="portA", input_names=["u"], output_names=["y"])
    assert port.name == "portA"
    assert port.evaluate(t=0.0, u=[123.0]) == 123.0
    assert port.evaluate(t=0.0, u=[]) is None

def test_control_port_subclasses_inport_outport_metadata():
    in_port = InPort(name="in1")
    out_port = OutPort(name="out1")

    assert in_port.input_names == []
    assert in_port.output_names == ["out"]

    assert out_port.input_names == ["in"]
    assert out_port.output_names == []


def test_control_model_port_blocks_filters_by_type():
    model = ControlModel()
    model.add_block(InPort(name="In1"))
    model.add_block(OutPort(name="Out1"))
    model.add_block(ControlPort(name="Generic", input_names=["u"], output_names=["y"]))
    model.add_block(LinearControlBlock(name="G", input_names=["u"], output_names=["y"]))

    assert set(model.port_blocks().keys()) == {"In1", "Out1", "Generic"}
    assert set(model.port_blocks(port_type="input").keys()) == {"In1"}
    assert set(model.port_blocks(port_type="output").keys()) == {"Out1"}
