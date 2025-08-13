import pytest
import numpy as np
from amps_simulation.core.control_orchestrator import ControlGraph, ControlOrchestrator, ControlSignal
from amps_simulation.core.control_port import ControlPort

class TestControlGraph:
    def test_create_empty_control_graph(self):
        cg = ControlGraph()
        assert len(cg.signals) == 0
        assert len(cg.ports) == 0
        assert len(cg.connections) == 0

    def test_add_signal(self):
        cg = ControlGraph()
        signal = ControlSignal("test_signal", 5.0)
        cg.add_signal(signal)
        
        assert "test_signal" in cg.signals
        assert cg.signals["test_signal"] is signal

    def test_add_port(self):
        cg = ControlGraph()
        port = ControlPort(name="test_port", variable="V1", port_type="source")
        cg.add_port(port)
        
        assert "test_port" in cg.ports
        assert cg.ports["test_port"] is port

    def test_connect_signal_to_port(self):
        cg = ControlGraph()
        signal = ControlSignal("sig1", 3.3)
        port = ControlPort(name="port1", variable="V1", port_type="source")
        
        cg.add_signal(signal)
        cg.add_port(port)
        cg.connect_signal_to_port("sig1", "port1")
        
        assert "port1" in cg.connections
        assert cg.connections["port1"] == ("sig1", 1.0)  # Updated to match tuple format

    def test_connect_nonexistent_signal_raises_error(self):
        cg = ControlGraph()
        port = ControlPort(name="port1", variable="V1", port_type="source")
        cg.add_port(port)
        
        with pytest.raises(ValueError, match="Signal 'nonexistent' not found"):
            cg.connect_signal_to_port("nonexistent", "port1")

    def test_connect_nonexistent_port_raises_error(self):
        cg = ControlGraph()
        signal = ControlSignal("sig1", 3.3)
        cg.add_signal(signal)
        
        with pytest.raises(ValueError, match="Port 'nonexistent' not found"):
            cg.connect_signal_to_port("sig1", "nonexistent")

    def test_get_source_ports_only(self):
        cg = ControlGraph()
        source_port = ControlPort(name="source_port", variable="V1", port_type="source")
        switch_port = ControlPort(name="switch_port", variable="S1", port_type="switch")
        other_port = ControlPort(name="other_port", variable="X1", port_type="other")
        
        cg.add_port(source_port)
        cg.add_port(switch_port)
        cg.add_port(other_port)
        
        source_ports = cg.get_source_ports()
        assert len(source_ports) == 1
        assert "source_port" in source_ports
        assert source_ports["source_port"] is source_port


class TestControlOrchestrator:
    def test_create_orchestrator(self):
        cg = ControlGraph()
        orchestrator = ControlOrchestrator(cg)
        assert orchestrator.control_graph is cg

    def test_compile_input_function_empty_ports(self):
        cg = ControlGraph()
        orchestrator = ControlOrchestrator(cg)
        
        input_func = orchestrator.compile_input_function([])
        result = input_func(1.0)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_compile_input_function_with_ports(self):
        cg = ControlGraph()
        
        # Create signals
        signal1 = ControlSignal("sig1", 5.0)
        signal2 = ControlSignal("sig2", lambda t: 2.0 * t)
        
        # Create ports
        port1 = ControlPort(name="port1", variable="V1", port_type="source")
        port2 = ControlPort(name="port2", variable="V2", port_type="source")
        
        # Add to graph
        cg.add_signal(signal1)
        cg.add_signal(signal2)
        cg.add_port(port1)
        cg.add_port(port2)
        
        # Connect signals to ports
        cg.connect_signal_to_port("sig1", "port1")
        cg.connect_signal_to_port("sig2", "port2")
        
        # Create orchestrator and compile
        orchestrator = ControlOrchestrator(cg)
        input_func = orchestrator.compile_input_function(["port1", "port2"])
        
        # Test at t=0
        result = input_func(0.0)
        assert len(result) == 2
        assert result[0] == 5.0
        assert result[1] == 0.0
        
        # Test at t=3
        result = input_func(3.0)
        assert len(result) == 2
        assert result[0] == 5.0
        assert result[1] == 6.0

    def test_compile_input_function_unconnected_port_defaults_zero(self):
        cg = ControlGraph()
        
        # Create port but no signal connection
        port1 = ControlPort(name="port1", variable="V1", port_type="source")
        cg.add_port(port1)
        
        # Create orchestrator and compile
        orchestrator = ControlOrchestrator(cg)
        input_func = orchestrator.compile_input_function(["port1"])
        
        # Should return zero for unconnected port
        result = input_func(2.0)
        assert len(result) == 1
        assert result[0] == 0.0

    def test_compile_input_function_nonexistent_port_raises_error(self):
        cg = ControlGraph()
        orchestrator = ControlOrchestrator(cg)
        
        with pytest.raises(ValueError, match="Ports not found in control graph"):
            orchestrator.compile_input_function(["nonexistent"])

    def test_compile_input_function_maintains_port_order(self):
        cg = ControlGraph()
        
        # Create signals with different values
        signal1 = ControlSignal("sig1", 10.0)
        signal2 = ControlSignal("sig2", 20.0)
        signal3 = ControlSignal("sig3", 30.0)
        
        # Create ports
        port1 = ControlPort(name="port1", variable="V1", port_type="source")
        port2 = ControlPort(name="port2", variable="V2", port_type="source")
        port3 = ControlPort(name="port3", variable="V3", port_type="source")
        
        # Add to graph
        cg.add_signal(signal1)
        cg.add_signal(signal2)
        cg.add_signal(signal3)
        cg.add_port(port1)
        cg.add_port(port2)
        cg.add_port(port3)
        
        # Connect in different order than ports
        cg.connect_signal_to_port("sig2", "port2")
        cg.connect_signal_to_port("sig1", "port1")
        cg.connect_signal_to_port("sig3", "port3")
        
        # Create orchestrator and compile with specific order
        orchestrator = ControlOrchestrator(cg)
        input_func = orchestrator.compile_input_function(["port3", "port1", "port2"])
        
        # Should return values in the order of ports specified
        result = input_func(0.0)
        assert len(result) == 3
        assert result[0] == 30.0  # port3 -> sig3
        assert result[1] == 10.0  # port1 -> sig1  
        assert result[2] == 20.0  # port2 -> sig2