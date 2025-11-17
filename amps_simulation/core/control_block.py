class ControlBlock():
 # in, out functions 

class ControlPort(ControlBlock):
 # in, out signals

class LinearControlBlock(ControlBlock):
 # matrix A, B, C, D 
 
class Gain(LinearControlBlock):
 D = gain 

class Sum(LinearControlBlock):
 D = I #adjust the signs according to '+-' 
 
class StateSpace(LinearControlBlock):
 A, B, C, D 

class ControlSource(ControlBlock):
 out = f(t) # e.g. PWM, sin() # for now: no input for nonlinear blocks class PWM(ControlSource) # duty cycle D, frequency 

class Constant(ControlSource):
 out = const