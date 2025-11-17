from typing import Union, Callable
import sympy as sp


class ControlSignal:
    """Represents a time-dependent control signal u_i(t)"""

    def __init__(self, signal_id: str, expression: Union[str, float, int, Callable]):
        self.signal_id = signal_id
        self.expression = expression
        self._compiled_func = self._compile_expression(expression)

    def _compile_expression(self, expr) -> Callable[[float], float]:
        """Compile expression for fast evaluation"""
        if callable(expr):
            return expr
        elif isinstance(expr, (int, float)):
            return lambda t: float(expr)  # Constant signal
        elif isinstance(expr, str):
            # Parse string like "sin(t)", "5*sin(2*pi*t)", "sawtooth(t)"
            try:
                t = sp.Symbol('t')
                # Add common functions to namespace for parsing
                namespace = {
                    'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
                    'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
                    'pi': sp.pi, 'e': sp.E,
                    't': t
                }
                sympy_expr = sp.sympify(expr, locals=namespace)
                return sp.lambdify(t, sympy_expr, 'numpy')
            except Exception as e:
                raise ValueError(f"Cannot parse expression '{expr}': {e}")
        else:
            raise ValueError(f"Unsupported expression type: {type(expr)}")

    def evaluate(self, t: float) -> float:
        """Fast evaluation at time t"""
        return float(self._compiled_func(t))

    def __repr__(self):
        return f"ControlSignal(id='{self.signal_id}', expr='{self.expression}')"
