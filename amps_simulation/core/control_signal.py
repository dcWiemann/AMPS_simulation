from __future__ import annotations

from typing import Any, Callable, List, Optional, Union
import sympy as sp


class ControlSignal:
    """
    Represents a control signal connecting two control blocks.

    The optional expression field preserves the legacy u(t) behavior for sources.
    """

    def __init__(
        self,
        name: str,
        expression: Optional[Union[str, float, int, Callable]] = None,
        *,
        src_block_name: Optional[str] = None,
        dst_block_name: Optional[str] = None,
        src_port_name: Optional[str] = None,
        dst_port_name: Optional[str] = None,
        src_port_idx: Optional[int] = None,
        dst_port_idx: Optional[int] = None,
        dtype: Any = None,
        shape: Optional[List[int]] = None,
    ):
        # Keep legacy attribute name for compatibility with existing code
        self.name = name
        self.signal_id = name
        self.expression = expression

        self.src_block_name = src_block_name
        self.dst_block_name = dst_block_name
        self.src_port_name = src_port_name
        self.dst_port_name = dst_port_name
        self.src_port_idx = src_port_idx
        self.dst_port_idx = dst_port_idx
        self.dtype = dtype
        self.shape = list(shape) if shape is not None else None

        self._compiled_func = self._compile_expression(expression) if expression is not None else None

    def _compile_expression(self, expr) -> Callable[[float], float]:
        """Compile expression for fast evaluation."""
        if callable(expr):
            return expr
        if isinstance(expr, (int, float)):
            return lambda t: float(expr)  # Constant signal
        if isinstance(expr, str):
            try:
                t = sp.Symbol('t')
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
        raise ValueError(f"Unsupported expression type: {type(expr)}")

    def evaluate(self, t: float) -> float:
        """Fast evaluation at time t."""
        if self._compiled_func is None:
            raise RuntimeError(f"No expression defined for signal '{self.name}'")
        return float(self._compiled_func(t))

    def __repr__(self) -> str:
        return f"ControlSignal(id='{self.signal_id}', expr='{self.expression}')"
