from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import numpy as np


@dataclass
class ControlBlock:
    """
    Base class for control blocks.

    This keeps only structural metadata for now; numerical behavior will be
    fleshed out later.
    """
    name: str
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    input_dtype: Any = None
    output_dtype: Any = None
    n_inputs: int = field(init=False, default=0)
    n_outputs: int = field(init=False, default=0)
    n_states: int = field(init=False, default=0)

    def __init__(
        self,
        name: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        *,
        input_dtype: Any = None,
        output_dtype: Any = None,
    ) -> None:
        self.name = name
        self.input_names = list(input_names) if input_names is not None else ["in"]
        self.output_names = list(output_names) if output_names is not None else ["out"]
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.__post_init__()

    def __post_init__(self) -> None:
        self.n_inputs = len(self.input_names)
        self.n_outputs = len(self.output_names)
        self.n_states = 0

    def evaluate(self, t: float, u: Sequence[Any], x: Optional[Sequence[float]] = None) -> Any:
        """Compute block output at time t for inputs u (and optional state x)."""
        raise NotImplementedError(f"evaluate() not implemented for {self.__class__.__name__}")


class ControlPort(ControlBlock):
    """Simple block representing a single control port."""

    def __init__(
        self,
        name: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ):
        super().__init__(name=name, input_names=input_names or [], output_names=output_names or [])
        self.input_dtype = Any
        self.output_dtype = Any

    def evaluate(self, t: float, u: Sequence[Any], x: Optional[Sequence[float]] = None) -> Any:
        return u[0] if u else None


class InPort(ControlPort):
    """
    Input block for the control layer.
    Used by Meters to report measured values back to the control system.
    """

    def __init__(self, name: str):
        super().__init__(name=name, input_names=[], output_names=["out"])
        self.output_dtype = Any

class OutPort(ControlPort):
    """
    Output block for the control layer.
    Used by Sources and Switches to receive control signals from the control system.
    """

    def __init__(self, name: str):
        super().__init__(name=name, input_names=["in"], output_names=[])
        self.input_dtype = Any


class LinearControlBlock(ControlBlock):
    """
    Base class for linear control blocks with state-space matrices.
    A, B, C, D definitions are placeholders for now.
    """

    def __init__(
        self,
        name: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ):
        super().__init__(name=name, input_names=input_names or [], output_names=output_names or [])
        self.input_dtype = float
        self.output_dtype = float

        # All linear blocks have A, B, C, D.
        # Stateless blocks use 0 state dimension, i.e. A is (0,0).
        self.A: np.ndarray = np.zeros((0, 0), dtype=float)
        self.B: np.ndarray = np.zeros((0, 0), dtype=float)
        self.C: np.ndarray = np.zeros((0, 0), dtype=float)
        self.D: np.ndarray = np.zeros((0, 0), dtype=float)

        # Optional state naming (used when nx > 0)
        self.x_names: List[str] = []

    @staticmethod
    def _as_2d_float_array(value: Any, *, name: str) -> np.ndarray:
        if value is None:
            raise ValueError(f"Matrix '{name}' must not be None for LinearControlBlock")
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape((1, 1))
        elif arr.ndim == 1:
            arr = arr.reshape((arr.shape[0], 1))
        elif arr.ndim != 2:
            raise ValueError(f"Matrix '{name}' must be 0D/1D/2D, got shape {arr.shape}")
        return arr

    def _validate_and_finalize_from_matrices(self) -> None:
        """
        Validate state-space dimensions and derive n_states/n_inputs/n_outputs.

        Expected shapes:
          A: (nx, nx)
          B: (nx, nu)
          C: (ny, nx)
          D: (ny, nu)
        """
        if self.A.ndim != 2 or self.B.ndim != 2 or self.C.ndim != 2 or self.D.ndim != 2:
            raise ValueError(f"Block '{self.name}' A/B/C/D must all be 2D arrays")

        nx = int(self.A.shape[0])
        if self.A.shape != (nx, nx):
            raise ValueError(f"Block '{self.name}' invalid A shape {self.A.shape}, expected ({nx},{nx})")
        if self.B.shape[0] != nx:
            raise ValueError(f"Block '{self.name}' B rows {self.B.shape[0]} must match n_states={nx}")
        if self.C.shape[1] != nx:
            raise ValueError(f"Block '{self.name}' C cols {self.C.shape[1]} must match n_states={nx}")

        ny = int(self.C.shape[0])
        nu = int(self.B.shape[1])
        if self.D.shape != (ny, nu):
            raise ValueError(f"Block '{self.name}' invalid D shape {self.D.shape}, expected ({ny},{nu})")

        if self.input_names:
            if len(self.input_names) != nu:
                raise ValueError(
                    f"Block '{self.name}' input_names length {len(self.input_names)} must match n_inputs={nu}"
                )
        else:
            self.input_names = ["u"] if nu == 1 else [f"u{i}" for i in range(1, nu + 1)]

        if self.output_names:
            if len(self.output_names) != ny:
                raise ValueError(
                    f"Block '{self.name}' output_names length {len(self.output_names)} must match n_outputs={ny}"
                )
        else:
            self.output_names = ["y"] if ny == 1 else [f"y{i}" for i in range(1, ny + 1)]

        self.n_states = nx
        self.n_inputs = nu
        self.n_outputs = ny

        if nx > 0 and not self.x_names:
            self.x_names = [f"x{i}" for i in range(1, nx + 1)]

    def evaluate(self, t: float, u: Sequence[Any], x: Optional[Sequence[float]] = None) -> Any:
        """
        For stateless linear blocks: y = D u
        For stateful linear blocks: y = C x + D u
        """
        if len(u) != self.n_inputs:
            raise ValueError(f"Block '{self.name}' expected {self.n_inputs} inputs, got {len(u)}")
        u_vec = np.asarray(u, dtype=float).reshape((-1, 1)) if self.n_inputs else np.zeros((0, 1), dtype=float)
        if self.n_states > 0:
            if x is None:
                raise ValueError(f"Block '{self.name}' requires state vector x of length {self.n_states}")
            if len(x) != self.n_states:
                raise ValueError(f"Block '{self.name}' expected state length {self.n_states}, got {len(x)}")
            x_vec = np.asarray(x, dtype=float).reshape((-1, 1))
            y = self.C @ x_vec + self.D @ u_vec
        else:
            y = self.D @ u_vec

        if y.shape == (1, 1) and self.n_outputs == 1:
            return float(y[0, 0])
        return y.reshape((-1,))


class Gain(LinearControlBlock):
    """Static gain block."""

    def __init__(self, name: str, gain: float = 1.0):
        super().__init__(name=name, input_names=["u"], output_names=["y"])
        self.gain = gain
        self.A = np.zeros((0, 0), dtype=float)
        self.B = np.zeros((0, 1), dtype=float)
        self.C = np.zeros((1, 0), dtype=float)
        self.D = np.asarray([[float(gain)]], dtype=float)
        self._validate_and_finalize_from_matrices()

    def evaluate(self, t: float, u: Sequence[Any], x: Optional[Sequence[float]] = None) -> Any:
        if not u:
            return 0.0
        return float(self.D[0, 0]) * float(u[0])


class Sum(LinearControlBlock):
    """Summing junction with sign pattern."""

    def __init__(self, name: str, signs: Optional[List[int]] = None, n_inputs: Optional[int] = None):
        signs = list(signs) if signs is not None else []
        if n_inputs is None:
            n_inputs = len(signs)
        if n_inputs <= 1:
            raise ValueError("n_inputs must be > 1")
        if signs and len(signs) != n_inputs:
            raise ValueError(f"Expected {n_inputs} signs, got {len(signs)}")

        super().__init__(name=name, input_names=[f"u{i}" for i in range(1, n_inputs + 1)], output_names=["y"])
        self.signs = signs if signs else [1 for _ in range(n_inputs)]

        self.A = np.zeros((0, 0), dtype=float)
        self.B = np.zeros((0, n_inputs), dtype=float)
        self.C = np.zeros((1, 0), dtype=float)
        self.D = np.asarray([self.signs], dtype=float)
        self._validate_and_finalize_from_matrices()

    def evaluate(self, t: float, u: Sequence[Any], x: Optional[Sequence[float]] = None) -> Any:
        if not u:
            return 0.0
        if len(u) != len(self.signs):
            raise ValueError(f"Block '{self.name}' expected {len(self.signs)} inputs, got {len(u)}")
        return float(sum(sign * float(val) for sign, val in zip(self.signs, u)))


class StateSpaceModel(LinearControlBlock):
    """Full state-space block."""

    def __init__(self, name: str, A=None, B=None, C=None, D=None, x_init: Optional[Any] = None,
                 x_names: Optional[List[str]] = None):
        super().__init__(name=name, input_names=[], output_names=[])
        self.A = self._as_2d_float_array(A, name="A")
        self.B = self._as_2d_float_array(B, name="B")
        self.C = self._as_2d_float_array(C, name="C")
        self.D = self._as_2d_float_array(D, name="D")
        self.x_init = x_init
        self.x_names = list(x_names) if x_names is not None else []
        self._validate_and_finalize_from_matrices()


class Integrator(LinearControlBlock):
    """Integrator block."""

    def __init__(self, name: str, x_init: float = 0.0, x_name: Optional[str] = None):
        super().__init__(name=name, input_names=["u"], output_names=["y"])
        self.x_init = x_init
        self.x_name = x_name
        self.A = np.asarray([[0.0]], dtype=float)
        self.B = np.asarray([[1.0]], dtype=float)
        self.C = np.asarray([[1.0]], dtype=float)
        self.D = np.asarray([[0.0]], dtype=float)
        self.x_names = [x_name] if x_name else ["x1"]
        self._validate_and_finalize_from_matrices()

    def evaluate(self, t: float, u: Sequence[Any], x: Optional[Sequence[float]] = None) -> Any:
        if x is None or len(x) != self.n_states:
            raise ValueError(f"Block '{self.name}' requires state vector x of length {self.n_states}")
        return float(x[0])


class TransferFunction(LinearControlBlock):
    """Transfer function block storing numerator/denominator."""

    def __init__(self, name: str, num=None, den=None):
        super().__init__(name=name, input_names=["u"], output_names=["y"])
        self.num = num
        self.den = den
        # Placeholder until we compute a realization.
        self.A = np.zeros((0, 0), dtype=float)
        self.B = np.zeros((0, 1), dtype=float)
        self.C = np.zeros((1, 0), dtype=float)
        self.D = np.zeros((1, 1), dtype=float)
        self._validate_and_finalize_from_matrices()

    def evaluate(self, t: float, u: Sequence[Any], x: Optional[Sequence[float]] = None) -> Any:
        raise NotImplementedError("TransferFunction.evaluate() requires a state-space realization (to be added)")


class ControlSource(ControlBlock):
    """Base class for source blocks u(t)."""

    def __init__(self, name: str, output_names: Optional[List[str]] = None):
        super().__init__(name=name, input_names=[], output_names=output_names or ["y"])
        self.input_dtype = []
        self.output_dtype = Any
        self.n_inputs = 0
        self.n_states = 0

    def evaluate(self, t: float, u: Sequence[Any], x: Optional[Sequence[float]] = None) -> Any:
        # Nonlinear/time functions will be implemented later.
        return None


class Step(ControlSource):
    """Step source block."""

    def __init__(self, name: str, t0: float = 0.0, initial_value: float = 0.0, final_value: float = 1.0):
        super().__init__(name=name, output_names=["y"])
        self.t0 = t0
        self.initial_value = initial_value
        self.final_value = final_value
        self.output_dtype = Any

    def evaluate(self, t: float, u: Sequence[Any], x: Optional[Sequence[float]] = None) -> Any:
        return self.initial_value if t < self.t0 else self.final_value


class Constant(ControlSource):
    """Constant output source."""

    def __init__(self, name: str, value: Any):
        super().__init__(name=name, output_names=["y"])
        self.value = value
        self.output_dtype = Any

    def evaluate(self, t: float, u: Sequence[Any], x: Optional[Sequence[float]] = None) -> Any:
        return self.value
