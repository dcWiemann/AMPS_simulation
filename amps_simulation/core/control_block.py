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
    inport_names: List[str] = field(default_factory=list)
    outport_names: List[str] = field(default_factory=list)
    inport_dtype: Any = None
    outport_dtype: Any = None
    inport_shape: List[int] = field(default_factory=list)
    outport_shape: List[int] = field(default_factory=list)
    n_inports: int = field(init=False, default=0)
    n_outports: int = field(init=False, default=0)
    n_states: int = field(init=False, default=0)

    def __init__(
        self,
        name: str,
        inport_names: Optional[List[str]] = None,
        outport_names: Optional[List[str]] = None,
        *,
        inport_dtype: Any = None,
        outport_dtype: Any = None,
        inport_shape: Optional[List[int]] = None,
        outport_shape: Optional[List[int]] = None,
    ) -> None:
        self.name = name
        self.inport_names = list(inport_names) if inport_names is not None else []
        self.outport_names = list(outport_names) if outport_names is not None else []
        self.inport_dtype = inport_dtype
        self.outport_dtype = outport_dtype
        self.inport_shape = list(inport_shape) if inport_shape is not None else []
        self.outport_shape = list(outport_shape) if outport_shape is not None else []
        self.__post_init__()

    def __post_init__(self) -> None:
        self.n_inports = len(self.inport_names)
        self.n_outports = len(self.outport_names)
        self.n_states = 0

    def evaluate(self, t: float, u: Sequence[Any], x: Optional[Sequence[float]] = None) -> Any:
        """Compute block output at time t for inputs u (and optional state x)."""
        raise NotImplementedError(f"evaluate() not implemented for {self.__class__.__name__}")


class ControlPort(ControlBlock):
    """Simple block representing a single control port."""

    def __init__(
        self,
        name: str,
        port_type: str = "generic",
        inport_names: Optional[List[str]] = None,
        outport_names: Optional[List[str]] = None,
    ):
        super().__init__(name=name, inport_names=inport_names or [], outport_names=outport_names or [])
        self.inport_dtype = Any
        self.outport_dtype = Any
        self.inport_shape = [1] if self.n_inports else []
        self.outport_shape = [1] if self.n_outports else []
        self.port_type = port_type

    def evaluate(self, t: float, u: Sequence[Any], x: Optional[Sequence[float]] = None) -> Any:
        return u[0] if u else None

class InPort(ControlPort):
    """
    Input block for the control layer.
    Used by Meters to report measured values back to the control system.
    """

    def __init__(self, name: str):
        super().__init__(name=name, port_type="input", inport_names=[f"{name}__in"], outport_names=[])
        self.inport_dtype = Any
        self.inport_shape = [1]
        self.outport_shape = []

class OutPort(ControlPort):
    """
    Output block for the control layer.
    Used by Sources and Switches to receive control signals from the control system.
    """

    def __init__(self, name: str):
        super().__init__(name=name, port_type="output", inport_names=[], outport_names=[f"{name}__out"])
        self.inport_dtype = []
        self.outport_dtype = Any
        self.inport_shape = []
        self.outport_shape = [1]


class LinearControlBlock(ControlBlock):
    """
    Base class for linear control blocks with state-space matrices.
    A, B, C, D definitions are placeholders for now.
    """

    def __init__(
        self,
        name: str,
        inport_names: Optional[List[str]] = None,
        outport_names: Optional[List[str]] = None,
    ):
        super().__init__(name=name, inport_names=inport_names or [], outport_names=outport_names or [])
        self.inport_dtype = float
        self.outport_dtype = float

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
        Validate state-space dimensions and derive n_states/n_inports/n_outports and port shapes.

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

        if self.inport_names:
            if len(self.inport_names) != nu:
                raise ValueError(
                    f"Block '{self.name}' inport_names length {len(self.inport_names)} must match n_inports={nu}"
                )
        else:
            self.inport_names = ["u"] if nu == 1 else [f"u{i}" for i in range(1, nu + 1)]

        if self.outport_names:
            if len(self.outport_names) != ny:
                raise ValueError(
                    f"Block '{self.name}' outport_names length {len(self.outport_names)} must match n_outports={ny}"
                )
        else:
            self.outport_names = ["y"] if ny == 1 else [f"y{i}" for i in range(1, ny + 1)]

        self.n_states = nx
        self.n_inports = nu
        self.n_outports = ny

        # Shapes are the per-port signal shapes (scalar/vector/matrix).
        # For now, each port carries a scalar by default.
        self.inport_shape = [1] if nu > 0 else []
        self.outport_shape = [1] if ny > 0 else []

        if nx > 0 and not self.x_names:
            self.x_names = [f"x{i}" for i in range(1, nx + 1)]

    def evaluate(self, t: float, u: Sequence[Any], x: Optional[Sequence[float]] = None) -> Any:
        """
        For stateless linear blocks: y = D u
        For stateful linear blocks: y = C x + D u
        """
        if len(u) != self.n_inports:
            raise ValueError(f"Block '{self.name}' expected {self.n_inports} inputs, got {len(u)}")
        u_vec = np.asarray(u, dtype=float).reshape((-1, 1)) if self.n_inports else np.zeros((0, 1), dtype=float)
        if self.n_states > 0:
            if x is None:
                raise ValueError(f"Block '{self.name}' requires state vector x of length {self.n_states}")
            if len(x) != self.n_states:
                raise ValueError(f"Block '{self.name}' expected state length {self.n_states}, got {len(x)}")
            x_vec = np.asarray(x, dtype=float).reshape((-1, 1))
            y = self.C @ x_vec + self.D @ u_vec
        else:
            y = self.D @ u_vec

        if y.shape == (1, 1) and self.outport_shape == [1]:
            return float(y[0, 0])
        return y.reshape((-1,))


class Gain(LinearControlBlock):
    """Static gain block."""

    def __init__(self, name: str, gain: float = 1.0):
        super().__init__(name=name, inport_names=["u"], outport_names=["y"])
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

    def __init__(self, name: str, signs: Optional[List[int]] = None, n_inports: Optional[int] = None):
        signs = list(signs) if signs is not None else []
        if n_inports is None:
            n_inports = len(signs)
        if n_inports <= 0:
            raise ValueError("n_inports must be > 0")
        if signs and len(signs) != n_inports:
            raise ValueError(f"Expected {n_inports} signs, got {len(signs)}")

        super().__init__(name=name, inport_names=[f"u{i}" for i in range(1, n_inports + 1)], outport_names=["y"])
        self.signs = signs if signs else [1 for _ in range(n_inports)]

        self.A = np.zeros((0, 0), dtype=float)
        self.B = np.zeros((0, n_inports), dtype=float)
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
        super().__init__(name=name, inport_names=[], outport_names=[])
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
        super().__init__(name=name, inport_names=["u"], outport_names=["y"])
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
        super().__init__(name=name, inport_names=["u"], outport_names=["y"])
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

    @staticmethod
    def _infer_shape(value: Any) -> List[int]:
        if isinstance(value, (int, float, bool)):
            return [1]
        if isinstance(value, (list, tuple)):
            return [len(value)]
        if hasattr(value, "shape"):
            try:
                return list(value.shape)
            except Exception:
                return []
        return []

    def __init__(self, name: str, outport_names: Optional[List[str]] = None):
        super().__init__(name=name, inport_names=[], outport_names=outport_names or ["y"])
        self.inport_dtype = []
        self.outport_dtype = Any
        self.inport_shape = []
        self.outport_shape = [1]
        self.n_inports = 0
        self.n_states = 0

    def evaluate(self, t: float, u: Sequence[Any], x: Optional[Sequence[float]] = None) -> Any:
        # Nonlinear/time functions will be implemented later.
        return None


class Step(ControlSource):
    """Step source block."""

    def __init__(self, name: str, t0: float = 0.0, initial_value: float = 0.0, final_value: float = 1.0):
        super().__init__(name=name, outport_names=["y"])
        self.t0 = t0
        self.initial_value = initial_value
        self.final_value = final_value
        self.outport_dtype = Any
        init_shape = self._infer_shape(initial_value)
        final_shape = self._infer_shape(final_value)
        self.outport_shape = init_shape if init_shape == final_shape else (init_shape or final_shape or [])

    def evaluate(self, t: float, u: Sequence[Any], x: Optional[Sequence[float]] = None) -> Any:
        return self.initial_value if t < self.t0 else self.final_value


class Constant(ControlSource):
    """Constant output source."""

    def __init__(self, name: str, value: Any):
        super().__init__(name=name, outport_names=["y"])
        self.value = value
        self.outport_dtype = Any
        self.outport_shape = self._infer_shape(value)

    def evaluate(self, t: float, u: Sequence[Any], x: Optional[Sequence[float]] = None) -> Any:
        return self.value
