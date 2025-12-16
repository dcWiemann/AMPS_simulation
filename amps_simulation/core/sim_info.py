from dataclasses import dataclass
from typing import Any, Sequence, Tuple


@dataclass
class SimInfo:
    """
    Container for runtime simulation metadata.

    Attributes:
        name: Identifier of the node/edge the info belongs to.
        path: Hierarchical path for the node/edge (e.g., ("component", "SW1")).
        value: Runtime value stored for simulation (e.g., switch state).
    """
    name: str
    path: Tuple[str, ...]
    value: Any = None

    def __post_init__(self) -> None:
        # Ensure path is always a tuple for consistency
        if isinstance(self.path, str):
            self.path = (self.path,)
        elif isinstance(self.path, Sequence):
            self.path = tuple(self.path)
        else:
            self.path = (str(self.path),)
