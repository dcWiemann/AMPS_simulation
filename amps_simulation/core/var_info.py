from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class VarInfo:
    """
    Immutable mapping between a hierarchical path and the variable names/symbols
    that represent it in a simulation context.
    """
    path: Tuple[str, ...]
    variables: Dict[str, Any] = field(default_factory=dict)
    value: Any = None

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("VarInfo.path cannot be empty")
        for segment in self.path:
            if not segment:
                raise ValueError("VarInfo.path entries must be non-empty strings")
            if "__" in segment:
                raise ValueError("VarInfo.path entries may not contain double underscores")
