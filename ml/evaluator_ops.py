"""Op-related code for ml evaluator"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Op(ABC):
    """Base operation with input/output type definitions.

    Each operation defines:
    - `name`: unique identifier for this operation
    - `input_types`: set of type names this operation requires as input
    - `output_type`: single type name this operation produces
    """
    name: str
    input_types: set[str]
    output_type: str

    @abstractmethod
    def execute(self, inputs: dict[str, Any]) -> Any:
        """Execute the operation with given inputs.

        - inputs: dict mapping type names to actual data

        Returns the output data of type `self.output_type`.
        """
        raise NotImplementedError()

    def get_cache_key(self, inputs: dict[str, Any]) -> str:
        """Generate cache key based on operation and inputs.

        This creates a hash of the operation name and input data to enable
        caching of results and avoiding duplicate work.
        """
        # Create a deterministic string representation of inputs
        input_items = []
        for key in sorted(inputs.keys()):
            value = inputs[key]
            # For complex objects, use their string representation
            if hasattr(value, '__dict__'):
                value_str = str(sorted(value.__dict__.items()))
            else:
                value_str = str(value)
            input_items.append(f"{key}:{value_str}")
        input_str = "|".join(input_items)
        combined = f"{self.name}_{input_str}"
        # Hash to keep cache keys manageable
        return hashlib.md5(combined.encode()).hexdigest()


@dataclass
class Result:
    """Stores results with full provenance tracking.

    Each result contains:
    - The operation that produced it
    - The actual data produced
    - Full provenance chain (list of operations that led to this result)
    - Cache key for deduplication
    - Timestamp and error information
    """
    op: Op
    data: Any
    provenance: list[Op]
    cache_key: str
    timestamp: float = field(default_factory=time.time)
    error: str = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        """Returns True if the operation completed successfully."""
        return self.error is None

    def get_provenance_string(self) -> str:
        """Human-readable provenance chain."""
        return " -> ".join(op.name for op in self.provenance)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/reporting.

        Note: Does not serialize the actual data as it may be large.
        """
        return {
            "op_name": self.op.name,
            "success": self.is_success(),
            "error": self.error,
            "provenance": self.get_provenance_string(),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "data_type": type(self.data).__name__ if self.data else None,
            "cache_key": self.cache_key,
        }


class OpRegistry:
    """Registry of all available operations.

    This maintains indexes by input and output types to enable automatic
    graph construction based on type matching.
    """
    def __init__(self):
        self.ops_by_output_type: dict[str, list[Op]] = defaultdict(list)
        self.ops_by_input_type: dict[str, list[Op]] = defaultdict(list)
        self.all_ops: list[Op] = []

    def register(self, op: Op) -> None:
        """Register an operation in the registry."""
        self.all_ops.append(op)
        self.ops_by_output_type[op.output_type].append(op)
        for input_type in op.input_types:
            self.ops_by_input_type[input_type].append(op)

    def get_producers(self, type_name: str) -> list[Op]:
        """Get all operations that produce a given type."""
        return self.ops_by_output_type[type_name]

    def get_consumers(self, type_name: str) -> list[Op]:
        """Get all operations that consume a given type."""
        return self.ops_by_input_type[type_name]

    def get_all_types(self) -> set[str]:
        """Get all known type names (both input and output)."""
        return set(self.ops_by_output_type.keys()) | set(self.ops_by_input_type.keys())
