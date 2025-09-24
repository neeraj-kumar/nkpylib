"""Op-related code for ml evaluator"""

from __future__ import annotations

import hashlib
import inspect
import time

from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from nkpylib.ml.feature_set import FeatureSet

# Global registry instance that all Ops will register with
_global_op_registry = None


@dataclass
class Result:
    """Stores results with full provenance tracking.

    Each result contains:
    - The actual data produced
    - Full provenance chain (list of operations that led to this result)
    - Cache key for deduplication
    - Timestamp and error information

    The operation that produced this result is always the last item in the provenance chain.
    """
    data: Any
    provenance: list[Op]
    cache_key: str
    timestamp: float = field(default_factory=time.time)
    error: str = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def op(self) -> Op:
        """The operation that produced this result (last in provenance chain)."""
        return self.provenance[-1] if self.provenance else None

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
            "op_name": self.op.name if self.op else None,
            "success": self.is_success(),
            "error": self.error,
            "provenance": self.get_provenance_string(),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "data_type": type(self.data).__name__ if self.data else None,
            "cache_key": self.cache_key,
        }


def find_subclasses(cls) -> list[type[Op]]:
    """Find all concrete (non-abstract) subclasses of `cls`."""
    ret = []
    for subclass in cls.__subclasses__():
        if not inspect.isabstract(subclass):
            ret.append(subclass)
        ret.extend(find_subclasses(subclass))  # Recursive for nested inheritance
    return ret


def generate_all_execution_paths(start_types: set[str] = None,
                               target_types: set[str] = None,
                               max_depth: int = 5) -> list[list[Op]]:
    """Generate all valid execution paths using graph search.

    - start_types: Types that are available at the beginning (empty set means no initial types)
    - target_types: Types we want to end up with (None means any final type is acceptable)
    - max_depth: Maximum number of operations in a path

    Returns list of operation sequences that form valid execution paths.
    """
    if start_types is None:
        start_types = set()

    # Get all concrete op classes and try to instantiate them
    op_classes = find_subclasses(Op)
    all_ops = []

    for op_cls in op_classes:
        try:
            # Try to instantiate with no args first
            op = op_cls()
            all_ops.append(op)
        except TypeError:
            # This op needs constructor arguments - skip for now
            # TODO: Handle parameterized ops by generating variants
            pass

    # Build type dependency mappings
    producers = defaultdict(list)  # type -> [ops that produce it]
    consumers = defaultdict(list)  # type -> [ops that consume it]

    for op in all_ops:
        producers[op.output_type].append(op)
        for input_type in op.input_types:
            consumers[input_type].append(op)

    # DFS to find all valid execution paths
    valid_paths = []

    def dfs(current_path: list[Op], available_types: set[str], depth: int):
        if depth > max_depth:
            return

        # Check if we've reached target types (if specified)
        if target_types and available_types & target_types:
            valid_paths.append(current_path.copy())
        elif not target_types and current_path:  # Any non-empty path is valid if no targets
            valid_paths.append(current_path.copy())

        # Try each op that can consume available types
        for type_name in available_types:
            for next_op in consumers[type_name]:
                # Skip if already in path (avoid cycles)
                if any(op.name == next_op.name for op in current_path):
                    continue

                # Check if all required inputs are available
                if not next_op.input_types.issubset(available_types):
                    continue

                # Add this op and recurse
                current_path.append(next_op)
                new_available = available_types | {next_op.output_type}
                dfs(current_path, new_available, depth + 1)
                current_path.pop()

    # Start DFS from ops that don't need inputs (or whose inputs are in start_types)
    for op in all_ops:
        if not op.input_types or op.input_types.issubset(start_types):
            initial_types = start_types | {op.output_type}
            dfs([op], initial_types, 1)

    return valid_paths


class OpRegistry:
    """Registry of all available operations.

    This maintains indexes by input and output types to enable automatic
    graph construction based on type matching.
    """
    def __init__(self):
        self.ops_by_output_type: dict[str, list[Op]] = defaultdict(list)
        self.ops_by_input_type: dict[str, list[Op]] = defaultdict(list)
        self.all_ops: list[Op] = []

    @staticmethod
    def register(op: Op) -> None:
        """Register an operation in the registry."""
        registry = OpRegistry.get_global_op_registry()
        registry.all_ops.append(op)
        registry.ops_by_output_type[op.output_type].append(op)
        for input_type in op.input_types:
            registry.ops_by_input_type[input_type].append(op)

    def get_producers(self, type_name: str) -> list[Op]:
        """Get all operations that produce a given type."""
        return self.ops_by_output_type[type_name]

    def get_consumers(self, type_name: str) -> list[Op]:
        """Get all operations that consume a given type."""
        return self.ops_by_input_type[type_name]

    def get_all_types(self) -> set[str]:
        """Get all known type names (both input and output)."""
        return set(self.ops_by_output_type.keys()) | set(self.ops_by_input_type.keys())

    @staticmethod
    def get_global_op_registry() -> OpRegistry:
        """Get the global registry, creating it if needed."""
        global _global_op_registry
        if _global_op_registry is None:
            _global_op_registry = OpRegistry()
        return _global_op_registry


@dataclass
class Op(ABC):
    """Base operation with input/output type definitions.

    Each operation defines:
    - `name`: unique identifier for this operation (class or instance variable)
    - `input_types`: set of type names this operation requires as input (class variable)
    - `output_type`: single type name this operation produces (class variable)

    When created, the operation automatically registers itself with the global registry.
    """
    name: str = ""

    # These should be overridden as class variables in subclasses if not dynamic
    input_types: frozenset[str] = frozenset()
    output_type: str = ""

    def __post_init__(self):
        """Automatically register this operation with the global registry."""
        # Use class name if instance name is not set
        if not self.name:
            self.name = self.__class__.__name__
        OpRegistry.register(self)

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


class LoadEmbeddingsOp(Op):
    """Load embeddings from paths into a FeatureSet."""

    name = 'load_embeddings'
    input_types = frozenset()
    output_type = "feature_set"

    def __init__(self, paths: list[str], **kwargs):
        self.paths = paths
        self.kwargs = kwargs

    def execute(self, inputs: dict[str, Any]) -> Any:
        return FeatureSet(self.paths, **self.kwargs)


class CheckDimensionsOp(Op):
    """Check that all embeddings have consistent dimensions."""

    name = "check_dimensions"
    input_types = frozenset({"feature_set"})
    output_type = "dimension_check_result"

    def execute(self, inputs: dict[str, Any]) -> Any:
        fs = inputs["feature_set"]
        dims = Counter()
        for key, emb in fs.items():
            dims[len(emb)] += 1
        is_consistent = len(dims) == 1
        return {
            "is_consistent": is_consistent,
            "dimension_counts": dict(dims),
            "error_message": None if is_consistent else f"Inconsistent embedding dimensions: {dims.most_common()}"
        }


class CheckNaNsOp(Op):
    """Check for NaN values in embeddings."""

    name = "check_nans"
    input_types = frozenset({"feature_set"})
    output_type = "nan_check_result"

    def execute(self, inputs: dict[str, Any]) -> Any:
        fs = inputs["feature_set"]
        n_nans = 0
        nan_keys = []
        for key, emb in fs.items():
            key_nans = np.sum(np.isnan(emb))
            n_nans += key_nans
            if key_nans > 0:
                nan_keys.append((key, int(key_nans)))
        has_nans = n_nans > 0
        return {
            "has_nans": has_nans,
            "total_nans": int(n_nans),
            "nan_keys": nan_keys,
            "error_message": None if not has_nans else f"Found {n_nans} NaNs in embeddings"
        }


class BasicChecksOp(Op):
    """Combine dimension and NaN checks into a single basic validation."""

    name = "basic_checks"
    input_types = frozenset({"dimension_check_result", "nan_check_result"})
    output_type = "basic_checks_report"

    def execute(self, inputs: dict[str, Any]) -> Any:
        dim_result = inputs["dimension_check_result"]
        nan_result = inputs["nan_check_result"]
        errors = []
        if not dim_result["is_consistent"]:
            errors.append(dim_result["error_message"])
        if nan_result["has_nans"]:
            errors.append(nan_result["error_message"])
        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "dimension_check": dim_result,
            "nan_check": nan_result
        }

if __name__ == '__main__':
    paths = generate_all_execution_paths()
    print(f'Generated {len(paths)} execution paths:')
    for path in paths:
        print(" -> ".join(op.name for op in path))
