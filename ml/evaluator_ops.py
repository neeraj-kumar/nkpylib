"""Op-related code for ml evaluator"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import time

from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from nkpylib.ml.feature_set import FeatureSet
from nkpylib.ml.features import make_jsonable

# Global registry instance that all Ops will register with
_global_op_registry = None

def find_subclasses(cls) -> list[type[Op]]:
    """Find all concrete (non-abstract) subclasses of `cls`."""
    ret = []
    for subclass in cls.__subclasses__():
        if not inspect.isabstract(subclass):
            ret.append(subclass)
        ret.extend(find_subclasses(subclass))  # Recursive for nested inheritance
    return ret


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
            "provenance_str": self.get_provenance_string(),
            "provenance": [op.cache_key for op in self.provenance],
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "data_type": type(self.data).__name__ if self.data else None,
            "cache_key": self.cache_key,
        }


class OpRegistry:
    """Registry of all available operations.

    This maintains indexes by input and output types to enable automatic
    graph construction based on type matching.

    It also deals with execution.
    """
    def __init__(self):
        self.ops_by_output_type: dict[str, list[Op]] = defaultdict(list)
        self.ops_by_input_type: dict[str, list[Op]] = defaultdict(list)
        self.all_ops: list[Op] = []
        self._results = defaultdict(dict) # output_type -> cache_key -> Result

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

    @staticmethod
    def add_result(op: Op,
                   data: Any,
                   inputs: dict[str, Any],
                   error: str = None,
                   metadata: dict[str, Any] = None) -> Result:
        """Add a result produced by an operation to the registry.

        This creates a `Result` object with full provenance and stores it.
        """
        registry = OpRegistry.get_global_op_registry()
        provenance = []
        for input_type in op.input_types:
            if input_type in inputs and hasattr(inputs[input_type], 'provenance'):
                provenance.extend(inputs[input_type].provenance)
        provenance.append(op)
        result = Result(data=data, provenance=provenance, cache_key=op.cache_key, error=error, metadata=metadata or {})
        registry._results[op.output_type][op.cache_key] = result
        return result

    def results_to_dict(self) -> dict[str, Any]:
        """Serialize all our results for logging/reporting.

        This calls `make_jsonable(result.to_dict())` on each Result.
        """
        ret = {}
        for type_name, results in self._results.items():
            ret[type_name] = {cache_key: make_jsonable(result.to_dict())
                              for cache_key, result in results.items()}
        return ret

    @staticmethod
    def get_results(type_names: set[str]) -> dict[str, Any]:
        """Get the latest results for the given set of type names.

        Returns a dict mapping type names to `Result` objects. You can get the underlying data
        via `result.data`.
        """
        registry = OpRegistry.get_global_op_registry()
        results = {}
        for type_name in type_names:
            type_results = registry._results.get(type_name, {})
            if type_results:
                # Get the most recent result
                latest_result = max(type_results.values(), key=lambda r: r.timestamp)
                results[type_name] = latest_result
        return results


class Op(ABC):
    """Base operation with input/output type definitions.

    Each operation defines:
    - `name`: unique identifier for this operation
    - `input_types`: set of type names this operation requires as input
    - `output_type`: single type name this operation produces

    These are by default defined at the class level, but if they are dependent on init parameters,
    you can also define them at the instance level. They are always referred to be `self.name`
    (e.g.), so python method resolution order will find the instance var first if it exists, then
    the class var.

    When created, the operation automatically registers itself with the global registry.
    """
    name: str = ""
    input_types: frozenset[str] = frozenset()
    output_type: str = ""

    def __init__(self, name: str|None = None, input_types: frozenset[str]|None=None, output_type: str|None=None):
        """Initialize the operation, optionally overriding class-level attributes.

        - name: optional instance-specific name
        - input_types: optional instance-specific input types
        - output_type: optional instance-specific output type

        If you don't specify these, the class-level ones are used. For `name`, if both are missing,
        the name of the python class is assigned to the class variable. For `output_type`, we
        require that either the class-level var or the arg to this func is non-empty.
        """
        if name is not None:
            self.name = name
        if input_types is not None:
            self.input_types = input_types
        if output_type is not None:
            self.output_type = output_type
        # make sure we have a name of some sort
        if not name and not self.__class__.name:
            self.__class__.name = self.__class__.__name__
        assert self.output_type, f"Op {self.name} must have a non-empty output_type"
        OpRegistry.register(self)
        self.cache_key = None

    def execute(self, inputs: dict[str, Any]|None=None) -> Any:
        """Execute the operation with given inputs.

        - inputs: dict mapping type names to actual data

        If the inputs are None, we get them from the registry

        Returns the output data of type `self.output_type`.
        """
        if inputs is None:
            full_inputs = OpRegistry.get_results(self.input_types)
            inputs = {k: v.data for k, v in full_inputs.items()}
        print(f'Going to execute op {self.name} with inputs: {inputs}, {self.input_types}, {self.__class__.input_types}')
        self.cache_key = self.get_cache_key(inputs)
        out = self._execute(inputs)
        result = OpRegistry.add_result(self, out, inputs=full_inputs)
        return result

    @abstractmethod
    def _execute(self, inputs: dict[str, Any]) -> Any:
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

    def _execute(self, inputs: dict[str, Any]) -> Any:
        return FeatureSet(self.paths, **self.kwargs)


class CheckDimensionsOp(Op):
    """Check that all embeddings have consistent dimensions."""

    name = "check_dimensions"
    input_types = frozenset({"feature_set"})
    output_type = "dimension_check_result"

    def _execute(self, inputs: dict[str, Any]) -> Any:
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

    def _execute(self, inputs: dict[str, Any]) -> Any:
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

    def _execute(self, inputs: dict[str, Any]) -> Any:
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


class NormalizeOp(Op):
    """Normalize embeddings from a FeatureSet based on normalization parameters."""

    name = "normalize"
    input_types = frozenset({"feature_set"})
    output_type = "normalized_embeddings"

    def __init__(self, normed: bool = False, scale_mean: bool = True, scale_std: bool = True):
        self.normed = normed
        self.scale_mean = scale_mean
        self.scale_std = scale_std
        super().__init__()

    def _execute(self, inputs: dict[str, Any]) -> Any:
        fs = inputs["feature_set"]
        keys, emb = fs.get_keys_embeddings(
            normed=self.normed,
            scale_mean=self.scale_mean,
            scale_std=self.scale_std
        )
        return (keys, emb)

if __name__ == '__main__':
    # run sequence of load, checks, basic and look at results
    ops = [
        LoadEmbeddingsOp(paths=[sys.argv[1]]),
        CheckDimensionsOp(),
        CheckNaNsOp(),
        BasicChecksOp(),
    ]
    reg = OpRegistry.get_global_op_registry()
    for op in ops:
        print(f"\nRunning op: {op.name} (inputs: {op.input_types}, output: {op.output_type})")
        result = op.execute()
        print(f"Result: {result.to_dict()}")
        print(f'Registry results: {dict(reg._results)}')
    print(json.dumps(reg.results_to_dict(), indent=2))
