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
            "provenance": self.get_provenance_string(),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "data_type": type(self.data).__name__ if self.data else None,
            "cache_key": self.cache_key,
        }


@dataclass
class ExecutionStep:
    """A single step in an execution plan."""
    step_id: str
    op_class: type[Op]
    params: dict[str, Any]
    input_mappings: dict[str, str]  # input_type -> step_id that produces it
    output_type: str

    @staticmethod
    def execute_plan(steps: list[ExecutionStep]) -> dict[str, Result]:
        """Execute a list of steps and return results for each step."""
        results: dict[str, Result] = {}

        for step in steps:
            # Gather inputs from previous steps
            inputs = {}
            for input_type, source_step_id in step.input_mappings.items():
                inputs[input_type] = results[source_step_id].data

            # Instantiate and execute the op
            op_instance = step.op_class(**step.params)
            data = op_instance.execute(inputs)

            # Build provenance chain
            provenance = [op_instance]
            for source_step_id in step.input_mappings.values():
                provenance = results[source_step_id].provenance + provenance

            # Store result
            results[step.step_id] = Result(
                data=data,
                provenance=provenance,
                cache_key=op_instance.get_cache_key(inputs)
            )

        return results


@dataclass
class ExecutionPlan:
    """A complete execution plan as an ordered list of steps."""
    steps: list[ExecutionStep]
    final_outputs: list[str]  # step_ids of final results we care about

    def execute(self) -> dict[str, Result]:
        """Execute this plan and return final results."""
        all_results = ExecutionStep.execute_plan(self.steps)
        return {step_id: all_results[step_id] for step_id in self.final_outputs}


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

    @staticmethod
    def gen_execution_plans(start_types: set[str] = None,
                           target_types: set[str] = None,
                           max_depth: int = 5) -> list[ExecutionPlan]:
        """Generate all valid execution plans using registered ops.

        - start_types: Types that are available at the beginning (None means use ops with empty input_types)
        - target_types: Types we want to end up with (None means any final type is acceptable)
        - max_depth: Maximum number of operations in a path

        Returns list of ExecutionPlan objects.
        """
        reg = OpRegistry.get_global_op_registry()
        if start_types is None:
            # Use output types from ops that have no input requirements
            op_classes = find_subclasses(Op)
            start_types = {op_class.output_type for op_class in op_classes if not op_class.input_types}
        if target_types is None:
            target_types = set()
        print(f'Start types: {start_types}, Target types: {target_types}')

        plans = []

        def build_plan(current_steps: list[ExecutionStep],
                      available_types: dict[str, str],  # type -> step_id
                      remaining_targets: set[str]):

            if not remaining_targets or len(current_steps) >= max_depth:
                if current_steps:  # Only add non-empty plans
                    final_outputs = [step.step_id for step in current_steps 
                                   if step.output_type in target_types or not target_types]
                    if final_outputs:
                        plans.append(ExecutionPlan(
                            steps=current_steps.copy(),
                            final_outputs=final_outputs
                        ))
                return

            # Try each op class that could help
            op_classes = find_subclasses(Op)
            for op_class in op_classes:
                # Skip if this op doesn't produce something we want
                if target_types and op_class.output_type not in remaining_targets:
                    continue

                # Check if we can satisfy its inputs
                if not op_class.input_types.issubset(available_types.keys()):
                    continue

                # Skip if we already have this op type in the plan (avoid cycles)
                if any(step.op_class == op_class for step in current_steps):
                    continue

                # Create step
                step = ExecutionStep(
                    step_id=f"{op_class.name}_{len(current_steps)}",
                    op_class=op_class,
                    params={},  # Could generate parameter variations here
                    input_mappings={t: available_types[t] for t in op_class.input_types},
                    output_type=op_class.output_type
                )

                # Recurse
                new_steps = current_steps + [step]
                new_available = available_types.copy()
                new_available[step.output_type] = step.step_id
                new_targets = remaining_targets - {step.output_type} if target_types else remaining_targets

                build_plan(new_steps, new_available, new_targets)

        # Start with ops that need no inputs
        op_classes = find_subclasses(Op)
        for op_class in op_classes:
            if not op_class.input_types or op_class.input_types.issubset(start_types):
                step = ExecutionStep(
                    step_id=f"{op_class.name}_0",
                    op_class=op_class,
                    params={},
                    input_mappings={t: f"start_{t}" for t in op_class.input_types & start_types},
                    output_type=op_class.output_type
                )

                available = dict(start_types) if isinstance(start_types, dict) else {t: f"start_{t}" for t in start_types}
                available[step.output_type] = step.step_id
                targets = target_types.copy() if target_types else {step.output_type}

                build_plan([step], available, targets)

        return plans


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

    def execute(self, inputs: dict[str, Any]) -> Any:
        fs = inputs["feature_set"]
        keys, emb = fs.get_keys_embeddings(
            normed=self.normed,
            scale_mean=self.scale_mean,
            scale_std=self.scale_std
        )
        return (keys, emb)

if __name__ == '__main__':
    plans = OpRegistry.gen_execution_plans(target_types={"basic_checks_report"})
    print(f'Generated {len(plans)} execution plans:')
    for i, plan in enumerate(plans):
        print(f"Plan {i}: {len(plan.steps)} steps")
        for step in plan.steps:
            inputs = " + ".join(step.input_mappings.keys()) if step.input_mappings else "∅"
            print(f"  {step.step_id}: {inputs} → {step.output_type}")
        print(f"  Final outputs: {plan.final_outputs}")
        print()
