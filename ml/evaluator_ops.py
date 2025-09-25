"""Op-related code for ml evaluator"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import sys
import time

from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from itertools import product
from pprint import pprint
from typing import Any, Callable

import numpy as np

from nkpylib.ml.feature_set import FeatureSet
from nkpylib.ml.features import make_jsonable

logger = logging.getLogger(__name__)

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
    variant: str|None = None
    timestamp: float = field(default_factory=time.time)
    error: str = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Result):
            return False
        return self.cache_key == other.cache_key

    def __repr__(self) -> str:
        status = "success" if self.is_success() else f"error: {self.error}"
        return (f"<Result op={self.op.name if self.op else None} variant={self.variant} status={status} cache_key={self.cache_key}>")

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

        Includes data only if the op is not intermediate.
        """
        return {
            "op_name": self.op.name if self.op else None,
            "success": self.is_success(),
            "error": self.error,
            "variant": self.variant,
            "provenance_str": self.get_provenance_string(),
            "provenance": [op.cache_key for op in self.provenance],
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "data_type": type(self.data).__name__ if self.data else None,
            "cache_key": self.cache_key,
            "data": None if self.op.is_intermediate else self.data,
        }

@dataclass
class Task:
    """A task to execute an operation with specific inputs."""
    status: str = "pending"  # pending, running, completed, failed
    op_cls: type[Op] = None # the op to run
    inputs: dict[str, Result] = field(default_factory=dict) # type name -> Result object
    variant_name: str|None = None # if applicable
    variant_kwargs: dict[str, Any]|None = None # if applicable
    error: Exception|None = None

    def __repr__(self) -> str:
        return (f"<Task op={self.op_cls.__name__} variant={self.variant_name} status={self.status}>")

    def same_type(self, other: Task) -> bool:
        """Checks if this task is the same type as the `other`.

        This checks op_cls, inputs, and variant_name
        """
        return (self.op_cls == other.op_cls and
                self.variant_name == other.variant_name and
                self.inputs == other.inputs)

    def run(self) -> Result | Exception:
        """Runs this task"""
        logger.info(f'Starting {self}')
        self.status = "running"
        op = self.op_cls(variant=self.variant_name, **(self.variant_kwargs or {}))
        try:
            result = op.execute(self.inputs)
            self.status = "completed"
            logger.info(f'Finished {self}')
            return result
        except Exception as e:
            self.status = "failed"
            logger.error(f"{self} failed with {type(e)}: {e}")
            # print stack trace
            logger.exception(e)
            self.error = e
            return e


class OpRegistry:
    """Registry of all available operations.

    This maintains indexes by input and output types to enable automatic
    graph construction based on type matching.

    It also deals with execution.
    """
    def __init__(self):
        # these contain lists of Op Classes, not instances!
        self.ops_by_output_type: dict[str, list[type[Op]]] = defaultdict(list)
        self.ops_by_input_type: dict[str, list[type[Op]]] = defaultdict(list)
        for op_cls in find_subclasses(Op):
            self.ops_by_output_type[op_cls.output_type].append(op_cls)
            for input_type in op_cls.input_types:
                self.ops_by_input_type[input_type].append(op_cls)
        pprint(self.ops_by_output_type)
        pprint(self.ops_by_input_type)
        # this contains a list of op instances
        self.all_ops: list[Op] = []
        self._results = defaultdict(dict) # output_type -> cache_key -> Result
        self.tasks = []
        self.done_tasks = []

    def start(self, op_cls: type[Op], inputs: Any) -> None:
        """Start processing tasks with given `op` and `inputs`.

        Note that this is a little different than normal op execution.
        """
        variants = op_cls.get_variants(inputs)
        if variants is None:
            variants = {None: {}}
        for name, kwargs in variants.items():
            task = Task(op_cls=op_cls, inputs=inputs, variant_name=name, variant_kwargs=kwargs)
            self.add_task(task)
        self.run_next()

    def add_task(self, task: Task) -> None:
        """Adds the given `task` if we don't already have it."""
        for existing in self.tasks + self.done_tasks:
            if existing.same_type(task):
                return
        self.tasks.append(task)
        logger.info(f'Added task {task}')

    def run_next(self) -> None:
        """Run the next pending task, if any."""
        if not self.tasks:
            logger.info("No more pending tasks to run")
            return
        task = self.tasks.pop(0)
        assert task.status == "pending"
        result = task.run()
        # the task running will also call create_tasks once it finishes
        self.done_tasks.append(task)

    def create_tasks(self, result: Result) -> None:
        """Create new work tasks to do based on a new result."""
        consumers = self.get_consumers(result.op.output_type)
        logger.info(f'Creating new tasks from {result}: consumers={[c.__class__.__name__ for c in consumers]}')
        for consumer in consumers:
            input_types = sorted(consumer.input_types)
            # generate lists of results for each input type
            input_results = [list(self._results.get(t, {}).values()) for t in input_types]
            if not input_results:
                continue
            # now generate cartesian product of tasks
            # (if any has an empty dict, it will result in no tasks)
            for cur_results in product(*input_results):
                inputs = {t: r for t, r in zip(input_types, cur_results)}
                variants = consumer.get_variants(inputs)
                if variants is None:
                    variants = {None: {}}
                for name, kwargs in variants.items():
                    task = Task(op_cls=consumer, inputs=inputs, variant_name=name, variant_kwargs=kwargs)
                    self.add_task(task)
        self.run_next()

    @staticmethod
    def register(op: Op) -> None:
        """Register an operation in the registry."""
        registry = OpRegistry.get_global_op_registry()
        registry.all_ops.append(op)

    def get_producers(self, type_name: str) -> list[type[Op]]:
        """Get all operations that produce a given type."""
        return self.ops_by_output_type[type_name]

    def get_consumers(self, type_name: str) -> list[type[Op]]:
        """Get all operations that consume a given type."""
        return self.ops_by_input_type[type_name]

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
        result = Result(data=data,
                        provenance=provenance,
                        cache_key=op.cache_key,
                        variant=op.variant,
                        error=error,
                        metadata=metadata or {})
        registry._results[op.output_type][op.cache_key] = result
        registry.create_tasks(result)
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


class Op(ABC):
    """Base operation with input/output type definitions.

    Each operation defines:
    - `name`: unique identifier for this operation
    - `input_types`: set of type names this operation requires as input
    - `output_type`: single type name this operation produces
    - `is_intermediate`: if True, the output data is not stored in results

    These are by default defined at the class level, but if they are dependent on init parameters,
    you can also define them at the instance level. They are always referred to be `self.name`
    (e.g.), so python method resolution order will find the instance var first if it exists, then
    the class var.

    When created, the operation automatically registers itself with the global registry.
    """
    name = ""
    input_types: frozenset[str] = frozenset()
    output_type = ""

    # by default we assume ops are not intermediate. If they are, override this
    is_intermediate = False

    def __init__(self, variant: str|None=None, name: str|None = None, input_types: frozenset[str]|None=None, output_type: str|None=None):
        """Initialize this operation.

        - variant: optional variant name for this instance
        - name: optional instance-specific name
        - input_types: optional instance-specific input types
        - output_type: optional instance-specific output type

        If you don't specify these, the class-level ones are used. For `name`, if both are missing,
        the name of the python class is assigned to the class variable. For `output_type`, we
        require that either the class-level var or the arg to this func is non-empty.
        """
        self.variant = variant
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

    @classmethod
    def get_variants(cls, inputs: dict[str, Any]) -> dict[str, Any]|None:
        """Gets variants of this operation based on the given inputs.

        The output is a dict mapping a variant name to a dict of init kwargs.
        If there's only one variant, return `None` (default behavior).
        """
        return None

    def execute(self, inputs: dict[str, Any]) -> Any:
        """Execute the operation with given inputs.

        - inputs: dict mapping type names to actual data

        Returns the output data of type `self.output_type`.
        """
        logger.debug(f'Going to execute op {self.name} with inputs: {inputs}')
        try:
            # try to get the underlying data from the result objects
            op_inputs = {k: v.data for k, v in inputs.items()}
        except Exception:
            op_inputs = inputs
        self.cache_key = self.get_cache_key(op_inputs)
        logger.info(f'Executing op {self.name} ({self.variant}) -> {self.cache_key}')
        out = self._execute(op_inputs)
        result = OpRegistry.add_result(self, out, inputs=inputs or op_inputs)
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
    is_intermediate = True

    #TODO return cartesian product of inputs as variants
    def _execute(self, inputs: dict[str, Any], **kwargs) -> Any:
        print(f'Got inputs: {inputs}')
        paths = inputs['paths']
        return FeatureSet(paths, **kwargs)


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
    is_intermediate = True

    @classmethod
    def get_variants(cls, inputs: dict[str, Any]) -> dict[str, Any]|None:
        """Returns different variants of this op based on normalization options."""
        return None #FIXME
        ret = {}
        for normed, scale_mean, scale_std in product([True, False], repeat=3):
            variant_name = f"normed:{int(normed)}_mean:{int(scale_mean)}_std:{int(scale_std)}"
            ret[variant_name] = {
                "normed": normed,
                "scale_mean": scale_mean,
                "scale_std": scale_std
            }
        return ret

    def __init__(self, normed: bool = False, scale_mean: bool = True, scale_std: bool = True, **kw):
        self.normed = normed
        self.scale_mean = scale_mean
        self.scale_std = scale_std
        super().__init__(**kw)

    def _execute(self, inputs: dict[str, Any]) -> Any:
        fs = inputs["feature_set"]
        keys, emb = fs.get_keys_embeddings(
            normed=self.normed,
            scale_mean=self.scale_mean,
            scale_std=self.scale_std
        )
        return (keys, emb)

def manual_main():
    # run sequence of load, checks, basic and look at results
    ops = [
        LoadEmbeddingsOp(paths=[sys.argv[1]]),
        CheckDimensionsOp(),
        CheckNaNsOp(),
        BasicChecksOp(),
    ]
    for op in ops:
        print(f"\nRunning op: {op.name} (inputs: {op.input_types}, output: {op.output_type})")
        result = op.execute()
        print(f"Result: {result.to_dict()}")
        print(f'Registry results: {dict(reg._results)}')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #manual_main()
    reg = OpRegistry.get_global_op_registry()
    reg.start(LoadEmbeddingsOp, {'paths': [sys.argv[1]]})
    print(json.dumps(reg.results_to_dict(), indent=2))
