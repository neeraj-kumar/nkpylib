"""Op-related code for ml evaluator.

TODO:
- updating progress to disk
  - global stats (tasks exceuted and where, throughput, etc)
- bayesian optimization
- flexible logging levels per op
- caching
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import sys
import time

from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from itertools import product
from pprint import pprint
from typing import Any, Callable

import numpy as np

from nkpylib.ml.features import make_jsonable
from nkpylib.ml.feature_set import JsonLmdb

logger = logging.getLogger(__name__)

# Global manager instance that all Ops will register with
_global_op_manager: 'OpManager'|None = None

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
    - The actual output produced
    - Full provenance chain (list of operations that led to this result)
    - Cache key for deduplication
    - Timestamp and error information

    The operation that produced this result is always the last item in the provenance chain.
    """
    output: Any
    provenance: list[Op]
    cache_key: str
    variant: str|None = None
    timestamps: dict[str, float] = field(default_factory=dict) # status -> timestamp
    error: Exception|None = None

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Result):
            return False
        return self.cache_key == other.cache_key

    def __repr__(self) -> str:
        status = "success" if self.is_success else f"error: {self.error}"
        return (f"<Result op={self.op.name if self.op else None} variant={self.variant} status={status} cache_key={self.cache_key}>")

    @property
    def op(self) -> Op|None:
        """The operation that produced this result (last in provenance chain)."""
        return self.provenance[-1] if self.provenance else None

    @property
    def start_ts(self) -> float:
        """Timestamp when the operation started."""
        return self.timestamps.get('running', 0)

    @property
    def end_ts(self) -> float:
        """Timestamp when the operation finished."""
        return self.timestamps.get('finished', 0)

    @property
    def is_success(self) -> bool:
        """Returns True if the operation completed successfully."""
        return self.error is None

    @property
    def provenance_str(self) -> str:
        """Human-readable provenance chain."""
        return " -> ".join(op.name for op in self.provenance)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/reporting.

        Includes output only if the op is not intermediate.
        """
        return {
            "op_name": self.op.name if self.op else None,
            "is_success": self.is_success,
            "error": self.error,
            "variant": self.variant,
            "provenance_str": self.provenance_str,
            "provenance": [op.cache_key for op in self.provenance],
            "timestamps": self.timestamps,
            "input_types": sorted(self.op.input_types) if self.op else None,
            "output_type": self.op.output_type if self.op else None,
            "output_py_type": type(self.output).__name__ if self.output else None,
            "cache_key": self.cache_key,
            "output": None if self.op.is_intermediate else self.output,
        }

task_statuses = ("pending", "running", "completed", "failed")

@dataclass
class Task:
    """A task to execute an operation with specific inputs."""
    _status: str = "pending"
    op_cls: type[Op]|None = None # the op to run
    inputs: dict[str, Result] = field(default_factory=dict) # type name -> Result object
    variant_name: str|None = None # if applicable
    variant_kwargs: dict[str, Any]|None = None # if applicable
    op: Op|None = None # the instantiated op (after running)
    output: Any = None # the output (after running)
    error: Exception|None = None # any error encountered
    timestamps: dict[str, float] = field(default_factory=dict) # status -> timestamp

    def __post_init__(self) -> None:
        assert self.op_cls is not None, "Task must have an op_cls"
        assert self._status in task_statuses, f"Invalid status {self._status}"
        self.timestamps[self._status] = time.time()

    def __repr__(self) -> str:
        return (f"<Task op={self.op_cls.__name__} variant={self.variant_name} status={self.status}>")

    # add a setter for status that also updates timestamps
    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, value: str) -> None:
        assert value in task_statuses, f"Invalid status {value}"
        self._status = value
        ts = time.time()
        self.timestamps[value] = ts
        if value in ('completed', 'failed'):
            self.timestamps['finished'] = ts
            self.timestamps['elapsed'] = ts - self.timestamps.get('running', ts)
        if value == 'running':
            self.timestamps['waiting'] = ts - self.timestamps.get('pending', ts)

    def same_type(self, other: Task) -> bool:
        """Checks if this task is the same type as the `other`.

        This checks op_cls, inputs, and variant_name
        """
        return (self.op_cls == other.op_cls and
                self.variant_name == other.variant_name and
                self.inputs == other.inputs)

    @property
    def run_mode(self) -> str:
        """Returns where this task should run (main/thread/process)."""
        return self.op_cls.run_mode

    def run(self) -> Task:
        """Runs this task, returning self when done."""
        logger.info(f'Starting {self}')
        self.status = "running"
        self.op = self.op_cls(variant=self.variant_name, **(self.variant_kwargs or {}))
        self.output, self.error = self.op.execute(self.inputs)
        self.status = 'failed' if self.error else 'completed'
        logger.info(f'Finished {self}')
        return self


class TaskPool:
    """A wrapper around a thread or process pool that manages task execution."""
    def __init__(self, pool_type: str, max_workers: int):
        """Initializes the pool (either 'thread' or 'process') with given number of workers."""
        cls_by_type = dict(thread=ThreadPoolExecutor, process=ProcessPoolExecutor)
        self.pool = cls_by_type[pool_type](max_workers=max_workers)
        self.max_workers = max_workers
        self.futures = {}  # Future -> Task mapping

    @property
    def n_working(self) -> int:
        """Returns number of currently running tasks."""
        return len([f for f in self.futures if not f.done()])

    @property
    def is_active(self) -> bool:
        """Returns if this pool is active (has running or finished tasks)."""
        return bool(self.futures)

    @property
    def has_capacity(self) -> bool:
        """Returns True if we have capacity to run more tasks."""
        return self.n_working < self.max_workers

    def submit_task_if_free(self, task: Task) -> Future|None:
        """Submit task if we have capacity, return Future or None."""
        if not self.has_capacity:
            return None
        future = self.pool.submit(task.run)
        self.futures[future] = task
        return future

    def get_completed_tasks(self) -> list[Task]:
        """Get all completed tasks, removing them from tracking.

        This updates the state on the task appropriately.
        """
        completed = []
        done_futures = [f for f in self.futures if f.done()]
        for future in done_futures:
            task = self.futures.pop(future)
            try:
                task = future.result()
            except Exception as e:
                task.status = 'failed'
                task.error = e
            completed.append(task)
        return completed

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Submits a function to the pool."""
        f = self.pool.submit(fn, *args, **kwargs)
        return f

    def submit_when_available(self, fn: Callable, *args, **kwargs) -> Future:
        """Submits a function to the pool when we have capacity, blocking till then."""
        for f in as_completed(self.futures):
            if self.has_capacity:
                return self.submit(fn, *args, **kwargs)
        else: # we have capacity now
            return self.submit(fn, *args, **kwargs)


class OpManager:
    """Manager of operations and task runner.

    This maintains indexes by input and output types to enable automatic
    graph construction based on type matching.

    It also deals with execution.
    """
    @staticmethod
    def get() -> OpManager:
        """Get the global manager, creating it if needed."""
        global _global_op_manager
        if _global_op_manager is None:
            _global_op_manager = OpManager()
        return _global_op_manager # type: ignore

    def __init__(self,
                 n_procs: int=4,
                 n_threads: int=4,
                 results_db_path: str=f'logs/results_{int(time.time())}.lmdb'):
        self.n_procs = n_procs
        self.n_threads = n_threads
        self.proc_pool = TaskPool('process', n_procs)
        self.thread_pool = TaskPool('thread', n_threads)
        # these contain lists of Op Classes, not instances!
        self.ops_by_output_type: dict[str, list[type[Op]]] = defaultdict(list)
        self.ops_by_input_type: dict[str, list[type[Op]]] = defaultdict(list)
        for op_cls in find_subclasses(Op):
            if op_cls.enabled is False:
                continue
            self.ops_by_output_type[op_cls.output_type].append(op_cls)
            for input_type in op_cls.input_types:
                self.ops_by_input_type[input_type].append(op_cls)
        # this contains a list of op instances
        self.all_ops: list[Op] = []
        self._results = defaultdict(dict) # output_type -> cache_key -> Result
        self.tasks = []
        self.done_tasks = []
        self.results_db = JsonLmdb.open(results_db_path, flag='c') if results_db_path else None

    def start(self, op_cls: type[Op], inputs: Any) -> None:
        """Start processing tasks with given `op` and `inputs`.

        This runs until we finish all generated tasks.
        """
        self.add_all_variants(op_cls, inputs)
        self.run_tasks()

    def add_task(self, task: Task) -> None:
        """Adds the given `task` to our run queue if we don't already have it."""
        for existing in self.tasks + self.done_tasks:
            if existing.same_type(task):
                return
        self.tasks.append(task)
        logger.info(f'Added task {task}')

    def add_all_variants(self, op_cls: type[Op], inputs: dict[str, Result]) -> None:
        """Adds tasks for all variants of the given `op_cls` with the given `inputs`."""
        variants = op_cls.get_variants(inputs)
        if variants is None:
            variants = {None: {}}
        for name, kwargs in variants.items():
            task = Task(op_cls=op_cls, inputs=inputs, variant_name=name, variant_kwargs=kwargs)
            self.add_task(task)

    def create_tasks(self, result: Result|Exception) -> None:
        """Create new work tasks to do based on a new result."""
        if isinstance(result, Exception):
            return
        consumers = self.get_consumers(result.op.output_type)
        logger.info(f'Creating new tasks from {result}: consumers={[c.__name__ for c in consumers]}')
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
                self.add_all_variants(consumer, inputs)

    def run_tasks(self) -> None:
        """Runs our task list using our thread and process pools."""
        t_pool, p_pool = self.thread_pool, self.proc_pool
        pools = dict(thread=t_pool, process=p_pool)
        matching_tasks = lambda *modes: [t for t in self.tasks if t.run_mode in modes and t.status == "pending"]

        while self.tasks or t_pool.is_active or p_pool.is_active:
            # submit pool tasks if they have capacity
            for task in matching_tasks("thread", "process"):
                if pools[task.run_mode].submit_task_if_free(task):
                    self.tasks.remove(task)
            # if we still have pending tasks, run one in main thread
            for_main = matching_tasks("main")
            if for_main:
                task = for_main[0]
                self.tasks.remove(task)
                task.run()
                self.finish_task(task)
            # check for completed tasks
            for pool in pools.values():
                for task in pool.get_completed_tasks():
                    self.finish_task(task)
            # avoid busy loop
            time.sleep(0.1)

    @staticmethod
    def register(op: Op) -> None:
        """Register an operation."""
        OpManager.get().all_ops.append(op)

    def get_producers(self, type_name: str) -> list[type[Op]]:
        """Get all operations that produce a given type."""
        return self.ops_by_output_type[type_name]

    def get_consumers(self, type_name: str) -> list[type[Op]]:
        """Get all operations that consume a given type."""
        return self.ops_by_input_type[type_name]

    def finish_task(self, task: Task) -> None:
        """Finishes a `task`.

        This creates a result from the task's output and adds it to our results.
        It also moves the task to done_tasks and creates new tasks based on the result.
        """
        self.done_tasks.append(task)
        result = self.add_result_from_task(task)
        self.create_tasks(result)

    @staticmethod
    def add_result_from_task(task: Task) -> Result:
        """Add a result from a completed `task`.

        This creates a `Result` object with full provenance and stores it in our results index.
        It also logs the result to our database.
        Returns the created `Result` object.
        """
        assert task.status in ('completed', 'failed'), f'Task {task} not done yet'
        op = task.op
        assert op is not None, f'Task {task} has no op'
        provenance = []
        for input_type in op.input_types:
            if input_type in task.inputs and hasattr(task.inputs[input_type], 'provenance'):
                provenance.extend(task.inputs[input_type].provenance)
        provenance.append(op)
        result = Result(output=task.output,
                        provenance=provenance,
                        cache_key=op.cache_key,
                        variant=op.variant,
                        error=task.error,
                        timestamps=task.timestamps)
        om = OpManager.get()
        om._results[op.output_type][op.cache_key] = result
        om.log_result(result)
        return result

    def log_result(self, result: Result) -> None:
        """Logs the given `result` to our database, as well as general logging updates."""
        if self.results_db is None:
            return
        print(f'Logging result: {result}')
        if isinstance(result, Exception):
            return
        all_results = []
        for results in self._results.values():
            all_results.extend(results.values())
        status_key = 'status:success' if result.is_success else 'status:failed'
        to_update = {
            # full result for this key
            f'key:{result.cache_key}': make_jsonable(result.to_dict()),
            # all cache keys of this type
            f'type:{result.op.output_type}': sorted(self._results[result.op.output_type]),
            # all cache keys of this op class
            f'op_name:{result.op.name}': sorted({r.cache_key for r in all_results if r.op and isinstance(r.op, result.op.__class__)}),
            # all cache keys with this status
            status_key: sorted({r.cache_key for r in all_results if r.is_success == result.is_success}),
            # all cache keys sorted by various criteria
            'by:start_time': [(r.cache_key, r.start_ts) for r in sorted(all_results, key=lambda r: r.start_ts)],
            'by:end_time': [(r.cache_key, r.end_ts) for r in sorted(all_results, key=lambda r: r.end_ts)],
            # current update time
            'last_update': time.time(),
        }
        print(f'Logged results: {to_update.keys()}')
        self.results_db.update(to_update)
        self.results_db.sync()

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
    - `is_intermediate`: if True, the output is not stored in results

    These are by default defined at the class level, but if they are dependent on init parameters,
    you can also define them at the instance level. They are always referred to be `self.name`
    (e.g.), so python method resolution order will find the instance var first if it exists, then
    the class var.

    When created, the operation automatically registers itself with the global manager.
    """
    name = ""
    input_types: frozenset[str] = frozenset()
    output_type = ""

    # by default we assume ops are not intermediate. If they are, override this
    is_intermediate = False

    # you can set this in your class to disable the op (e.g. for debugging)
    enabled = True

    # where this op should run (main=main process, thread=thread pool, process=process pool)
    run_mode = "main" # main, thread, process

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
        assert self.output_type, f"{self} must have a non-empty output_type"
        assert self.enabled, f"{self} is disabled!"
        assert self.run_mode in ("main", "thread", "process"), f"{self} has invalid run_mode {self.run_mode}"
        OpManager.register(self)
        self.cache_key = ''

    def __repr__(self) -> str:
        cls_name = 'Op' if self.enabled else 'DisabledOp'
        variant_str = f' variant={self.variant}' if self.variant else ''
        return f'<{cls_name} name={self.name}{variant_str}>'

    @classmethod
    def get_variants(cls, inputs: dict[str, Any]) -> dict[str, Any]|None:
        """Gets variants of this operation based on the given inputs.

        The output is a dict mapping a variant name to a dict of init kwargs.
        If there's only one variant, return `None` (default behavior).
        """
        return None

    def execute(self, inputs: dict[str, Any]) -> tuple[Any, Exception|None]:
        """Execute the operation with given inputs.

        - inputs: dict mapping type names to Result objects

        Returns `(output, error)` where the output is of type `self.output_type` and the error
        is the exception if any occurred, else None.
        """
        logger.debug(f'Going to execute op {self.name} with inputs: {inputs}')
        try:
            # try to get the underlying output from the result objects
            op_inputs = {k: v.output for k, v in inputs.items()}
        except Exception:
            op_inputs = inputs
        self.cache_key = self.get_cache_key(op_inputs)
        logger.info(f'Executing op {self.name} ({self.variant}) -> {self.cache_key}')
        out, error = None, None
        try:
            output = self._execute(op_inputs)
        except Exception as e:
            logger.error(f'Op {self.name} failed with {type(e)}: {e}')
            # print stack trace
            logger.exception(e)
            error = e
        return output, error

    @abstractmethod
    def _execute(self, inputs: dict[str, Any]) -> Any:
        """Execute the operation with given inputs.

        - inputs: dict mapping type names to actual data

        Returns the output type `self.output_type`.
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
