"""Op-related code for ml evaluator.

TODO:
- continuation of runs?
- curses-style view of logs vs tasks vs other stuff?
  - different panes? scrollable?
  - separate op logs from management logs?
  - maybe write stuff to disk and use e.g. multitail?
- better visualization of existing results
- compare eval runs by id
- find structurally similar outputs in different runs (i.e., same provenance/keys)
- annotate existing runs with warnings
- separate storage of warnings?
- updating progress to disk
  - global stats (tasks executed and where, throughput, etc)
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
from pprint import pprint, pformat
from typing import Any, Callable

import numpy as np

from nkpylib.utils import getmem
from nkpylib.time_utils import PerfTracker
from nkpylib.ml.features import make_jsonable
from nkpylib.ml.feature_set import JsonLmdb

# Create specialized loggers for different components
logger = logging.getLogger(__name__)
task_logger = logging.getLogger("evaluator.tasks")
perf_logger = logging.getLogger("evaluator.perf")
result_logger = logging.getLogger("evaluator.results")
error_logger = logging.getLogger("evaluator.errors")

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
    - Full provenance chain (list of previous results that led to this result)
    - Key for deduplication
    - Timestamp and error information

    The operation that produced this result is accessible via the op property.
    """
    output: Any
    op: Op  # The operation that produced this result
    provenance: list['Result']  # List of previous Result objects
    key: str
    variant: str|None = None
    timestamps: dict[str, float] = field(default_factory=dict) # status -> timestamp
    error: Exception|None = None

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Result):
            return False
        return self.key == other.key

    def __repr__(self) -> str:
        status = "success" if self.is_success else f"error: {self.error}"
        return (f"<R {self.op.name}:{self.variant} {self.key} [{status}]>")

    def __hash__(self) -> int:
        return hash(self.key+str(self.variant))

    @property
    def mem(self) -> int:
        """Approximate memory usage of this result in bytes."""
        return getmem(self.output) + sum(r.mem for r in self.provenance)

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
        return " -> ".join(r.op.name for r in self.provenance) + (f" -> {self.op.name}")

    @property
    def all_ops(self) -> list[Op]:
        """Returns all operations in the provenance chain, including this result's op."""
        return [r.op for r in self.provenance] + [self.op]

    @property
    def all_keys(self) -> list[str]:
        """Returns all keys in the provenance chain, including this result's key."""
        return [r.key for r in self.provenance] + [self.key]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/reporting.

        Includes output only if the op is not intermediate.
        """
        return {
            "op_name": self.op.name,
            "is_success": self.is_success,
            "error": self.error,
            "variant": self.variant,
            "provenance_str": self.provenance_str,
            "provenance": [r.key for r in self.provenance],
            "timestamps": self.timestamps,
            "input_types": sorted(self.op.input_types),
            "output_types": sorted(self.op.output_types),
            "output_py_type": type(self.output).__name__ if self.output is not None else None,
            "key": self.key,
            "output": None if self.op.is_intermediate else self.output,
        }

task_statuses = ("pending", "running", "completed", "failed")

@dataclass
class Task:
    """A task to execute an operation with specific inputs."""
    op_cls: type[Op] # the op to run
    _status: str = "pending"
    inputs: dict[str, Result] = field(default_factory=dict) # type name -> Result object
    variant_name: str|None = None # if applicable
    variant_kwargs: dict[str, Any]|None = None # if applicable
    op: Op|None = None # the instantiated op (after running)
    output: Any = None # the output (after running)
    error: Exception|None = None # any error encountered
    timestamps: dict[str, float] = field(default_factory=dict) # status -> timestamp

    def __post_init__(self) -> None:
        assert self._status in task_statuses, f"Invalid status {self._status}"
        self.timestamps[self._status] = time.time()

    def __repr__(self) -> str:
        return (f"<T {self.name} [{self.status}]>")

    @property
    def name(self) -> str:
        ret = self.op_cls.name
        if self.variant_name:
            ret += f':{self.variant_name}'
        return ret

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
        #return 'main' #FIXME
        return self.op_cls.run_mode

    def run(self) -> Task:
        """Runs this task, returning self when done."""
        task_logger.info(f'Starting {self}')
        self.status = "running"
        self.op = self.op_cls(variant=self.variant_name, **(self.variant_kwargs or {}))
        self.output, self.error = self.op.execute(self.inputs)
        self.status = 'failed' if self.error else 'completed'
        task_logger.info(f'Finished {self}')
        return self


class TaskPool:
    """A wrapper around a thread or process pool that manages task execution."""
    def __init__(self, pool_type: str, max_workers: int):
        """Initializes the pool (either 'thread' or 'process') with given number of workers."""
        cls_by_type = dict(thread=ThreadPoolExecutor, process=ProcessPoolExecutor)
        self.pool_type = pool_type
        self.pool = cls_by_type[pool_type](max_workers=max_workers)
        self.max_workers = max_workers
        self.futures: dict[Future, Task] = {}  # Future -> Task mapping

    def __repr__(self) -> str:
        return f'<{self.pool_type.title()}Pool max_workers={self.max_workers} n_working={self.n_working} [{self.is_active}]>'

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
        try:
            future = self.pool.submit(task.run)
        except Exception as e:
            task.status = 'failed'
            task.error = e
            error_logger.error(f'Error submitting task {task}: {e}')
            error_logger.exception(e)
            return None
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
                 results_db_path: str=f'results/results_{int(time.time())}.lmdb'):
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
            task_logger.info(f'Registering op class {op_cls}, run_mode {op_cls.run_mode}')
            # Convert to frozensets for safe internal operations
            frozen_output_types = frozenset(op_cls.output_types)
            # Handle input_types - could be set or dict (contracts)
            if isinstance(op_cls.input_types, set):
                frozen_input_types = frozenset(op_cls.input_types)
                input_types_to_register = frozen_input_types
            else:
                # For contract format, collect all input types from all contracts
                all_input_types: set[str] = set()
                for input_tuple in op_cls.input_types.keys():
                    all_input_types.update(input_tuple)
                input_types_to_register = frozenset(all_input_types)
            for output_type in frozen_output_types:
                self.ops_by_output_type[output_type].append(op_cls)
            for input_type in input_types_to_register:
                self.ops_by_input_type[input_type].append(op_cls)
        # this contains a list of op instances
        self.all_ops: list[Op] = []
        self._results: dict[str, Result] = {}  # key -> Result
        self._results_by_type: dict[str, list[str]] = defaultdict(list)  # output_type -> list of keys
        self.tasks: list[Task] = []
        self.done_tasks: list[Task] = []
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
        counts = lambda field: Counter(getattr(t, field) for t in self.tasks).most_common()
        task_logger.info(f'Added task {task}, {len(self.tasks)} tasks: {counts("status")}, {counts("name")}')

    def add_all_variants(self, op_cls: type[Op], inputs: dict[str, Result]) -> None:
        """Adds tasks for all variants of the given `op_cls` with the given `inputs`."""
        # convert inputs to the actual output from the previous step, not Result objects
        _inputs = {}
        for k, v in inputs.items():
            if isinstance(v, list):
                _inputs[k] = [item.output if isinstance(item, Result) else item for item in v]
            else:
                _inputs[k] = v.output if isinstance(v, Result) else v
        variants = op_cls.get_variants(_inputs)
        if variants is None:
            variants = {'': {}}
        for name, kwargs in variants.items():
            task = Task(op_cls=op_cls, inputs=inputs, variant_name=name, variant_kwargs=kwargs)
            self.add_task(task)

    def create_tasks(self, result: Result|Exception) -> None:
        """Create new work tasks to do based on a new result."""
        if isinstance(result, Exception) or result is None or not result.is_success:
            return
        # Get consumers for all output types of this result
        consumers = set()
        for output_type in result.op.output_types:
            consumers.update(self.get_consumers(output_type))
        task_logger.info(f'Creating new tasks from {result}: consumers={[c.__name__ for c in consumers]}')
        for consumer in consumers:
            self._create_tasks_for_consumer(consumer, result)

    def _create_tasks_for_consumer(self, consumer: type[Op], new_result: Result) -> None:
        """Create tasks for a specific `consumer` op, respecting input contracts.

        A contract is a dict mapping input type tuples to consistency requirements. E.g.:
          ("input_a", "input_b"): {"consistency_fields": ["field1"]}

        For that contract, it requires that the inputs be of type "input_a" and "input_b", and that
        they contain field "field1" and it be consistent across both inputs. This is often a variant
        name.
        """
        contracts = consumer.get_input_contracts()
        for input_tuple, contract in contracts.items():
            task_logger.debug(f'Checking contract {input_tuple}, {contract} for {consumer}')
            # Check if this contract can use the new result
            if not any(output_type in input_tuple for output_type in new_result.op.output_types):
                continue
            # Generate lists of results for each input type in this contract
            input_results = []
            for input_type in input_tuple:
                keys = self._results_by_type.get(input_type, [])
                results = [self._results[key] for key in keys if self._results[key].is_success]
                input_results.append(results)
            if not input_results or any(not results for results in input_results):
                continue
            # Generate cartesian product of tasks for this contract and add variants
            for cur_results in product(*input_results):
                # Count occurrences of each type in the input tuple to determine how to handle them
                type_counts = Counter(input_tuple)
                # Generate dict of inputs, properly handling multiple inputs of the same type
                inputs = {}
                type_indices = defaultdict(int)
                for t, r in zip(input_tuple, cur_results):
                    if type_counts[t] > 1: # appears multiple times, so it should be a list
                        if t not in inputs:
                            inputs[t] = [None] * type_counts[t]
                        inputs[t][type_indices[t]] = r
                        type_indices[t] += 1
                    else: # appears only once
                        inputs[t] = r
                if self._satisfies_contract(inputs, contract):
                    self.add_all_variants(consumer, inputs)

    def _satisfies_contract(self, inputs: dict[str, Result], contract: dict[str, Any]) -> bool:
        """Check if `inputs` satisfy the `contract` requirements."""
        consistency_fields = contract.get("consistency_fields", [])
        seen = set()
        # first make sure we're not using the same result twice
        for result in inputs.values():
            if isinstance(result, list): # Handle list of results (for same input_type)
                if len(set(result)) != len(result):
                    task_logger.info(f'Got duplicate results: {result} for {inputs}')
                    return False
                seen.update(result)
            else: # single result for input_type
                if result in seen:
                    return False
                seen.add(result)
        # now check for consistency fields
        for field in consistency_fields:
            values = set()
            for result in inputs.values():
                if isinstance(result, list): # Handle list of results
                    for r in result:
                        if value := r.output.get(field):
                            values.add(value)
                else: # Handle single result
                    if value := result.output.get(field):
                        values.add(value)
            if len(values) > 1:  # Inconsistent values found
                return False
        return True

    def run_tasks(self) -> None:
        """Runs our task list using our thread and process pools."""
        task_logger.info(f'Starting task run with {len(self.tasks)} initial tasks')
        t_pool, p_pool = self.thread_pool, self.proc_pool
        pools = dict(thread=t_pool, process=p_pool)
        matching_tasks = lambda *modes: [t for t in self.tasks if t.run_mode in modes and t.status == "pending"]

        while self.tasks or t_pool.is_active or p_pool.is_active:
            # submit pool tasks if they have capacity
            for task in matching_tasks("thread", "process"):
                task_logger.info(f'Submitting task {task} with run mode {task.run_mode} to {pools[task.run_mode]}')
                if pools[task.run_mode].submit_task_if_free(task):
                    self.tasks.remove(task)
            # if we still have pending tasks, run one in main thread
            for_main = matching_tasks("main")
            if for_main:
                task = for_main[0]
                self.tasks.remove(task)
                task.run()
                self.finish_task(task)
            # check for completed async tasks
            for pool in pools.values():
                for task in pool.get_completed_tasks():
                    self.finish_task(task)
            # avoid busy loop
            time.sleep(0.1)
        assert not self.tasks, f'Leftover tasks: {self.tasks}'
        task_logger.info(f'All {len(self.done_tasks)} tasks completed, statuses: {Counter(t.status for t in self.done_tasks)}')

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

        This creates a `Result` object with full provenance and stores it in our results index. It
        also logs the result to our database.

        Returns the created `Result` object.
        """
        assert task.status in ('completed', 'failed'), f'Task {task} not done yet'
        op = task.op
        assert op is not None, f'Task {task} has no op'

        # Collect all previous results from inputs
        provenance = []
        for input_type, input_value in task.inputs.items():
            if isinstance(input_value, list):
                # Handle list of results
                for item in input_value:
                    if isinstance(item, Result):
                        provenance.append(item)
            elif isinstance(input_value, Result):
                provenance.append(input_value)

        # Create the result with the provenance chain of previous results
        result = Result(
            output=task.output,
            op=op,
            provenance=provenance,
            key=op.key,
            variant=op.variant,
            error=task.error,
            timestamps=task.timestamps,
        )
        result_logger.info(f'From {task} -> {result}: {result.provenance_str}')
        result_logger.info(pformat(make_jsonable(result.to_dict())))
        om = OpManager.get()
        # Store result by key
        if op.key in om._results:
            error_logger.warning(f'Duplicate result key {op.key} for {op}, previous: {om._results[op.key]}')
            return None
        om._results[op.key] = result
        # Store key under all output types
        for output_type in op.output_types:
            if op.key not in om._results_by_type.setdefault(output_type, []):
                om._results_by_type[output_type].append(op.key)
        om.log_result(result)
        mem = sum(r.mem for r in om._results.values())
        perf_logger.info(f'Logged result {result}, {len(om._results)} total, mem: {mem}B')
        return result

    def log_result(self, result: Result) -> None:
        """Logs the given `result` to our database, as well as general logging updates."""
        if self.results_db is None:
            return
        if isinstance(result, Exception):
            return
        all_results = list(self._results.values())
        status_key = 'status:success' if result.is_success else 'status:failed'
        to_update = {
            # full result for this key
            f'key:{result.key}': make_jsonable(result.to_dict()),
            # all keys of each output type
            **{f'type:{output_type}': sorted(self._results_by_type[output_type])
               for output_type in result.op.output_types},
            # all keys of this op class
            f'op_name:{result.op.name}': sorted({r.key for r in all_results if isinstance(r.op, result.op.__class__)}),
            # all keys with this status
            status_key: sorted({r.key for r in all_results if r.is_success == result.is_success}),
            # all keys sorted by various criteria
            'by:start_time': [(r.key, r.start_ts) for r in sorted(all_results, key=lambda r: r.start_ts)],
            'by:end_time': [(r.key, r.end_ts) for r in sorted(all_results, key=lambda r: r.end_ts)],
            # current update time
            'last_update': time.time(),
        }
        perf_logger.debug(f'Logged results: {to_update.keys()}')
        self.results_db.update(to_update)
        self.results_db.sync()

    def results_to_dict(self) -> dict[str, Any]:
        """Serialize all our results for logging/reporting.

        This calls `make_jsonable(result.to_dict())` on each Result.
        Returns a dict with keys:
        - results: mapping key -> result dict
        - result_keys_by_type: mapping output_type -> list of keys
        """
        ret = dict(
            results={key: make_jsonable(r.to_dict()) for key, r in self._results.items()},
            result_keys_by_type=self._results_by_type,
        )
        return ret


class Op(ABC):
    """Base operation with input/output type definitions.

    Each operation defines:
    - `name`: unique identifier for this operation
    - `input_types`: set of type names this operation requires as input, OR dict of input contracts
    - `output_types`: set of type names this operation produces (it's a single output, but it can be
      "known" by multiple types).
    - `is_intermediate`: if True, the output is not stored in results

    For `input_types`, you can use either:
    1. Simple set format: {"feature_set", "labels"} - for basic cases
    2. Contract format: {("input_a", "input_b"): {"consistency_fields": ["field1"]}} - for complex cases

    These are by default defined at the class level, but if they are dependent on init parameters,
    you can also define them at the instance level. They are always referred to be `self.name`
    (e.g.), so python method resolution order will find the instance var first if it exists, then
    the class var.

    When created, the operation automatically registers itself with the global manager.
    """
    name = ""
    input_types: set[str] | dict[tuple[str, ...], dict[str, Any]] = set()
    output_types: set[str] = set()

    # by default we assume ops are not intermediate. If they are, override this
    is_intermediate = False

    # you can set this in your class to disable the op (e.g. for debugging)
    enabled = True

    # where this op should run (main=main process, thread=thread pool, process=process pool)
    run_mode = "main" # main, thread, process

    def __init__(self, variant: str|None=None, name: str|None = None, input_types: set[str] | dict[tuple[str, ...], dict[str, Any]] | None=None, output_types: set[str]|None=None):
        """Initialize this operation.

        - variant: optional variant name for this instance
        - name: optional instance-specific name
        - input_types: optional instance-specific input types (set or contract dict)
        - output_types: optional instance-specific output types

        If you don't specify these, the class-level ones are used. For `name`, if both are missing,
        the name of the python class is assigned to the class variable. For `output_types`, we
        require that either the class-level var or the arg to this func is non-empty.
        """
        self.variant = variant
        if name is not None:
            self.name = name
        if input_types is not None:
            self.input_types = input_types
        if output_types is not None:
            self.output_types = output_types
        # make sure we have a name of some sort
        if not name and not self.__class__.name:
            self.__class__.name = self.__class__.__name__
        assert self.output_types, f"{self} must have a non-empty output_types"
        assert self.enabled, f"{self} is disabled!"
        assert self.run_mode in ("main", "thread", "process"), f"{self} has invalid run_mode {self.run_mode}"
        OpManager.register(self)
        self.key = ''

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

        Returns `(output, error)` where the output is of types `self.output_types` and the error
        is the exception if any occurred, else None.
        """
        task_logger.debug(f'Going to execute op {self.name} with inputs: {inputs}')
        self.key = self.get_key(inputs)
        try:
            # try to get the underlying output from the result objects
            op_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, list):
                    # Handle multiple inputs of the same type
                    op_inputs[k] = [item.output for item in v]
                else:
                    op_inputs[k] = v.output
        except Exception:
            op_inputs = inputs
        task_logger.info(f'Executing {self} ({self.variant}) -> {self.key}')
        output, error = None, None
        try:
            with PerfTracker.track(op_inputs=op_inputs) as tracker:
                output = self._execute(op_inputs)
            perf_logger.info(f'Op {self} finished, perf: {tracker.stats()}')
        except Exception as e:
            error_logger.error(f'Op {self} failed with {type(e)}: {e}')
            # print stack trace
            error_logger.exception(e)
            error = e
        return output, error

    @classmethod
    def get_input_contracts(cls) -> dict[tuple[str, ...], dict[str, Any]]:
        """Convert input_types to contract format for consistent processing."""
        if isinstance(cls.input_types, set):
            # Convert simple set to contract format
            return {tuple(sorted(cls.input_types)): {}}
        elif isinstance(cls.input_types, dict):
            # Already in contract format
            return cls.input_types
        else:
            raise ValueError(f"Invalid input_types format: {type(cls.input_types)}")

    @abstractmethod
    def _execute(self, inputs: dict[str, Any]) -> Any:
        """Execute the operation with given inputs.

        - inputs: dict mapping type names to actual data

        Returns the output of types `self.output_types`.
        """
        raise NotImplementedError()

    def old_get_key(self, inputs: dict[str, Result]) -> str:
        """Generate key based on operation and inputs and full provenance.

        This creates a hash of the operation name and input data (including provenance) to enable
        caching of results and avoiding duplicate work.
        """
        # Create a deterministic string representation of inputs
        input_items = []
        for key in sorted(inputs.keys()):
            value = inputs[key]
            if isinstance(value, Result):
                # use the provenance chain and key for Result objects
                prov_str = "->".join(op.name for op in value.provenance)
            else:
                # For complex objects, use their string representation
                if hasattr(value, '__dict__'):
                    value_str = str(sorted(value.__dict__.items()))
                else:
                    value_str = str(value)
                input_items.append(f"{key}:{value_str}")
        input_str = "|".join(input_items)
        combined = f"{self.name}_{self.variant}_{input_str}"
        # Hash to keep keys manageable
        return hashlib.md5(combined.encode()).hexdigest()

    def get_key(self, inputs: dict[str, Result]) -> str:
        """Generate key based on operation and inputs (including full provenance)."""
        dict_v = lambda d: '&'.join(sorted(f'{k}={value_key(v)}' for k, v in sorted((str(k1), v1) for k1, v1 in d.items())))
        def value_key(v: Any) -> str:
            if isinstance(v, Result): # all keys in the provenance chain
                return ','.join(sorted(v.all_keys))+':'+value_key(v.op)+':'+value_key(v.output)
            elif isinstance(v, (list, tuple)): # Handle lists/tuples of things
                return ','.join(sorted(value_key(item) for item in v))
            elif isinstance(v, Op): # for objects, use sorted items
                return dict_v(v.__dict__)
            elif isinstance(v, dict): # for dicts, use sorted items
                return dict_v(v)
            else: # simple string representation
                return str(v)

        input_str = dict_v(inputs)
        combined = f"{self.name}_{self.variant}_{input_str}"
        task_logger.info(f'{self} got key str: {combined}')
        # Hash to keep keys manageable
        return hashlib.md5(combined.encode()).hexdigest()
