#!/usr/bin/env python
"""Simple task/dependency management system, written by Neeraj Kumar.

Licensed under the 3-clause BSD License:

Copyright (c) 2021, Neeraj Kumar (http://neerajkumar.org)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the author nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL NEERAJ KUMAR BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import inspect
import logging
import threading
import time

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from enum import auto, Enum
from pprint import pformat
from subprocess import check_output
from typing import Dict, List, Tuple, Type


def abstract_class_attributes(*names):
    """Class decorator to add one or more abstract attributes

    From: https://stackoverflow.com/questions/45248243/
    """

    def _func(cls, *names):
        """Function that extends the __init_subclass__ method of a class."""

        # Add each attribute to the class with the value of NotImplemented
        for name in names:
            setattr(cls, name, NotImplemented)

        # Save the original __init_subclass__ implementation, then wrap
        # it with our new implementation.
        orig_init_subclass = cls.__init_subclass__

        def new_init_subclass(cls, **kwargs):
            """
            New definition of __init_subclass__ that checks that
            attributes are implemented.
            """

            # The default implementation of __init_subclass__ takes no
            # positional arguments, but a custom implementation does.
            # If the user has not reimplemented __init_subclass__ then
            # the first signature will fail and we try the second.
            try:
                orig_init_subclass(cls, **kwargs)
            except TypeError:
                orig_init_subclass(**kwargs)

            # Check that each attribute is defined.
            for name in names:
                if getattr(cls, name, NotImplemented) is NotImplemented:
                    raise NotImplementedError(f"You forgot to define {name}!!!")

        # Bind this new function to the __init_subclass__.
        # For reasons beyond the scope here, it we must manually
        # declare it as a classmethod because it is not done automatically
        # as it would be if declared in the standard way.
        cls.__init_subclass__ = classmethod(new_init_subclass)

        return cls

    return lambda cls: _func(cls, *names)


def get_subprocess_lines(*args, **kwargs):
    """Runs a subprocess, returning a list of non-empty lines"""
    return [
        line.strip()
        for line in check_output(*args, **kwargs).decode("utf-8", "ignore").split("\n")
        if line.strip()
    ]


class RunStyle(Enum):
    """What style of running does this need"""

    MAIN = auto()
    PROCESS = auto()


class Muxing(Enum):
    """An enum for different kinds of muxing"""

    ONE_TO_ONE = auto()
    ONE_TO_N = auto()
    N_TO_ONE = auto()
    N_TO_N = auto()


class Status(Enum):
    BLOCKED = auto()
    READY = auto()
    PROCESSING = auto()
    DONE = auto()
    FAILED = auto()


class Task:
    """Generic task definition"""

    RUN_STYLE = RunStyle.MAIN
    INPUTS = {}  # type: Dict[str, Tuple[Type[Task], Muxing]]
    DEPENDENCIES = []  # type: List[Type[Task]]
    LOG_LEVEL = logging.INFO

    def __init__(self, key, **kwargs):
        """You must specify a `key` that uniquely identifies this task.

        The key must be unique among its class only, not across all tasks.
        """
        self.key = key
        self.kwargs = kwargs
        self.status = Status.BLOCKED

    def __repr__(self):
        return f"{self.__class__.__name__}({self.key}: {self.status.name})"
        # return f"{self.__class__.__name__}({self.key}: {self.status.name}: {self.kwargs})"

    def __getitem__(self, key):
        """Returns the kwarg for the given `key`"""
        return self.kwargs[key]

    def __setitem__(self, key, value):
        """Sets the kwargs for the given `key` to `value`"""
        self.kwargs[key] = value

    def is_done(self):
        """Checks if this task is done already.

        By default it calls `self._load()` and checks if it's not `None`. This will work in most
        cases if you implement `_load()` properly, but it's not necessarily the most efficient.
        Override if you want different behavior, or something more efficient.

        If this returns an exception, that counts as not being done.
        """
        return self._load() is not None

    def _run(self):
        """Implementation for running this task. Override this."""
        pass

    def _load(self):
        """Loads cached results. Override if wanted."""
        pass

    def _save(self, *args, **kwargs):
        """Saves cached results. Override if wanted."""
        pass

    def run(self):
        """Wrapper to call _run if needed. Do not override"""
        try:
            done = self.is_done()
            if done:
                self.status = Status.DONE
                return self._load()
        except Exception:
            done = False
        if not done:
            try:
                ret = self._run()
                self._save(ret)
                self.status = Status.DONE
                return ret
            except Exception as e:
                logging.exception("%s failed", self)
                self.status = Status.FAILED


def run_task(task):
    output = task.run()
    return (output, task.status)


def get_all_concrete_subclasses(cls):
    """Recursively finds all subclasses of `cls` that are not abstract"""
    ret = []
    for subcls in cls.__subclasses__():
        if not inspect.isabstract(subcls):
            ret.append(subcls)
        ret.extend(get_all_concrete_subclasses(subcls))
    return ret


class TaskRunner:
    """A runner for tasks"""

    def __init__(self, starting_task, n_procs=10):
        """Initializes this task runner with given `starting_task`.

        This first generates the graph and topologically sorts it.
        """
        # build up a graph of tasks to then sort
        self.graph = {}
        self.children_by_task = defaultdict(list)
        self.run_loop_done = False
        for task_cls in get_all_concrete_subclasses(Task):
            self.graph[task_cls] = set(task_cls.DEPENDENCIES)
            if task_cls.INPUTS:
                for key, (input_cls, muxing) in task_cls.INPUTS.items():
                    self.graph[task_cls].add(input_cls)
                    self.children_by_task[input_cls].append((task_cls, muxing, key))
        # self.task_order = TopologicalSorter(self.graph).static_order()
        # assert self.task_order[0] == starting_task

        # create actual tasks
        starting_task.status = Status.READY
        self.tasks_by_key = defaultdict(dict)
        self.tasks_by_key[starting_task.key][starting_task.__class__] = starting_task
        self.futures = {}
        self.pool = ProcessPoolExecutor(n_procs)
        self.futures_thread = threading.Thread(target=self.process_futures)
        self.futures_thread.start()

    def process_futures(self, delay=1):
        logging.info("In process futures")
        while self.futures or not self.run_loop_done:
            logging.debug("Iterating through %d futures", len(self.futures))
            t0 = time.time()
            to_del = []
            for future, task in list(self.futures.items()):
                if future.done():
                    output, status = future.result()
                    task.status = status
                    self.process_output(output, task)
                    to_del.append(future)
            for future in to_del:
                del self.futures[future]
            remaining = delay - (time.time() - t0)
            if remaining > 0:
                time.sleep(remaining)

    def get_task_by_key(self, key, task_cls):
        """Returns a task by given `key` and `task_cls` if it exists, else None"""
        return self.tasks_by_key[key].get(task_cls, None)

    def __del__(self):
        self.pool.shutdown()

    def process_output(self, output, task):
        if task.status != Status.DONE:
            return
        logging.log(task.LOG_LEVEL, "Finished task %s, with output of type %s", task, type(output))
        for child_task_cls, muxing, child_key_field in self.children_by_task[task.__class__]:
            if muxing == Muxing.N_TO_ONE:
                # the output is a sequence, so generate tasks for each item in the seq
                for child_key in output:  # TODO limit outputs here
                    ct = self.get_task_by_key(child_key, child_task_cls)
                    if not ct:
                        kwargs = {child_key_field: child_key}
                        ct = child_task_cls(**kwargs)
                        self.tasks_by_key[child_key][child_task_cls] = ct
                    # check if the child task is ready to run
                    for dep_cls in ct.DEPENDENCIES:
                        dep_task = self.get_task_by_key(child_key, dep_cls)
                        if dep_task and dep_task.status != Status.DONE:
                            ct.status = Status.BLOCKED
                    else:
                        ct.status = Status.READY
            elif muxing == Muxing.ONE_TO_ONE:
                # the output directly maps to the next input
                child_key = output
                ct = self.get_task_by_key(child_key, child_task_cls)
                if not ct:
                    kwargs = {child_key_field: child_key}
                    ct = child_task_cls(**kwargs)
                    self.tasks_by_key[child_key][child_task_cls] = ct
                # check if the child task is ready to run
                for dep_cls in ct.DEPENDENCIES:
                    dep_task = self.get_task_by_key(child_key, dep_cls)
                    if dep_task and dep_task.status != Status.DONE:
                        ct.status = Status.BLOCKED
                else:
                    ct.status = Status.READY
            else:
                raise NotImplementedError("Don't know how to handle muxing %s" % (muxing))

    def _run_task(self, task):
        output, status = run_task(task)
        task.status = status
        self.process_output(output, task)

    def _task_ready(self, task):
        """Checks if a given task is ready to run"""
        return True

    def run(self):
        """Runs our tasks"""
        done = False
        while not done:
            for key, clses in list(self.tasks_by_key.items()):
                for cls, task in clses.items():
                    if task.status == Status.READY:
                        if task.RUN_STYLE == RunStyle.MAIN:
                            self._run_task(task)
                        elif task.RUN_STYLE == RunStyle.PROCESS:
                            logging.log(task.LOG_LEVEL, "Submitting future for %s", task)
                            future = self.pool.submit(run_task, task)
                            task.status = Status.PROCESSING
                            self.futures[future] = task
            done = True
            for key, clses in list(self.tasks_by_key.items()):
                for cls, task in clses.items():
                    if task.status != Status.DONE and task.status != Status.FAILED:
                        logging.debug("Task %s not done", task)
                        done = False
            time.sleep(0.1)
        logging.info("Exited run loop")
        self.run_loop_done = True
