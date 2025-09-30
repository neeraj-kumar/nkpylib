"""Time-related utilities and constants"""

import datetime
import time

from contextlib import contextmanager

import numpy as np
import psutil

DAY_SECS = 24*3600
YEAR_SECS = 365*DAY_SECS

def parse_ts(ts: float|int|str) -> float:
    """Parses the timestamp to a float"""
    if isinstance(ts, str):
        # strip Z
        if ts.endswith('Z'):
            ts = ts[:-1]
        # remove fractional part for adding later
        tz: int|str
        frac: float|str
        if '.' in ts:
            ts, frac = ts.split('.', 1)
            frac = float(f'0.{frac}')
        else:
            frac = 0.0
        # figure out how to deal with timezones
        if '+' in ts:
            ts, tz = ts.split('+', 1)
            tz = int(tz)
        ret = datetime.datetime.fromisoformat(ts).timestamp()
        return ret + frac
    return ts


class PerfTracker:
    """A class to track performance metrics like time and memory usage of a code block"""
    def __init__(self, input_dict):
        self.input_dict = input_dict
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.matrix_shapes = {}

    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss
        self._analyze_dict(self.input_dict)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.end_memory = psutil.Process().memory_info().rss

    def _analyze_dict(self, d, prefix=""):
        """Analyze dictionary contents, especially matrices"""
        for key, value in d.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, np.ndarray):
                self.matrix_shapes[path] = value.shape
            elif isinstance(value, dict):
                self._analyze_dict(value, path)
            # Add more types as needed

    def stats(self):
        """Return collected statistics"""
        return {
            "time_taken": self.end_time - self.start_time,
            "memory_used": (self.end_memory - self.start_memory) / (1024 * 1024),  # MB
            "matrix_shapes": self.matrix_shapes
        }

    @staticmethod
    def track(input_dict):
        return PerfTracker(input_dict)

    @staticmethod
    def example():
        input_dict = {"matrix1": np.random.rand(1000, 1000)}
        with PerfTracker.track(input_dict) as tracker:
            # Your code here
            result = some_computation(input_dict)
        print(tracker.stats())
