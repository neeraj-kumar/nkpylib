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
    def __init__(self, *args):
        self.args = args
        self.stats_data = {
            "start_time": None,
            "end_time": None,
            "start_memory": None,
            "end_memory": None,
            "peak_memory": None,
            "matrix_shapes": {},
            "sequence_lengths": {}
        }
        self.process = psutil.Process()

    def __enter__(self):
        self.stats_data["start_time"] = time.time()
        self.stats_data["start_memory"] = self.process.memory_info().rss
        self.stats_data["peak_memory"] = self.stats_data["start_memory"]
        # Analyze all input arguments
        for i, arg in enumerate(self.args):
            self._analyze_object(arg, f"arg{i}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stats_data["end_time"] = time.time()
        self.stats_data["end_memory"] = self.process.memory_info().rss

    def _update_peak_memory(self):
        """Update peak memory usage"""
        current = self.process.memory_info().rss
        if current > self.stats_data["peak_memory"]:
            self.stats_data["peak_memory"] = current

    def _analyze_object(self, obj, path=""):
        """Analyze any object for size and shape information"""
        if isinstance(obj, np.ndarray):
            self.stats_data["matrix_shapes"][path] = obj.shape
        elif isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                self._analyze_object(value, new_path)
        elif isinstance(obj, (list, tuple, set, frozenset, str, bytes, bytearray)):
            self.stats_data["sequence_lengths"][path] = len(obj)
            # For sequences that aren't strings or bytes, analyze their contents too
            if isinstance(obj, (list, tuple)) and obj and not isinstance(obj[0], (str, bytes, int, float, bool)):
                for i, item in enumerate(obj):
                    self._analyze_object(item, f"{path}[{i}]")
        # Check memory after analyzing each object
        self._update_peak_memory()

    def stats(self):
        """Return collected statistics"""
        return {
            "time_taken": self.stats_data["end_time"] - self.stats_data["start_time"],
            "memory_used": (self.stats_data["end_memory"] - self.stats_data["start_memory"]) / (1024 * 1024),  # MB
            "peak_memory": self.stats_data["peak_memory"] / (1024 * 1024),  # MB
            "matrix_shapes": self.stats_data["matrix_shapes"],
            "sequence_lengths": self.stats_data["sequence_lengths"],
            "raw_stats": self.stats_data
        }

    @staticmethod
    def track(*args):
        return PerfTracker(*args)

    @staticmethod
    def example():
        matrix = np.random.rand(1000, 300)
        list_data = [1, 2, 3, 4, 5]
        dict_data = {"key1": matrix, "key2": list_data}
        with PerfTracker.track(matrix, list_data, dict_data) as tracker:
            # Simulate some computation
            time.sleep(0.5)
            result = np.dot(matrix, matrix)
        print(tracker.stats())
