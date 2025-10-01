"""Time-related utilities and constants"""

from __future__ import annotations

import datetime
import inspect
import time

from contextlib import contextmanager

import numpy as np
import psutil
from scipy.optimize import curve_fit

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
    # Static variable to store stats across multiple runs
    _all_stats = {}

    def __init__(self, **kw):
        self.kw = kw
        self.stats_data = {
            "start_time": None,
            "end_time": None,
            "start_memory": None,
            "end_memory": None,
            "peak_memory": None,
            "matrix_shapes": {},
            "sequence_lengths": {},
            "input_sizes": {}
        }
        self.process = psutil.Process()

        # Get the calling frame to identify the code block
        frame = inspect.currentframe().f_back
        self.block_id = f"{frame.f_code.co_filename}:{frame.f_lineno}"

    def __enter__(self):
        self.stats_data["start_time"] = time.time()
        self.stats_data["start_memory"] = self.process.memory_info().rss
        self.stats_data["peak_memory"] = self.stats_data["start_memory"]

        # Analyze all input arguments
        for name, arg in self.kw.items():
            self._analyze_object(arg, name)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stats_data["end_time"] = time.time()
        self.stats_data["end_memory"] = self.process.memory_info().rss

        # Store this run's stats in the class variable
        if self.block_id not in self._all_stats:
            self._all_stats[self.block_id] = []
        self._all_stats[self.block_id].append(self.stats())

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
    def track(**kw):
        return PerfTracker(**kw)

    @classmethod
    def get_all_stats(cls, block_id=None):
        """Get stats for all runs or for a specific block_id"""
        if block_id:
            return cls._all_stats.get(block_id, [])
        return cls._all_stats

    @classmethod
    def clear_stats(cls, block_id=None):
        """Clear stats for all runs or for a specific block_id"""
        if block_id:
            if block_id in cls._all_stats:
                del cls._all_stats[block_id]
        else:
            cls._all_stats.clear()
            
    @classmethod
    def determine_complexity(cls, sizes, times):
        """Determine the likely complexity class based on fitting different models.
        
        Args:
            sizes: List of input sizes
            times: List of corresponding execution times
            
        Returns:
            A tuple of (complexity_class, score, params) where:
            - complexity_class is a string like "O(n)", "O(n²)", etc.
            - score is the mean squared error of the fit
            - params are the fitted parameters
        """
        import numpy as np
        from scipy.optimize import curve_fit
        
        # Convert to numpy arrays
        sizes = np.array(sizes, dtype=float)
        times = np.array(times, dtype=float)
        
        # Define models for different complexity classes
        models = {
            "O(1)": lambda x, a: a * np.ones_like(x),
            "O(log n)": lambda x, a, b: a * np.log(x) + b,
            "O(n)": lambda x, a, b: a * x + b,
            "O(n log n)": lambda x, a, b: a * x * np.log(x) + b,
            "O(n²)": lambda x, a, b: a * x**2 + b,
            "O(n³)": lambda x, a, b: a * x**3 + b,
            "O(2^n)": lambda x, a, b, c: a * np.power(b, x) + c
        }
        
        best_model = None
        best_score = float('inf')
        best_params = None
        
        for name, model_func in models.items():
            try:
                # For O(1), we need special handling
                if name == "O(1)":
                    params, cov = curve_fit(model_func, sizes, times)
                    predictions = model_func(sizes, *params)
                else:
                    # Skip very small sizes for log models to avoid log(0)
                    if "log" in name and any(sizes <= 0):
                        valid_indices = sizes > 0
                        if sum(valid_indices) < 3:  # Need at least 3 points
                            continue
                        params, cov = curve_fit(model_func, sizes[valid_indices], times[valid_indices])
                        predictions = model_func(sizes[valid_indices], *params)
                        times_subset = times[valid_indices]
                    else:
                        params, cov = curve_fit(model_func, sizes, times)
                        predictions = model_func(sizes, *params)
                        times_subset = times
                
                # Calculate mean squared error
                score = np.mean((predictions - times_subset)**2)
                
                if score < best_score:
                    best_score = score
                    best_model = name
                    best_params = params
            except Exception as e:
                # Skip models that fail to fit
                continue
        
        return (best_model, best_score, best_params) if best_model else ("Unknown", float('inf'), None)
    
    @classmethod
    def analyze_complexity(cls, block_id=None, metric="time_taken"):
        """Analyze the complexity of a code block based on multiple runs.
        
        Args:
            block_id: The specific block to analyze, or None for all blocks
            metric: Which metric to analyze ("time_taken" or "memory_used")
            
        Returns:
            A dictionary mapping block_ids to complexity analysis results
        """
        results = {}
        blocks = [block_id] if block_id else cls._all_stats.keys()
        
        for block in blocks:
            if block not in cls._all_stats or len(cls._all_stats[block]) < 3:
                continue  # Need at least 3 data points for meaningful analysis
                
            # Extract input sizes and metrics
            runs = cls._all_stats[block]
            
            # Group by input size to handle multiple runs of the same size
            size_to_metrics = {}
            for run in runs:
                # Determine the input size from the largest dimension or sequence
                input_size = 0
                for shape in run["matrix_shapes"].values():
                    if isinstance(shape, tuple):
                        input_size = max(input_size, max(shape))
                for length in run["sequence_lengths"].values():
                    input_size = max(input_size, length)
                
                if input_size not in size_to_metrics:
                    size_to_metrics[input_size] = []
                size_to_metrics[input_size].append(run[metric])
            
            # Average metrics for each size
            sizes = []
            metrics = []
            for size, values in sorted(size_to_metrics.items()):
                sizes.append(size)
                metrics.append(sum(values) / len(values))
            
            if len(sizes) < 3:
                continue  # Need at least 3 different sizes
                
            # Determine complexity
            complexity, score, params = cls.determine_complexity(sizes, metrics)
            
            results[block] = {
                "complexity": complexity,
                "score": score,
                "params": params.tolist() if params is not None else None,
                "data_points": list(zip(sizes, metrics)),
                "metric": metric
            }
            
        return results

    @staticmethod
    def example():
        matrix = np.random.rand(1000, 300)
        list_data = [1, 2, 3, 4, 5]
        dict_data = {"key1": matrix, "key2": list_data}
        with PerfTracker.track(matrix=matrix, list_data=list_data, dict_data=dict_data) as tracker:
            # Simulate some computation
            time.sleep(0.5)
            result = np.dot(matrix, matrix.T)
        print(tracker.stats())
        
    @staticmethod
    def complexity_example():
        """Example showing how to use PerfTracker to analyze algorithm complexity"""
        # Clear previous stats
        PerfTracker.clear_stats()
        
        # Run with different input sizes
        sizes = [100, 500, 1000, 2000, 5000]
        for size in sizes:
            matrix = np.random.rand(size, size)
            with PerfTracker.track(matrix=matrix) as tracker:
                # O(n²) operation
                result = np.dot(matrix, matrix.T)
                
        # Analyze complexity
        complexity_results = PerfTracker.analyze_complexity()
        for block_id, result in complexity_results.items():
            print(f"Block {block_id}:")
            print(f"  Complexity: {result['complexity']}")
            print(f"  Score: {result['score']}")
            print(f"  Data points: {result['data_points']}")
