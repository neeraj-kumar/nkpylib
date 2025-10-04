"""A script to help analyze results"""

from __future__ import annotations

import json
import logging
import os
import tempfile

from argparse import ArgumentParser
from glob import glob
from os.path import dirname, abspath, join
from subprocess import run
from typing import Any

from objexplore import explore # type: ignore

from nkpylib.ml.feature_set import JsonLmdb

logger = logging.getLogger(__name__)

RESULTS_DIR = 'results/'

def get_latest_result_path(dir=RESULTS_DIR) -> str:
    """Get the path to the latest results JsonLMDB database."""
    result_files = sorted(glob(join(dir, 'results_*.lmdb')))
    if not result_files:
        raise FileNotFoundError('No results files found in results directory')
    return result_files[-1]


def build_provenance_analysis(result_data: dict[str, Any], db: JsonLmdb):
    """Build analysis dict with full provenance chain."""
    provenance_keys = result_data.get('provenance', [])
    all_keys = provenance_keys + [result_data['key']]
    chain_results = []
    by_end = [f'key:{k}' for (k, _) in db['by:end_time']]
    for key in all_keys:
        full_key = f'key:{key}' if not key.startswith('key:') else key
        if full_key in db:
            chain_results.append((key, db[full_key]))
            if key not in by_end:
                by_end.append(key)
    chain_results.sort(key=lambda pair: by_end.index(pair[0]))
    ret = {}
    for key, result in chain_results:
        ret[key] = {
            'op': result.get('op_name', ''),
            'key': key,
            'analysis': result.get('analysis', {}),
            'variant': result.get('variant', ''),
            'instance_vars': result.get('instance_vars', {}),
        }
    return ret

def explore_results(db: JsonLmdb, **kw):
    """Uses visidata to explore the results database."""
    # get result keys ordered by end time
    by_end = [f'key:{k}' for (k, _) in db['by:end_time']]
    # get all results, filtered down
    results = {k: db[k] for k in by_end[:300]}
    # get full provenance analysis
    for k, result_data in results.items():
        results[k] = {**result_data, 'analysis': build_provenance_analysis(result_data, db=db)}
    logger.info(f'Exploring {len(results)} results')
    #explore(results, **kw)
    # write to a tempfile then run visidata on it
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as f:
        for k, v in results.items():
            f.write(json.dumps({'key': k, **v}) + '\n')
        temp_path = f.name
    logger.info(f'Wrote results to {temp_path} of size {os.path.getsize(temp_path)/1024/1024} MB')
    run(['visidata', temp_path])
    # cleanup
    os.remove(temp_path)


def analyze_op_dependencies():
    """Analyze which ops can run based on input/output type matching."""
    from nkpylib.ml.evaluator.evaluator_ops import find_subclasses, Op
    from collections import defaultdict
    
    # Get all enabled op classes
    op_classes = [cls for cls in find_subclasses(Op) if cls.enabled]
    
    # Build dependency graph
    producers = defaultdict(list)  # output_type -> [op_classes]
    consumers = defaultdict(list)  # input_type -> [op_classes]
    
    for op_cls in op_classes:
        # Handle output types
        for output_type in op_cls.output_types:
            producers[output_type].append(op_cls)
        
        # Handle input types (both set and contract formats)
        if isinstance(op_cls.input_types, set):
            for input_type in op_cls.input_types:
                consumers[input_type].append(op_cls)
        else:  # contract format
            for input_tuple in op_cls.input_types.keys():
                for input_type in input_tuple:
                    consumers[input_type].append(op_cls)
    
    return producers, consumers, op_classes

def print_dependency_analysis():
    """Print analysis of op dependencies."""
    producers, consumers, op_classes = analyze_op_dependencies()
    
    print("=== PRODUCERS (what each type is produced by) ===")
    for output_type, ops in sorted(producers.items()):
        print(f"{output_type}: {[op.name for op in ops]}")
    
    print("\n=== CONSUMERS (what each type is consumed by) ===")
    for input_type, ops in sorted(consumers.items()):
        print(f"{input_type}: {[op.name for op in ops]}")
    
    print("\n=== ORPHANED TYPES (produced but not consumed) ===")
    orphaned = set(producers.keys()) - set(consumers.keys())
    for output_type in sorted(orphaned):
        print(f"{output_type}: produced by {[op.name for op in producers[output_type]]}")
    
    print("\n=== MISSING TYPES (consumed but not produced) ===")
    missing = set(consumers.keys()) - set(producers.keys())
    for input_type in sorted(missing):
        print(f"{input_type}: needed by {[op.name for op in consumers[input_type]]}")

def simulate_execution_path(starting_inputs: set[str]):
    """Simulate which ops could run given starting input types."""
    
    producers, consumers, op_classes = analyze_op_dependencies()
    
    available_types = set(starting_inputs)
    runnable_ops = []
    
    # Keep adding ops until no more can run
    changed = True
    iteration = 0
    while changed:
        changed = False
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print(f"Available types: {sorted(available_types)}")
        
        for op_cls in op_classes:
            if op_cls in [op.__class__ for op in runnable_ops]:
                continue  # Already added
            
            # Check if this op can run
            can_run = False
            
            if isinstance(op_cls.input_types, set):
                # Simple set format - need all input types
                if op_cls.input_types.issubset(available_types):
                    can_run = True
            else:
                # Contract format - need at least one contract satisfied
                for input_tuple in op_cls.input_types.keys():
                    if set(input_tuple).issubset(available_types):
                        can_run = True
                        break
            
            if can_run:
                runnable_ops.append(op_cls)
                available_types.update(op_cls.output_types)
                changed = True
                print(f"  Can run: {op_cls.name} -> produces {sorted(op_cls.output_types)}")
    
    return runnable_ops

def analyze_contracts():
    """Analyze complex input contracts."""
    from nkpylib.ml.evaluator.evaluator_ops import find_subclasses, Op
    
    op_classes = [cls for cls in find_subclasses(Op) if cls.enabled]
    
    print("=== COMPLEX CONTRACTS ===")
    for op_cls in op_classes:
        if isinstance(op_cls.input_types, dict):
            print(f"\n{op_cls.name}:")
            for input_tuple, contract in op_cls.input_types.items():
                consistency = contract.get("consistency_fields", [])
                print(f"  {input_tuple} (consistency: {consistency})")

def full_dependency_analysis():
    """Complete analysis of op dependencies."""
    
    print("=== OP DEPENDENCY ANALYSIS ===\n")
    
    # Basic analysis
    print_dependency_analysis()
    
    print("\n" + "="*50)
    
    # Contract analysis  
    analyze_contracts()
    
    print("\n" + "="*50)
    
    # Execution simulation
    print("=== SIMULATION STARTING WITH argparse ===")
    runnable = simulate_execution_path({"argparse"})
    print(f"\nTotal runnable ops: {len(runnable)}")
    print("Runnable ops in order:", [op.name for op in runnable])

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(name)s\t%(funcName)s\t%(message)s', level=logging.INFO)
    funcs = {f.__name__: f for f in [explore_results, full_dependency_analysis, print_dependency_analysis, simulate_execution_path, analyze_contracts]}
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('func', type=str, choices=funcs, help=f'Function to run [choices: {", ".join(funcs)}]')
    parser.add_argument('--result_path', type=str, help='Path to the results JsonLMDB database [default latest]')
    parser.add_argument('--starting_inputs', type=str, nargs='*', default=['argparse'], help='Starting input types for simulation')
    args = parser.parse_args()
    
    if args.func in ['explore_results']:
        if not args.result_path:
            args.result_path = get_latest_result_path()
        db = JsonLmdb.open(args.result_path)
        kw = vars(args)
        kw.pop('result_path')
        kw.pop('starting_inputs')
        func = kw.pop('func')
        funcs[func](db=db, **kw) # type: ignore
    elif args.func == 'simulate_execution_path':
        simulate_execution_path(set(args.starting_inputs))
    else:
        func = args.func
        funcs[func]()
