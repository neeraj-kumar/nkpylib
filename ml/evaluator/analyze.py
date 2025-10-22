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
    results = {k: db[k] for k in by_end[:500]}
    # print all 'warnings' from analysis objects
    for key, r in results.items():
        analysis = r.get('analysis', {})
        warnings = analysis.get('warnings', [])
        for w in warnings:
            print(f"Result {key} warning: {w}")

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
    # wait for user to press enter to continue
    input('Press Enter to launch visidata...')
    run(['visidata', temp_path])
    # cleanup
    os.remove(temp_path)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(name)s\t%(funcName)s\t%(message)s', level=logging.INFO)
    funcs = {f.__name__: f for f in [explore_results]}
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
