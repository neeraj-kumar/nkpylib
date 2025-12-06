"""A script to help analyze results"""

from __future__ import annotations

import json
import logging
import os
import tempfile

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from os.path import dirname, abspath, join
from subprocess import run
from typing import Any

from objexplore import explore # type: ignore
from tqdm import tqdm

from nkpylib.ml.feature_set import JsonLmdb

logger = logging.getLogger(__name__)

RESULTS_DIR = 'results/'

def get_matching_result_path(key='-1', dir=RESULTS_DIR) -> str:
    """Get the path to the matching results for `key` from a JsonLMDB database.

    The key can be:
    - '' or '-1': the latest results file
    - a substring to match in the results filenames (latest match)
    - an index (pos or neg) to select from the sorted list of results files (as a string)
    """
    result_files = sorted(glob(join(dir, 'results_*.lmdb')))
    if not result_files:
        raise FileNotFoundError('No results files found in results directory')
    # first try as int
    try:
        key = int(key)
        return result_files[key]
    except ValueError:
        pass
    # now look for substrings
    matching_files = [f for f in result_files if key in os.path.basename(f)]
    if matching_files:
        return matching_files[-1]
    raise ValueError(f'No results file matching key "{key}" found in {dir}')


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


def make_output(rd: dict[str, Any], db_path: str) -> dict[str, Any]:
    db = JsonLmdb.open(db_path)
    return {**rd, 'analysis': build_provenance_analysis(rd, db=db)}

def explore_results(db: JsonLmdb, **kw):
    """Uses visidata to explore the results database."""
    # get result keys ordered by end time
    by_end = [f'key:{k}' for (k, _) in db['by:end_time']]
    # get all results, filtered down
    results = {k: db[k] for k in by_end}
    # print all 'warnings' from analysis objects
    warnings = []
    for key, r in results.items():
        analysis = r.get('analysis', {})
        if isinstance(analysis, str):
            try:
                analysis = json.loads(analysis)
            except Exception:
                continue
        cur = analysis.get('warnings', [])
        output = r.get('output', {})
        provenance_str = r.get('provenance_str', '')
        for w in cur:
            #print(f"Result {key} warning: {w}")
            warnings.append(dict(result_key=key, provenance_str=provenance_str, **w, output=output))
    # sort warnings first by label_key then by unit then by descending score then by descending
    # value
    warnings.sort(key=lambda w: (w.get('label_key', ''), w.get('unit', ''), -w.get('score', 0), -w.get('value', 0)))
    # get full provenance analysis
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = []
        for k, result_data in (results.items()):
            if result_data['op_name'] != 'run_prediction':
                continue
            #results[k] = {**result_data, 'analysis': build_provenance_analysis(result_data, db=db)}
            futures.append((k, pool.submit(make_output, result_data, db_path=db.env.path())))
            if len(futures) > 1000:
                break
        for k, future in tqdm(futures):
            results[k] = future.result()
    logger.info(f'Exploring {len(results)} results')
    #explore(results, **kw)
    # write to a tempfile then run visidata on it
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as f:
        f.write(json.dumps({'key': 'warnings', 'instance_vars': warnings}) + '\n')
        for k, v in results.items():
            f.write(json.dumps({'key': k, **v}) + '\n')
        temp_path = f.name
    logger.info(f'Wrote results to {temp_path} of size {os.path.getsize(temp_path)/1024/1024} MB')
    # wait for user to press enter to continue
    #input('Press Enter to launch visidata...')
    run(['visidata', temp_path])
    # cleanup
    os.remove(temp_path)

def redo_analysis(db: JsonLmdb, **kw):
    """Redoes the analysis on the given results database."""
    raise NotImplementedError('Redo analysis not implemented yet')
    by_end = [f'key:{k}' for (k, _) in db['by:end_time']]
    for full_key in by_end:
        result_data = db[full_key]
        logger.info(f'Redoing analysis for result {full_key}')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(name)s\t%(funcName)s\t%(message)s', level=logging.INFO)
    funcs = {f.__name__: f for f in [explore_results]}
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('func', type=str, choices=funcs, help=f'Function to run [choices: {", ".join(funcs)}]')
    parser.add_argument('-k', '--key', type=str, default='-1', help='Key to match results database [default latest]')
    parser.add_argument('--result_path', type=str, help='Path to the results JsonLMDB database [default latest]')
    parser.add_argument('--starting_inputs', type=str, nargs='*', default=['argparse'], help='Starting input types for simulation')
    args = parser.parse_args()

    if args.func in ['explore_results']:
        if not args.result_path:
            args.result_path = get_matching_result_path(args.key)
        db = JsonLmdb.open(args.result_path)
        kw = vars(args)
        kw.pop('result_path')
        kw.pop('starting_inputs')
        kw.pop('key')
        func = kw.pop('func')
        funcs[func](db=db, **kw) # type: ignore
    else:
        func = args.func
        funcs[func]()
