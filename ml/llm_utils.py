"""Various higher-level utilities for working with LLMs."""

#TODO figure out how to match two sets of dicts of data using LLMs/etc
#TODO   see https://chatgpt.com/share/674361be-3528-8012-9c9b-a83859cdf170

from __future__ import annotations

import json
import logging
import re

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pprint import pformat
from typing import Any, Callable, Iterator, Sequence

import tiktoken

from bs4 import BeautifulSoup, Comment
from bs4.element import NavigableString

from nkpylib.ml.client import call_llm, chunked

logger = logging.getLogger(__name__)

def load_llm_json(s: str) -> Any:
    """"Tries to load a cleaned up version of JSON output from an LLM.

    This tries a few common things to clean up JSON and then load it.
    It might still fail, in which case, it will raise a `ValueError`.
    """
    # look for the first { or [
    delims = '{['
    endlims = '}]'
    starts = [(s.index(d), d) for d in delims if d in s]
    if not starts:
        raise ValueError("No JSON delimiters found")
    start, delim = min(starts)
    # look for the last matching delim
    end = s.rindex(endlims[delims.index(delim)])
    # extract the JSON
    json_str = s[start:end+1]
    # try to load it
    try:
        return json.loads(json_str)
    except Exception as e:
        raise ValueError(f"Could not load JSON from {json_str}: {e}")

CHUNKED_PROMPT = """%s

%d input items are enumerated below, one per line, prefixed by the item number. Return exactly one
output item per input item, in the same order and with the same item numbers. Return no other text.
If you have an error processing an input item, return the text "error" for that item (prefixed with
the item number).

\n\n%s"""

def llm_transform_list(base_prompt: str,
                       items: Sequence[str],
                       max_tokens:int =128000,
                       model:str ='llama3',
                       chunk_size:int = 10,
                       prompt_fmt: str=CHUNKED_PROMPT,
                       sys_prompt: str='',
                       **kw) -> Iterator[str|None]:
    """Transforms a list of items using an LLM.

    Often you have a list of items that you want to transform using an LLM. If the list is very
    long, you cannot process it in a single call (either due to hard context-length limitations, or
    due to quality issues as the model starts to forget some of the input instructions). In
    addition, there might be issues processing some of the inputs, but you don't want to throw
    everything away if there's an error in only some of the inputs.

    This utility function helps with that, by (1) enumerating items and requiring the same of the
    outputs, and (2) breaking the list into chunks. For each chunk, we call the LLM with:

        {base_prompt}\n\n{len} input items are enumerated below, one per line, prefixed by the item
        number. Return exactly one output item per input item, in the same order and with the same
        item numbers. Return no other text. If you have an error processing an input item, return
        the text "error" for that item (prefixed with the item number).\n

        1. <item 1>
        2. <item 2>
        ...

    Uses the 'llama3' model by default.

    Returns a list of outputs of the same length as the input. If we cannot parse a valid item
    number, then we return `None` for that input.

    Any additional keyword arguments are passed to the LLM call (call_llm.single_future()).
    """
    futures = []
    output_re = re.compile(r'^(\d+)\. (.*)$')
    for i, s in enumerate(items):
        assert '\n' not in s, f'Newline in input item {i+1}: {s}'
    for chunk in chunked(items, chunk_size):
        lst = ''.join(f'{i+1}. {item}\n' for i, item in enumerate(chunk))
        prompt = prompt_fmt % (base_prompt, len(lst), lst)
        prompts: list[tuple[str, str]] | str
        if sys_prompt:
            prompts = [('system', sys_prompt), ('user', prompt)]
        else:
            prompts = prompt
        f = call_llm.single_future(prompts, max_tokens=max_tokens, model=model, **kw)
        futures.append((chunk, f))
    for chunk, future in futures:
        try:
            llm_output = future.result()
        except Exception:
            logger.exception(f'Error processing chunk {chunk}')
            for _ in chunk:
                yield None
            continue
        logger.debug(f'for input {chunk} got output {llm_output}')
        lines = [output_re.match(l) for l in llm_output.split('\n')]
        # make a dict from output item number to output text
        out = {int(l.group(1)): l.group(2) for l in lines if l is not None}
        # now we can build the output list
        for i, input in enumerate(chunk):
            if i+1 in out:
                yield out[i+1]
            else:
                yield None


EncOrNameT = str|tiktoken.core.Encoding

def get_tiktoken_encoder(enc_or_model_name: EncOrNameT|None=None) -> tiktoken.core.Encoding:
    """Returns a tiktoken encoding object.

    You can either pass in a tiktoken encoding directly, or a model name that tiktoken knows how to
    get the encoding for. Or if you leave it as `None`, we use the encoding for the 'gpt-4o' model.
    """
    if enc_or_model_name is None:
        enc_or_model_name = 'gpt-4o'
    if isinstance(enc_or_model_name, str):
        return tiktoken.encoding_for_model(enc_or_model_name)
    else:
        return enc_or_model_name

def count_tokens(s: str, enc_or_model_name: EncOrNameT|None=None):
    """Counts the number of tokens in a string.

    You can either pass in a tiktoken encoding directly, or a model name that tiktoken knows how to
    get the encoding for. Or if you leave it as `None`, we use the encoding for the 'gpt-4o' model.
    """
    enc = get_tiktoken_encoder(enc_or_model_name)
    return len(enc.encode(s))

def show_tokenized_str(s: str,
                       enc_or_model_name: EncOrNameT|None=None,
                       dlm: bytes=b'|') -> bytes:
    """Shows you what the tokenized version of `s` looks like.

    You can either pass in a tiktoken encoding directly, or a model name that tiktoken knows how to
    get the encoding for. Or if you leave it as `None`, we use the encoding for the 'gpt-4o' model.

    You can also pass in a delimiter to use between tokens in the output (make sure it's bytes!)
    """
    enc = get_tiktoken_encoder(enc_or_model_name)
    return dlm.join([enc.decode_single_token_bytes(t) for t in enc.encode(s)])

def match_obj_schema(objs: list[dict],
                     schema: dict[str, Any],
                     allow_new_selects:bool=False,
                     **kw) -> Iterator[dict]:
    """Given `objs` and an `schema`, runs a query to map the objects.

    This was original written for Airtable, but it should work for any schema that has a similar
    structure. In airtable, you can get the schema for a given table by calling my `get_base_schema`
    function, which returns the schema for the entire base (all tables and views), and then picking
    the table you want to add the object to.

    The schema is a dict of field names to field data. The field data is a dict:
    - 'type': the field type (e.g. 'singleSelect', 'multipleSelects', etc)
    - 'description': the field description (optional)
    - 'options': the field options (optional, only for singleSelect and multipleSelects)

    This function will use the name, 'type', and 'description' (if present) fields in the schema
    to prompt the LLM. In addition, for single- or multi-select fields, we list the existing options.

    If `allow_new_selects` is `True`, we allow the LLM to generate new options for selects, else not.

    The `kw` are directly passed to `llm_transform_list()`.

    A practical tip: in airtable, add a description only to those fields that you want to map into
    (skipping primary keys, formulae, notes, etc), and then only include fields with descriptions in
    the target schema.
    """
    sys_prompt = f'''You are an intelligent data mapper. Given a target table
    schema and a list of objects, you try to map as many of each object's fields into the schema as possible.
    Output only the JSON object as described in the main prompt, no other text or explanations.'''

    prompt = f'''You will be given a target schema and a list of objects in JSON format. Your task
    is to map each object into the target schema. The target schema has field names, types, and
    descriptions (where present).

    If the field is of type singleSelect or multipleSelects, then you will also be given the current
    list of option values. In the former case, you can choose at most one of the options; in the
    latter case, you can choose as many of the options that match.

    For this run, you are {'' if allow_new_selects else 'NOT '}allowed to generate new options for
    single- or multi-select fields.

    The input object might have nested data structures, and sometimes the target schema might need a
    value that is nested. In other cases, a target value might need a simple combination of multiple
    input field values.

    You might not be able to map all fields of an object, or fill all target fields, that is fine.
    If you are unsure of a field mapping, it is better to skip it.

    For each input object, output a single JSON object with the following top-level fields:
    - 'target': The mapped target object, with field names as keys and the object's values as values.
    - 'unmapped': A list of field names from the object that you could not map to the target schema.
    - 'unfilled': A list of field names from the target schema that you could not fill from the object.

    TARGET SCHEMA:
    '''
    for field, data in schema.items():
        prompt += f"- {field} ({data['type']})"
        if 'description' in data:
            prompt += f': {data["description"]}'
        if data['type'] in ('singleSelect', 'multipleSelects'):
            opts = [o['name'] for o in data['options']['choices']]
            prompt += f', Options: {"; ".join(opts)}'
        prompt += '\n'
    prompt += '\n\n'
    logger.debug(f'Template matcher prompt: {prompt}')
    for output in llm_transform_list(prompt, [json.dumps(o) for o in objs], sys_prompt=sys_prompt, **kw):
        if output is not None:
            yield load_llm_json(output)
        else:
            yield {}

def search_objects_llm(search_objs: list[dict],
                       db_objs: list[dict],
                       search_preprocess_fn:None|Callable=None,
                       db_preprocess_fn:None|Callable=None,
                       max_workers:int|None=None,
                       additional_prompt: str='',
                       search_chunk_size: int=0,
                       db_chunk_size: int=0,
                       **kw) -> list[tuple[int, float]]:
    """Searches for correspondences between 2 lists of objects using an LLM.

    For each item in `search_objs`, we ask the LLM to find the best match in `db_objs`, as well as a
    confidence value between 0-1. We return a list corresponding to `search_objs`, where each item
    is a tuple `(db_obj_idx, confidence)`. Note that if there is no good match, the index will be
    -1.

    If given, we first run each search object through `search_preprocess_fn(obj)`, and each db object
    through `db_preprocess_fn(obj)` (both in parallel via a thread pool). These functions should
    take a single object and return a single object. They are useful for cleaning up the objects,
    such as removing extraneous fields, etc.

    The `max_workers` argument is passed to the thread pool executor. If `None`, we use the default
    number of workers.

    We optionally chunk the search and db objects into smaller chunks, using the `search_chunk_size` and
    `db_chunk_size` arguments. If either is 0, we do not chunk that list. Note that even for modern
    models like llama4 that have large context windows, in practice they tend to do much better with
    shorter inputs. But there is no science to it (yet), so you just have to experiment.

    In the case of db object chunking, we take the db obj index with the highest confidence for each
    search object.

    All other keyword arguments are passed to the LLM call (call_llm.single()). You can also pass in
    `additional_prompt` which is inserted in the system prompt (before output formatting
    instructions).
    """
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        if search_preprocess_fn:
            search_objs = list(pool.map(search_preprocess_fn, search_objs))
        if db_preprocess_fn:
            db_objs = list(pool.map(db_preprocess_fn, db_objs))

    sys_prompt = f'''You are an intelligent data matcher. Given a list of search objects, you try to
    match each one to the best match in a list of database objects. The objects are likely to have
    very similar schemas, but not identical. Use your best judgement and consider the entire set of
    database objects and search objects to figure out which database object is most similar to each
    search object.

    {additional_prompt}

    Output a JSON list with one output item for each input search object. Each output item is a
    pair [db_object_index, confidence], where db_object_index is the index of the best match in the
    database objects list, and confidence is a number between 0 and 1, indicating how confident you
    are that this is the best match. For example, with 2 search objects and 45 db objects, the
    output might look like: [[21,0.4],[39,0.7]]

    If you are not confident that there is any good match, return -1 for the index and 0 for the
    confidence.

    Output no other text or explanations, just the JSON object.
    '''
    logger.debug(f'Search matcher sys prompt: {sys_prompt}')

    def chunkify_indices(objs: list[dict], chunk_size: int) -> list[list[int]]:
        """Chunks indices of `objs` into smaller lists of size `chunk_size`."""
        if chunk_size <= 0:
            return [list(range(len(objs)))]
        else:
            return list(chunked(range(len(objs)), chunk_size)) # type: ignore[arg-type]

    search_chunks = chunkify_indices(search_objs, search_chunk_size)
    db_chunks = chunkify_indices(db_objs, db_chunk_size)
    futures = []
    for search_chunk in search_chunks:
        for db_chunk in db_chunks:
            search_objs_str = '\n'.join([f'{i}. {json.dumps(search_objs[idx])}' for i, idx in enumerate(search_chunk)])
            db_objs_str = '\n'.join([f'{i}. {json.dumps(db_objs[idx])}' for i, idx in enumerate(db_chunk)])
            user_prompt = f'''SEARCH OBJECTS:
            {search_objs_str}

            DATABASE OBJECTS:
            {db_objs_str}
            '''
            logger.debug(f'Got user prompt: {user_prompt}')
            msgs = [('system', sys_prompt), ('user', user_prompt)]
            future = call_llm.single_future(msgs, **kw)
            futures.append((search_chunk, db_chunk, future))
    logger.debug(f'Got {len(futures)} futures')
    ret = [[-1, 0] for o in search_objs]
    for search_chunk, db_chunk, future in futures:
        llm_output = future.result()
        logger.debug(f'for search chunk {search_chunk} and db chunk {db_chunk} got output {llm_output}')
        out = json.loads(llm_output)
        for search_idx, (db_idx, conf) in zip(search_chunk, out):
            db_idx += db_chunk[0]
            cur_idx, cur_conf = ret[search_idx]
            if cur_idx == -1 or cur_conf < conf:
                ret[search_idx] = [db_idx, conf]
    return ret
