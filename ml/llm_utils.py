"""Various higher-level utilities for working with LLMs."""

#TODO figure out how to match two sets of dicts of data using LLMs/etc
#TODO   see https://chatgpt.com/share/674361be-3528-8012-9c9b-a83859cdf170

from __future__ import annotations

import json
import logging
import re

from collections import Counter
from pprint import pformat
from typing import Any, Iterator, Optional, Sequence

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
                       **kw) -> Iterator[Optional[str]]:
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

def clean_html_for_llm(s: str, max_length=200000, **kw) -> str:
    """Cleans up html for LLM input, primarily for length.

    - `max_length`: truncate the string to this length

    Pass in some `kw` else we do all of them. These are:
    - `image_data=True`: replace b64 image data with a placeholder
    - `quoted_image_data=True`: replace quoted b64 image data with a placeholder
    - `remove_svg=True`: remove all SVG elements
    - `remove_links=True`: remove all <link> elements (not actual <a> links)
    """
    orig_len = len(s)
    default_kw = dict(image_data=True,
                      quoted_image_data=True,
                      remove_svg=True,
                      remove_links=True,
                      remove_style=True,
                      remove_script=True,
                      )
    if not kw:
        kw = default_kw
    img_exts = ['png', 'jpg', 'jpeg', 'gif']
    img_data_prefix = 'data:image/(' + '|'.join(img_exts) + ')'
    def remove_tag(tag, s):
        # note that tags can be nested
        return re.sub(rf'<{tag}[^>]*>.*?</{tag}>', '', s, flags=re.DOTALL)

    for key, value in kw.items():
        if key == 'image_data' and value:
            data_re = re.compile(img_data_prefix + r';base64,[^"]+')
            s = data_re.sub('data:image/png;base64,PLACEHOLDER', s)
        if key == 'quoted_image_data' and value:
            quoted_data_re = re.compile(r'data:image/png%3bbase64%2c[^"]+')
            s = quoted_data_re.sub('data:image/png%3bbase64%2cPLACEHOLDER', s)
        if key == 'remove_links' and value:
            s = re.sub(r'<link [^>]*>', '', s)
        if key == 'remove_svg' and value:
            s = remove_tag('svg', s)
        if key == 'remove_style' and value:
            s = remove_tag('style', s)
        if key == 'remove_script' and value:
            s = remove_tag('script', s)

    msg = f'Got input string type {type(s)}, len {orig_len} -> {len(s)}'
    s = s[:max_length]
    msg += f' -> {len(s)}, {s[-100:]}'
    logger.info(msg)
    return s

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
    logger.info(f'Template matcher prompt: {prompt}')
    for output in llm_transform_list(prompt, [json.dumps(o) for o in objs], sys_prompt=sys_prompt, **kw):
        if output is not None:
            yield json.loads(output)
        else:
            yield {}

def simplify_html(html: str, idx: int|None=None) -> str:
    """Simplifies htmls by checking against text content.

    For each node, if the node has no text content, we remove it.
    Or, if it has the exact same text content as one of its children, we remove it.
    """
    soup = BeautifulSoup(html, "html.parser")
    orig_clean = soup.prettify()
    counts: Counter[str] = Counter()
    for node in soup.find_all():
        counts['total'] += 1
        if isinstance(node, Comment):
            #print(f"  Killing comment node with {len(node.text)} bytes: {node.text[:100]}...")
            counts['comment'] += 1
            counts['killed'] += 1
            node.extract()
#       elif node.name in ['img']:
#           counts['img'] += 1
#           counts['alive'] += 1
        elif node.name in ['style', 'script', 'footer']:
            #print(f"  Killing {node.name} node with {len(node.text)} bytes: {node.text[:100]}...")
            counts[node.name] += 1
            counts['killed'] += 1
            node.extract()
        elif not node.text.strip():
            #print(f'    Killing {node.name} node with no text content: {str(node)[:100]}...')
            node.extract()
            counts['empty'] += 1
            counts['killed'] += 1
        # check for any nodes with a style attribute that contains 'display: none' or 'visibility: hidden'
        elif node.has_attr('style') and ('display: none' in node['style'] or 'visibility: hidden' in node['style']):
            #print(f'    Killing {node.name} node with hidden style: {str(node)[:100]}...')
            node.extract()
            counts['hidden'] += 1
            counts['killed'] += 1
        else:
            counts['alive'] += 1
            #counts[f'type-{node.name}'] += 1

    def simplify_node(node):
        """Recursive function to check if a node can be simplified.

        If this node has a single child with text content identical to this node itself, then we can
        remove this node and replace it with the child.
        """
        if not hasattr(node, 'children'):
            return
        if not node.children:
            return
        children = list(node.children)
        if len(children) == 1:
            child = children[0]
            if child.text == node.text:
                #print(f"  Simplifying node {node.name} with {len(node.text)} bytes: {node.text[:100]}...")
                node.replace_with(child)
                counts['simplified'] += 1
        for child in children:
            simplify_node(child)

    first = str(soup)
    C = lambda s: f'{len(s)}b/{count_tokens(s)}t ({100.0*count_tokens(s) / len(s):.0f}%)'
    simplify_node(soup)
    ret = str(soup)
    logger.info(f"Simplified html from {C(html)} -> {C(first)} -> {C(ret)}, {pformat(counts)}")
    # write versions to disk if we were given an idx
    if idx is not None:
        with open(f'streeteasy_raw_{idx}.html', 'w') as f:
            f.write(orig_clean)
        with open(f'streeteasy_clean_{idx}.html', 'w') as f:
            f.write(soup.prettify())
    return ret
