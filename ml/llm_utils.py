"""Various higher-level utilities for working with LLMs."""

#TODO figure out how to match two sets of dicts of data using LLMs/etc
#TODO   see https://chatgpt.com/share/674361be-3528-8012-9c9b-a83859cdf170

from __future__ import annotations

import json
import logging
import re

from typing import Any, Iterator, Optional, Sequence

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
                       max_tokens:int =100000,
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

    s = s[:max_length]
    logger.info(f'Got input string type {type(s)}, len {orig_len} -> {len(s)}, {s[-100:]}')
    return s
