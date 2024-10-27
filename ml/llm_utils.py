"""Various higher-level utilities for working with LLMs."""

import re

from typing import Iterator, Optional, Sequence

from nkpylib.ml.client import call_llm, chunked

CHUNKED_PROMPT = """%s

%d input items are enumerated below, one per line, prefixed by the item number. Return exactly one
output item per input item, in the same order and with the same item numbers. Return no other text.
If you have an error processing an input item, return the text "error" for that item (prefixed with
the item number).

\n\n%s"""

def llm_transform_list(base_prompt: str,
                       items: Sequence[str],
                       max_tokens:int =8000,
                       model:str ='llama3',
                       chunk_size:int = 10,
                       prompt_fmt: str=CHUNKED_PROMPT,
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
    for chunk in chunked(items, chunk_size):
        lst = ''.join(f'{i+1}. {item}\n' for i, item in enumerate(chunk))
        prompt = prompt_fmt % (base_prompt, len(lst), lst)
        f = call_llm.single_future(prompt, max_tokens=max_tokens, model=model, **kw)
        futures.append((chunk, f))
    for chunk, future in futures:
        llm_output = future.result()
        lines = [output_re.match(l) for l in llm_output.split('\n')]
        # make a dict from output item number to output text
        out = {int(l.group(1)): l.group(2) for l in lines if l is not None}
        # now we can build the output list
        for i, input in enumerate(chunk):
            if i+1 in out:
                yield out[i+1]
            else:
                yield None
