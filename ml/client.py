"""Client for our LLM/embeddings server.

You can safely import this code without needing numpy, torch, etc.

The *_raw functions return the raw json response from the server (as a dict).

Note that by default, we use caching for all calls. Set `use_cache=False` to disable caching.
"""

#TODO add async versions

from __future__ import annotations

import tempfile

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Union

import requests

from .llm_constants import SERVER_BASE_URL, SERVER_API_VERSION

def single_call(endpoint: str, model:Optional[str]=None, **kw) -> dict:
    """Calls a single endpoint on the server. Returns the raw json response (as a dict)."""
    url = f"{SERVER_BASE_URL}/v{SERVER_API_VERSION}/{endpoint}"
    data = dict(**kw)
    if model is not None:
        data['model'] = model
    return requests.post(url, json=data).json()

def call_llm_raw(prompt: str, max_tokens:int =128, model:Optional[Any] =None, use_cache=True, **kw) -> dict:
    """Calls our local llm server for a completion.

    Uses the 'mistral-7b-instruct-v0.2.Q4_K_M.gguf' by default.

    Returns the raw json response (as a dict).
    """
    return single_call("completions",
                       prompt=prompt,
                       max_tokens=max_tokens,
                       model=model,
                       use_cache=use_cache,
                       **kw)

def call_llm(prompt: str, max_tokens:int =128, model:Optional[Any] =None, use_cache=True, **kw) -> str:
    """Calls our local llm server for a completion.

    Uses the 'mistral-7b-instruct-v0.2.Q4_K_M.gguf' by default.

    Returns the completion (as a string).
    """
    return call_llm_raw(prompt, max_tokens, model, use_cache, **kw)['choices'][0]['text']

def embed_text_raw(s: str, model='sentence', use_cache=True, **kw) -> dict:
    """Embeds a string using the specified model.

    Models:
    - 'sentence': BAAI/bge-large-en-v1.5 [default]
    - 'clip': openai/clip-vit-large-patch14

    Returns the raw json response (as a dict).
    """
    return single_call("embeddings", input=s, model=model, use_cache=use_cache, **kw)

def embed_text(s: str, model='sentence', use_cache=True, **kw) -> list:
    """Embeds a string using the specified model.

    Models:
    - 'sentence': BAAI/bge-large-en-v1.5 [default]
    - 'clip': openai/clip-vit-large-patch14

    Returns the embedding (as a list of floats).
    """
    return embed_text_raw(s, model=model, use_cache=use_cache, **kw)['data'][0]['embedding']

def embed_texts(input_strings: list[str], model='sentence', use_cache=True, **kw) -> list:
    """Embeds a list of strings using the specified model.

    Models:
    - 'sentence': BAAI/bge-large-en-v1.5 [default]
    - 'clip': openai/clip-vit-large-patch14

    This uses a ThreadPoolExecutor to parallelize the calls.

    Returns lists of embeddings (as a list of floats) in the same order as the inputs.
    """
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(embed_text, s, model, use_cache, **kw) for s in input_strings]
        return [future.result() for future in futures]

def strsim_raw(a: str, b: str, model='clip', use_cache=True, **kw) -> float:
    """Computes the similarity between two strings.

    Uses the 'openai/clip-vit-large-patch14' model by default.

    Returns the raw json response (as a dict).
    """
    return single_call("strsim", a=a, b=b, model=model, use_cache=use_cache, **kw)

def strsim(a: str, b: str, model='clip', use_cache=True, **kw) -> float:
    """Computes the similarity between two strings.

    Uses the 'openai/clip-vit-large-patch14' model by default.

    Returns the similarity score (as a float).
    """
    return strsim_raw(a, b, model=model, use_cache=use_cache, **kw)['similarity']

def strsims(input_pairs: list[tuple[str, str]], model='clip', use_cache=True, **kw) -> list[float]:
    """Computes the similarity between a list of string pairs.

    Uses the 'openai/clip-vit-large-patch14' model by default.

    This uses a ThreadPoolExecutor to parallelize the calls.

    Returns the similarity scores (as a list of floats) in the same order as the inputs.
    """
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(strsim, a, b, model, use_cache, **kw) for a, b in input_pairs]
        return [future.result() for future in futures]

def embed_image_url_raw(url: str, model='image', use_cache=True, **kw) -> dict:
    """Embeds an image (url or local path) using the specified model.

    Uses the 'openai/clip-vit-large-patch14' model by default.

    Returns the raw json response (as a dict).
    """
    return single_call("image_embeddings", url=url, model=model, use_cache=use_cache, **kw)

def embed_image_url(url: str, model='image', use_cache=True, **kw) -> dict:
    """Embeds an image (url or local path) using the specified model.

    Uses the 'openai/clip-vit-large-patch14' model by default.

    Returns the embedding (as a list of floats).
    """
    return embed_image_url_raw(url, model=model, use_cache=use_cache, **kw)['data'][0]['embedding']

def embed_image_urls(urls: list[str], model='image', use_cache=True, **kw) -> list:
    """Embeds a list of images (urls or local paths) using the specified model.

    Uses the 'openai/clip-vit-large-patch14' model by default.

    This uses a ThreadPoolExecutor to parallelize the calls.

    Returns lists of embeddings (as a list of floats) in the same order as the inputs.
    """
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(embed_image_url, url, model, use_cache, **kw) for url in urls]
        return [future.result() for future in futures]


def embed_image_raw(img: Any, model='image', use_cache=True, **kw) -> dict:
    """Embeds an image (loaded PIL image) using the specified model.

    Uses the 'openai/clip-vit-large-patch14' model by default.

    Returns the raw json response (as a dict).
    """
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        img.save(f.name)
        ret = single_call("image_embeddings", url=f.name, model=model, use_cache=use_cache, **kw)
    return ret

def embed_image(img: Any, model='image', use_cache=True, **kw) -> dict:
    """Embeds an image (loaded PIL image) using the specified model.

    Uses the 'openai/clip-vit-large-patch14' model by default.

    Returns the embedding (as a list of floats).
    """
    return embed_image_raw(img, model=model, use_cache=use_cache, **kw)['data'][0]['embedding']

def embed_images(images: list, model='image', use_cache=True, **kw) -> list:
    """Embeds a list of images (loaded PIL images) using the specified model.

    Uses the 'openai/clip-vit-large-patch14' model by default.

    This uses a ThreadPoolExecutor to parallelize the calls.

    Returns lists of embeddings (as a list of floats) in the same order as the inputs.
    """
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(embed_image, img, model, use_cache, **kw) for img in images]
        return [future.result() for future in futures]



if __name__ == '__main__':
    # check that we're not importing torch or numpy, etc
    import sys
    disallowed = ['torch', 'numpy', 'transformers', 'PIL']
    for key in sys.modules.keys():
        if any([dis in key for dis in disallowed]):
            print(f"Error: {key} is imported.")
            sys.exit(1)
    # test all client functions (raw)
    s = "Once upon a time, "
    print(call_llm_raw(s))
    print(embed_text_raw(s))
    print(embed_text_raw(s, model='clip'))
    print(strsim_raw("dog", "cat"))
    print(strsim_raw("dog", "philosophy"))
    print(embed_image_url_raw("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Ichthyotitan_Size_Comparison.svg/512px-Ichthyotitan_Size_Comparison.svg.png"))
    print(embed_image_url_raw("512px-Ichthyotitan_Size_Comparison.svg.png"))
    from PIL import Image
    img = Image.open("512px-Ichthyotitan_Size_Comparison.svg.png")
    print(embed_image_raw(img))
    # test all non-raw client functions
    print(embed_text(s))
    print(embed_text(s, model='clip'))
    print(strsim("dog", "cat"))
    print(strsim("dog", "philosophy"))
    print(embed_image_url("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Ichthyotitan_Size_Comparison.svg/512px-Ichthyotitan_Size_Comparison.svg.png"))
    print(embed_image_url("512px-Ichthyotitan_Size_Comparison.svg.png"))
    print(embed_image(img))
    # test all batch functions
    print(embed_texts([s, "dog", "cat"]))
    print(strsims([("dog", "cat"), ("dog", "philosophy")]))
    print(embed_image_urls(["https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Ichthyotitan_Size_Comparison.svg/512px-Ichthyotitan_Size_Comparison.svg.png", "512px-Ichthyotitan_Size_Comparison.svg.png"]))
    print(embed_images([img, img]))
