"""LLM and embedding server code.

The ap
"""

import functools
import time
import uuid

from typing import Any, Optional, Union

import fastapi
import numpy as np
import requests

from PIL import Image
from pydantic import BaseModel

from llm_constants import DEFAULT_MODELS

app = fastapi.FastAPI()
app.state.models = {}
app.state.cache = {}

device = 'cpu'

@functools.cache
def load_clip(model_name=DEFAULT_MODELS['clip']):
    """Loads clip and returns two embedding functions: one for text, one for images"""
    import torch
    from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    def get_image_features(image_or_path):
        if isinstance(image_or_path, str):
            if image_or_path.startswith('http'):
                image = Image.open(requests.get(image_or_path, stream=True).raw)
            else:
                image = Image.open(image_or_path)
        else:
            image = image_or_path
        with torch.no_grad():
            return model.get_image_features(**processor(images=image, return_tensors="pt"))[0]

    def get_text_features(text):
        with torch.no_grad():
            return model.get_text_features(**processor(text=text, return_tensors="pt"))[0]

    return get_text_features, get_image_features


def get_clip_text_embedding(s: str):
    """Gets the clip text embedding for the given `s`"""
    clip_text, clip_image = load_clip()
    return clip_text(s).numpy()


def get_clip_image_embedding(img: Union[str, Image.Image]):
    """Gets the clip image embedding for the given `img` (url, path, or image)"""
    clip_image, clip_image = load_clip()
    return clip_image(img).numpy()



def llama(prompt, methods, model_dir='models', get_embeddings=False):
    from llama_cpp import Llama
    from llama_cpp import (
        LLAMA_POOLING_TYPE_NONE,
        LLAMA_POOLING_TYPE_MEAN,
        LLAMA_POOLING_TYPE_CLS,
    )
    print(f'Loading model from {model_dir}, get_embeddings={get_embeddings}')
    model_path = f"{model_dir}/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
    model_path = f"{model_dir}/Meta-Llama-3-8B-Instruct.Q8_0.gguf"
    model_path = f"{model_dir}/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    if get_embeddings:
        model_path = f"{model_dir}/ggml-gritlm-7b-q4_k.gguf"

    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    t0 = time.time()
    llm = Llama(
      model_path=model_path,  # Download the model file first
      n_ctx=8192,#32768,      # The max sequence length to use - note that longer sequence lengths require much more resources
      n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
      n_gpu_layers=35,        # The number of layers to offload to GPU, if you have GPU acceleration available
      embedding=get_embeddings,  # Whether to create embeddings for the input and output
      #pooling_type=LLAMA_POOLING_TYPE_MEAN,
    )
    t1 = time.time()
    print(f'Loaded model using llama in {t1-t0}s')

    if get_embeddings:
        embeddings = llm.create_embedding(prompt)
        t2 = time.time()
        print(f'Took {t2-t1}s to create embeddings for prompt: {embeddings}')
    else:
        output = llm(
          f"<s>[INST] {prompt} [/INST]", # Prompt
          max_tokens=512,  # Generate up to 512 tokens
          stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
          echo=False,      # Whether to echo the prompt
        )
        t2 = time.time()
        print(f'Took {t2-t1}s to generate using llama-cpp for prompt: {prompt}')
        print(output)


def test(prompt = "Summarize the plot of the movie Fight Club."):
    methods = ['greedy', 'sample', 'beam']
    #methods = ['beam'] # 1.7 tokens/s

    func = llama
    func(prompt, methods)

# setup fastapi completion endpoint
class CompletionRequest(BaseModel):
    prompt: str
    model: Optional[str]='mistral-7b-instruct-v0.2.Q4_K_M.gguf'
    max_tokens: Optional[int]=128
    kwargs: Optional[dict]={}
    use_cache: Optional[bool]=False

def load_model(model_name: str, load_func: callable, **kw) -> tuple[object, dict, bool]:
    """Loads given `model_name` using the given `load_func`.

    Returns `(model, model-specific-cache, did_load)`, where:
    - `model` is the loaded model
    - `model-specific-cache` is a cache for the model (a dict)
    - `did_load` is a boolean indicating whether the model was loaded or not

    This caches the model in `app.state.models` and creates a cache for the model in
    `app.state.cache[model_name]` if it doesn't exist.
    """
    if model_name not in app.state.models:
        t0 = time.time()
        model = load_func(model_name, **kw)
        t1 = time.time()
        app.state.models[model_name] = model
        app.state.cache[model_name] = {}
        app.state.load_time = t1-t0
        did_load = True
    else:
        model = app.state.models[model_name]
        did_load = False
    return model, app.state.cache[model_name], did_load


def generic_run_model(
        input: Any,
        model_name: str,
        load_func: callable,
        run_func: callable,
        cache_key: Optional[str]=None,
        **kw) -> dict:
    """Generic code for loading and running a model, including caching behavior.
    """
    t0 = time.time()
    model, cache, did_load = load_model(model_name, load_func)
    t1 = time.time()
    found_cache = False
    if cache_key in cache:
        ret = cache[cache_key]
        found_cache = True
    else:
        ret = run_func(input, model, **kw)
        if cache_key is not None:
            cache[cache_key] = ret
    t2 = time.time()
    # the output is already in openai compatible format, just do some cleanup
    ret['timing'] = dict(
        load_time=t1-t0 if did_load else 0,
        generate=t2-t1,
        did_load=did_load,
        found_cache=found_cache,
    )
    return ret

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    """Generates completions for the given prompt using the given model."""
    cache_key = f"{req.max_tokens}:{req.prompt}" if req.use_cache else None
    if req.model in DEFAULT_MODELS:
        req.model = DEFAULT_MODELS[req.model]
    if 'llama-3' in req.model:
        # run this using replicate
        from replicate_wrapper import llm_complete
        def load_func(model, **kw):
            return model

        def run_func(input, model, **kw):
            print(f'Running llama-3 model: {model} on input: {input}')
            if model.startswith('models/'):
                model = model.split('/', 1)[-1]
            ret = llm_complete(prompt=input, model_name=model, **kw)
            # convert the returned string to a standard openai-compatible response
            ret = dict(
                object='text',
                model=model,
                choices=[dict(text=ret)],
            )
            return ret

    else:
        def load_func(model, **kw):
            from llama_cpp import Llama
            return Llama(
              model_path=model,
              n_ctx=8192,#32768,
              n_threads=8,
              n_gpu_layers=35,
            )

        def run_func(input, model, **kw):
            return model(
              f"<s>[INST] {input} [/INST]",
              max_tokens=kw['max_tokens'],
              stop=["</s>"],
              echo=False,
            )

    ret = generic_run_model(
        input=req.prompt,
        model_name=f'models/{req.model}',
        load_func=load_func,
        run_func=run_func,
        cache_key=cache_key,
        max_tokens=req.max_tokens,
    )
    # the output is already in openai compatible format, just do some cleanup
    ret['prompt'] = req.prompt
    ret['model'] = ret['model'].split('/')[-1]
    ret['max_tokens'] = req.max_tokens
    return ret


class TextEmbeddingRequest(BaseModel):
    input: str
    model: Optional[str]=DEFAULT_MODELS['st']
    use_cache: Optional[bool]=False


@app.post("/v1/embeddings")
async def text_embeddings(req: TextEmbeddingRequest):
    """Generates embeddings for the given text using the given model."""
    if req.model in DEFAULT_MODELS:
        req.model = DEFAULT_MODELS[req.model]
    def load_func(model, **kw):
        from sentence_transformers import SentenceTransformer
        if model == DEFAULT_MODELS['clip']:
            return load_clip(model)[0]
        else:
            return SentenceTransformer(model)

    def run_func(input, model, **kw):
        if req.model == DEFAULT_MODELS['clip']:
            embedding = get_clip_text_embedding(input)
        else:
            #embedding = model.encode_multi_process(documents, pool)
            embedding = model.encode([input], normalize_embeddings=True)[0]
        return dict(
            object='list',
            data=[dict(
                object='embedding',
                index=0,
                embedding=embedding.tolist(),
            )],
            model=req.model,
            n_dims=len(embedding),
            input=input,
        )

    return generic_run_model(
        input=req.input,
        model_name=req.model,
        load_func=load_func,
        run_func=run_func,
        cache_key=req.input if req.use_cache else None,
    )


class ImageEmbeddingRequest(BaseModel):
    url: str
    model: Optional[str]=DEFAULT_MODELS['image']
    use_cache: Optional[bool]=False


@app.post("/v1/image_embeddings")
async def image_embeddings(req: ImageEmbeddingRequest):
    """Generates embeddings for the given image url (or local path) using the given model."""
    if req.model in DEFAULT_MODELS:
        req.model = DEFAULT_MODELS[req.model]
    def run_func(input, model, **kw):
        embedding = get_clip_image_embedding(input)
        return dict(
            object='list',
            data=[dict(
                object='embedding',
                index=0,
                embedding=embedding.tolist(),
            )],
            model=req.model,
            n_dims=len(embedding),
            url=input,
        )

    return generic_run_model(
        input=req.url,
        model_name=req.model,
        load_func=lambda model, **kw: load_clip(model)[1],
        run_func=run_func,
        cache_key=req.url if req.use_cache else None,
    )


class StrSimRequest(BaseModel):
    a: str
    b: str
    model: Optional[str]=DEFAULT_MODELS['clip']
    use_cache: Optional[bool]=False

@app.post("/v1/strsim")
async def strsim(req: StrSimRequest):
    """Computes the strsim between `a` and `b` (higher is more similar)"""
    if req.model in DEFAULT_MODELS:
        req.model = DEFAULT_MODELS[req.model]
    # embed both texts by calling the text_embeddings function
    timings = {}
    async def call(s):
        ret = await text_embeddings(TextEmbeddingRequest(input=s, model=req.model, use_cache=req.use_cache))
        # update timings by adding all numbers, and taking "or" of boolean values
        for k, v in ret['timing'].items():
            if isinstance(v, bool):
                timings[k] = timings.get(k, False) or v
            else:
                timings[k] = timings.get(k, 0) + v
        return np.array(ret['data'][0]['embedding'])

    x = await call(req.a)
    y = await call(req.b)
    # compute the actual similarity
    ret = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    #print(f'strsim({a}, {b}) = {ret}')
    return dict(
        model=req.model,
        a=req.a,
        b=req.b,
        similarity=ret,
        timings=timings,
    )


if __name__ == '__main__':
    test()
