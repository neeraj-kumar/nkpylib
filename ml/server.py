"""LLM and embedding (and other ML) server code.

This is a server you can run that provides a REST API for various kinds of ML, particularly those
with large models. The idea is that all the heavy dependencies (and different implementations) are
encapsulated here, and you can interact with this using the `client.py` module without taking on any
of the dependencies.

In the future, I hope to make this more general and more robust, but for now it's a simple fastapi
server.

To run it:

    uvicorn server:app --reload --port 8234 --host 0.0.0.0

In general, most functions require a model name, but there's always a sensible default.
The list of default models (with short names) is in `nkpylib.ml.constants.DEFAULT_MODELS`.

By default, we cache results for each model, but you can turn this off by setting `use_cache=False`
in your request.

The server always returns "raw" responses (which are typically JSON). The client code has options
for getting a "processed" version of the response, which is more useful for most cases.

Anywhere a `url` is specified, it can also be a local path on this machine.

Current ML types and models:
- Text completion:
  - `mistral-7b-instruct-v0.2.Q4_K_M.gguf`: The Meta-Llama model with 7B parameters, fine-tuned on
    instruct prompts, with a Q4_K_M quantization
  - `llama-3`: The Llama-3-70b model running via Replicate, which is close to the state of the art
    as of Mid-2024.
- Text embeddings:
  - `st`: The SentenceTransformers library, which has a variety of models. The default is
    'BAAI/bge-large-en-v1.5', which is a large English model, and performs well on benchmarks for longish text.
  - `clip`: The OpenAI CLIP model, which can embed both text and images. Use this for cases where
    you expect to be matching text against images. Note that the text cannot be too long.
- String similarity:
  - This embeds two strings using any of the text embedding models, and then computes the cosine
    similarity between the two embeddings. Higher is more similar.
- Image embeddings:
  - `clip`: The OpenAI CLIP model, which can embed both text and images (in the same space).
"""

#TODO have some kind of context manager for deciding where to run llm functions from? (local, replicate, openai)
#TODO some way to turn an LLM query into an embeddings + code query (e.g. recipe pdf name correction)

import asyncio
import functools
import logging
import tempfile
import time
import uuid

from hashlib import sha256
from typing import Any, Callable, Literal, Optional, Union
from urllib.request import urlretrieve

import fastapi
import numpy as np
import requests

from PIL import Image
from pydantic import BaseModel

from nkpylib.ml.constants import DEFAULT_MODELS, LOCAL_MODELS, Role, Msg, data_url_from_file
from nkpylib.ml.providers import call_external, call_provider
from nkpylib.ml.text import get_text
from nkpylib.web_utils import make_request_async, dl_temp_file
from nkpylib.utils import is_instance_of_type
from nkpylib.ml.newserver import ExternalChatModel, LocalChatModel

logger = logging.getLogger(__name__)

app = fastapi.FastAPI()
app.state.models = {}
app.state.cache = {}

# load func takes model name and **kw, and returns the loaded model
LoadFuncT = Callable[[Any], Any]
RunFuncT = Callable[[Any, Any], dict]

@functools.cache
def load_clip(model_name=DEFAULT_MODELS['clip']):
    """Loads clip and returns two embedding functions: one for text, one for images"""
    import torch
    from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoTokenizer # type: ignore
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


async def load_model(model_name: str, load_func: LoadFuncT, **kw) -> tuple[object, dict, bool]:
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
        model = await asyncio.to_thread(load_func, model_name, **kw)
        t1 = time.time()
        app.state.models[model_name] = model
        app.state.cache[model_name] = {}
        app.state.load_time = t1-t0
        did_load = True
    else:
        model = app.state.models[model_name]
        did_load = False
    return model, app.state.cache[model_name], did_load

async def generic_run_model(
        input: Any,
        model_name: str,
        load_func: LoadFuncT,
        run_func: RunFuncT,
        cache_key: Optional[str]=None,
        **kw) -> dict:
    """Generic code for loading and running a model, including caching behavior.

    This is a helper function that loads the model into a global cache using `load_func`, and then
    runs the model using `run_func`. It also caches the output of `run_func` if `cache_key` is
    specified.

    The `load_func` calls `load_model(model_name, load_func)` and returns `(model, cache, did_load)`.
    The 2nd returned arg is the model's own cache dict for caching its outputs, and the 3rd is a bool
    indicating whether the model was actually loaded or not (i.e., it was already loaded).

    The model is run using `run_func(input, model, **kw)`.

    We return a dict with the output returned from the `run_func`, as well as a key `timing` with
    various timing information.
    """
    t0 = time.time()
    model, cache, did_load = await load_model(model_name, load_func)
    #logger.debug(f'Loaded model {model_name} with cache keys {cache.keys()}, checking cache_key {cache_key}')
    t1 = time.time()
    found_cache = False
    if cache_key in cache:
        ret = cache[cache_key]
        found_cache = True
    else:
        ret = await asyncio.to_thread(run_func, input, model, **kw)
        if cache_key is not None:
            cache[cache_key] = ret
    t2 = time.time()
    ret['timing'] = dict(
        load_time=t1-t0 if did_load else 0,
        generate=t2-t1,
        did_load=did_load,
        found_cache=found_cache,
    )
    return ret


# setup fastapi chat endpoint
class ChatRequest(BaseModel):
    prompts: str|list[Msg]
    model: Optional[str]='chat' # this will get mapped bsaed on DEFAULT_MODELS['chat']
    max_tokens: Optional[int]=1024
    kwargs: Optional[dict]={}
    use_cache: Optional[bool]=False
    provider: Optional[str]=''

@app.post("/v1/chat")
async def chat(req: ChatRequest):
    """Generates chat response for the given prompts using the given model."""
    if req.model in DEFAULT_MODELS:
        req.model = DEFAULT_MODELS[req.model]
    if req.model in LOCAL_MODELS:
        model = LocalChatModel(model_name=req.model, device='cpu', use_cache=req.use_cache)
    else:
        model = ExternalChatModel(model_name=req.model, use_cache=req.use_cache)
    ret = await model.run(
        input=req.prompts,
        max_tokens=req.max_tokens,
        provider=req.provider,
        **req.kwargs
    )
    return ret

# setup fastapi VLM endpoint
class VLMRequest(BaseModel):
    image: str # path/url/image directly?
    prompts: str|list[Msg]
    model: Optional[str]='vlm' # this will get mapped bsaed on DEFAULT_MODELS['vlm']
    max_tokens: Optional[int]=1024
    kwargs: Optional[dict]={}
    use_cache: Optional[bool]=False
    provider: Optional[str]=''

@app.post("/v1/vlm")
async def vlm(req: VLMRequest):
    """Generates VLM chat response for the given image and prompts using the given model."""
    cache_key = f"{req.max_tokens}:{req.image}:{str(req.prompts)}" if req.use_cache else None
    assert req.model is not None, "Model must be specified for vlm request"
    if req.model in DEFAULT_MODELS:
        req.model = DEFAULT_MODELS[req.model]
    logger.debug('VLM request:', req, cache_key)
    # run this using external api
    def load_func(model, **kw):
        return model

    def run_func(input, model, **kw):
        logger.debug(f'Running external vlm model: {model} on input: {input} with kw {kw}')
        image, prompts = input
        if isinstance(prompts, str):
            prompts = [('user', prompts)]
        # check that `prompts` is now a sequence of Msg
        assert is_instance_of_type(prompts, list[Msg]), f"Prompts should be of type {list[Msg]}, actually: {prompts}"
        if not kw:
            kw = {}
        kw['messages'] = [{'role': role, 'content': text} for role, text in prompts]
        # if the image is not a web url, it must be a local path, so convert to a data url
        if not image.startswith('http'):
            with open(image, 'rb') as f:
                image = data_url_from_file(f)
        # the image goes in the first user message
        for msg in kw['messages']:
            if msg['role'] == 'user':
                cur_text = msg['content']
                msg['content'] = [{"type": "text", "text": cur_text},
                                  {"type": "image_url", "image_url": {"url": image}}]
                break
        ret = asyncio.run(call_external(endpoint='/chat/completions', provider_name=req.provider, model=model, **kw))
        return ret

    ret = await generic_run_model(
        input=(req.image, req.prompts),
        model_name=req.model,
        load_func=load_func,
        run_func=run_func,
        cache_key=cache_key,
        max_tokens=req.max_tokens,
    )
    # the output is already in openai compatible format, just do some cleanup
    ret['prompts'] = req.prompts
    ret['model'] = ret.get('model', req.model).split('/', 1)[-1]
    ret['max_tokens'] = req.max_tokens
    logger.debug('VLM response:', ret)
    return ret


class TextEmbeddingRequest(BaseModel):
    input: str
    model: Optional[str]=DEFAULT_MODELS['st']
    provider: Optional[str]=''
    use_cache: Optional[bool]=False


@app.post("/v1/embeddings")
async def text_embeddings(req: TextEmbeddingRequest):
    """Generates embeddings for the given text using the given model."""
    assert req.model is not None, "Model must be specified for embeddings request"
    if req.model in DEFAULT_MODELS:
        req.model = DEFAULT_MODELS[req.model]
    logger.debug('checking embedding req against model name', req.model)
    if req.model == DEFAULT_MODELS['clip']:
        model_type = 'clip'
    elif req.model == DEFAULT_MODELS['sentence']:
        model_type = 'sentence'
    else:
        model_type = 'external'
    print('settled on model type', model_type)

    def load_func(model, **kw):
        from sentence_transformers import SentenceTransformer # type: ignore
        if model_type == 'clip':
            return load_clip(model)[0]
        elif model_type == 'sentence':
            return SentenceTransformer(model)
        else:
            return model

    def run_func(input, model, **kw):
        def postprocess(embedding):
            return dict(
                object='list',
                data=[dict(
                    object='embedding',
                    index=0,
                    embedding=embedding.tolist(),
                )],
                model=req.model,
                n_dims=len(embedding),
            )

        if model_type == 'clip': # get clip text embedding
            embedding = postprocess(get_clip_text_embedding(input))
        elif model_type == 'sentence': # sentence transformer
            #embedding = postprocess(model.encode_multi_process(documents, pool))
            embedding = postprocess(model.encode([input], normalize_embeddings=True)[0])
        else: # external api call
            embedding = asyncio.run(call_external(endpoint='/embeddings', provider_name=req.provider, model=model, input=input))
            print('embedding', embedding)
        # also add the input to the returned embedding object
        embedding['input'] = input
        return embedding

    return await generic_run_model(
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
    assert req.model is not None, "Model must be specified for image embeddings request"
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

    return await generic_run_model(
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
    timings: dict[str, float] = {}
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


class GetTextRequest(BaseModel):
    url: str
    use_cache: Optional[bool]=False
    kw: Any=None

@app.post("/v1/get_text")
async def get_text_api(req: GetTextRequest, cache={}):
    """Gets the text from the given URL or path of pdf, image, or text."""
    # check cache
    cache_key = req.url if req.use_cache else None
    if cache_key in cache:
        return cache[cache_key]
    # if it's a url or data url, download it
    kw = req.kw or {}
    async with dl_temp_file(req.url) as path:
        ret = await asyncio.to_thread(get_text, path, **kw)
    _ret = dict(url=req.url, text=ret)
    if cache_key is not None:
        cache[cache_key] = _ret
    return _ret


class TranscriptionRequest(BaseModel):
    url: str|bytes # path/url to file or audio bytes
    model: Optional[str]='transcription' # this will get mapped bsaed on DEFAULT_MODELS['Transcription']
    language: Optional[str]='en'
    chunk_level: Optional[str]='segment'
    kwargs: Optional[dict]={}
    use_cache: Optional[bool]=False
    provider: Optional[str]='deepinfra'

@app.post("/v1/transcription")
async def speech_transcription(req: TranscriptionRequest):
    """Generates a transcription object for the given audio (path, url, or bytes)."""
    assert req.model is not None, "Model must be specified for speech transcription request"
    if req.model in DEFAULT_MODELS:
        req.model = DEFAULT_MODELS[req.model]
    audio = req.url
    if isinstance(req.url, str): # it's a url/path; download it
        async with dl_temp_file(req.url) as path:
            with open(path, 'rb') as f:
                audio = f.read()
    # at this point `audio` is a bytes object
    sha = sha256(audio).hexdigest()
    cache_key = f'{sha}:{req.language}:{req.chunk_level}' if req.use_cache else None
    kw = req.kwargs or {}
    kw.update(
        language=req.language,
        chunk_level=req.chunk_level,
    )
    logger.debug(f'Speech request: {req}, {cache_key}, {kw}')
    # run this using external api
    def load_func(model, **kw):
        return model

    def run_func(input, model, **kw):
        logger.debug(f'Running external transcription model: {model} with {kw}')
        audio = input
        loop = asyncio.get_event_loop()
        ret = loop.run_until_complete(call_provider(req.provider,
                                  endpoint=f'https://api.deepinfra.com/v1/inference/{model}',
                                  data=audio,
                                  **kw))
        return ret

    ret = await generic_run_model(
        input=audio,
        model_name=req.model,
        load_func=load_func,
        run_func=run_func,
        cache_key=cache_key,
        **kw,
    )
    # the output is already in openai compatible format, just do some cleanup
    ret['model'] = ret.get('model', req.model).split('/', 1)[-1]
    logger.debug('Speech response:', ret)
    return ret


@app.get("/test")
async def test_api():
    print(f'got request for test: {time.time()}')
    #await asyncio.sleep(10)
    return "Hello world\n"


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test()
