"""LLM and embedding (and other ML) server code.

This is a server you can run that provides a REST API for various kinds of ML, particularly those
with large models. The idea is that all the heavy dependencies (and different implementations) are
encapsulated here, and you can interact with this using the `client.py` module without taking on any
of the dependencies.

To run it:

    uvicorn server:app --reload --port 8234 --host 0.0.0.0

In general, most functions require a model name, but there's always a sensible default.
The list of default models (with short names) is in `nkpylib.ml.constants.DEFAULT_MODELS`.

By default, we cache results for each model, but you can turn this off by setting `use_cache=False`
in your request.

The server always returns "raw" responses (which are typically JSON). The client code has options
for getting a "processed" version of the response, which is more useful for most cases.

Anywhere a `url` is specified, it can also be a local (absolute) path on this machine.

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
    this to clip, except that it's quite a bit slower.
  - `mobilenet`: A MobileNetV3 model that provides reasonably good image embeddings quickly.
"""

from __future__ import annotations

#TODO some way to turn an LLM query into an embeddings + code query (e.g. recipe pdf name correction)

import asyncio
import atexit
import functools
import json
import logging
import os
import signal
import tempfile
import time
import uuid

from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import wraps
from hashlib import sha256
from pprint import pformat
from asyncio import Lock, Condition
from typing import Any, Callable, Literal, Optional, Union, Type
from urllib.request import urlretrieve

import anyio
import fastapi
import numpy as np
import requests
import torch

from PIL import Image
from fastapi import HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tqdm import tqdm

from nkpylib.ml.constants import (
    data_url_from_file,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODELS,
    LOCAL_MODELS,
    Msg,
    Role,
)
from nkpylib.ml.ml_types import nparray1d, array1d, array2d
from nkpylib.ml.providers import call_external, call_provider
from nkpylib.ml.text import get_text
from nkpylib.thread_utils import Singleton
from nkpylib.web_utils import make_request_async, dl_temp_file
from nkpylib.utils import is_instance_of_type

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """Context manager for FastAPI app lifespan.

    Do any startup setup before the first yield, and then any cleanup (that's done on reload as
    well) after that.
    """
    # Startup
    yield
    # Shutdown - this will be called on reload
    cleanup_executors()

app = fastapi.FastAPI(lifespan=lifespan)

# request queue sizes for different types of limits
SEMAPHORES = {
    "extfast": asyncio.Semaphore(50),
    "extslow": asyncio.Semaphore(20),
    "fast": asyncio.Semaphore(50),
    "medium": asyncio.Semaphore(10),
    "slow": asyncio.Semaphore(2),
}

MODEL_CACHE: dict = {}
RESULTS_CACHE: dict = {}

RESULTS_CACHE_LIMIT = 90000

# Global list to track all executors for cleanup
_EXECUTORS: list[ProcessPoolExecutor] = []

# load func takes model name and **kw, and returns the loaded model
LoadFuncT = Callable[[Any], Any]
RunFuncT = Callable[[Any, Any], dict]

# GLOBAL model storage (per process)
PROC_MODELS: dict[str, tuple] = {}  # {model_name: (text_func, image_func)}

def _default(name: str) -> str:
    """Returns the default model full name.

    - name: Short model name or full model name

    If the name is a key in `DEFAULT_MODELS`, returns the full model name.
    Otherwise returns the name unchanged.
    """
    if name not in DEFAULT_MODELS:
        return name
    return DEFAULT_MODELS[name].name


async def pil_image_from_input(image_or_path: Union[str, Image.Image], force_rgb: bool=False) -> Image.Image:
    """Loads and returns a PIL Image from a URL, local path, or PIL Image.

    If you set `force_rgb=True`, converts the image to RGB mode.
    """
    if isinstance(image_or_path, str):
        if image_or_path.startswith('http'):
            resp = await make_request_async(image_or_path, method='get', stream=True, min_delay=0.1)
            image = Image.open(resp.raw)
        else:
            image = Image.open(image_or_path)
    else:
        image = image_or_path
    if force_rgb and image.mode != 'RGB':
        image = image.convert('RGB')
    return image

@functools.cache
def load_jina(model_name=_default('jina'), dims:int =DEFAULT_MODELS['jina'].default_dims):
    """Loads Jina CLIP model and returns embedding functions.

    - model_name: Jina model name to load
    - dims: Embedding dimensions (max 1024)

    Returns a tuple of (text_embedding_func, image_embedding_func).
    Both functions return normalized numpy arrays.

    Based on the research paper, going from the max dims of 1024 to 768 doesn't hurt performance at
    all (and might even slightly improve it for some tasks).
    """
    #FIXME this seems to be semi-broken on my machine
    from sentence_transformers import SentenceTransformer
    cfg = DEFAULT_MODELS['jina']
    assert dims <= cfg.max_dims, "Jina-clip-v2 only supports up to 1024 dimensions"
    model = SentenceTransformer(model_name, trust_remote_code=True, truncate_dim=dims)

    def get_image_features(image_or_path):
        return model.encode([image_or_path], normalize_embeddings=True)[0]

    def get_text_features(text):
        # note that in the example code, for "query" embeddings,
        # they also set prompt_name='retrieval.query'...
        return model.encode([text], normalize_embeddings=True)[0]

    return get_text_features, get_image_features


class Model(ABC):
    """Base class providing a common interface for loading and running models.

    A model's basic functionality is:
    - Load the model (once, and then cached)
    - Run the model on inputs
    - Caching (of models, and optionally of results)
      - Both caches are global across all Model instances
      - The results cache is LRU and limited in size to a fixed number of entries
    - Timing and stats
    - Automatic batching (for performance, optional)

    In addition, we provide some other useful common functionality, such as image loading from
    various types of inputs, etc.

    Each model instance must have a string name; the same class can be used for multiple models.
    Models are loaded using arbitrary **kw. They are run with a single parameter `input` and
    arbitrary `kw`.

    As much as possible, we try to keep things async, so that multiple requests can be handled
    efficiently (concurrently where possible). We also use locks and conditions to coordinate
    between multiple requests for the same model/input (mostly trying to prevent duplicate work).

    As a subclass, you should implement the following methods:
    - `_load()`: Loads the model (can be anything you want) and returns it. Assigns to `self.model`.
      - If you need no loading (e.g. for external APIs), the base implementation returns the model
        name, which is often sufficient.
    - `_get_cache_key()`: Returns the cache key for a given input.
      - This is used to cache results. If you don't want caching, you can return None.
    - `_run()`: Runs the model on the given input and returns a dict.

    Optionally, you can also override:
    - `_run_batch()`: Runs a batch of inputs in an optimized way.
    - `update_kw()`: Updates input keyword arguments with model-specific defaults.
      - For example, if you want to set a default `max_tokens` parameter for text models.
    - `postprocess()`: Postprocesses the raw output dict from `_run()` into a standard format.
      - For example, chat models can add standard OpenAI-compatible fields here.
    """
    def __init__(self,
                 model_name: str='',
                 use_cache: bool=True,
                 max_cache_entries: int=RESULTS_CACHE_LIMIT,
                 enable_auto_batching: bool=False,
                 max_batch_size: int=10,
                 max_wait_ms: float=50,
                 **kw):
        self.model_cfg = DEFAULT_MODELS.get(model_name)
        if self.model_cfg:
            model_name = self.model_cfg.name
            self.max_tokens = self.model_cfg.max_tokens
        else:
            self.max_tokens = DEFAULT_MAX_TOKENS
        self.model_name = model_name
        self.use_cache = use_cache
        self.max_cache_entries = max_cache_entries
        # local state
        self.lock = Lock()
        self.condition = Condition()
        self.model: Any = None
        self.cache = RESULTS_CACHE.setdefault(self, OrderedDict())
        self.timing: dict[str, Any] = dict(
            model=model_name,
            n_calls=0,
            n_inferences=0,
            avg_generate=0,
            throughput=0,
            n_cache_hits=0,
            n_cache_misses=0,
            generate_all=0,
            inference_time=0,
            n_batch_calls=0,
            n_batch_inferences=0,
        )
        self.current: set[Any] = set() # what's currently running
        self.callers: Counter = Counter() # current set of callers
        # Auto-batching setup
        self.enable_auto_batching = enable_auto_batching
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        if enable_auto_batching:
            self.pending_requests: list = []
            self.batch_lock = Lock()
            self.batch_timer: Optional[asyncio.Task] = None
        logger.warning(f'Starting up {self.__class__.__name__} {self.model_name} with use_cache {self.use_cache}, auto_batching {self.enable_auto_batching}')

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}:{self.model_name}>'

    async def _load(self, **kw) -> Any:
        """Load implementation.

        This is called only once in the lifecycle of a Model instance, and should do all the setup
        needed. It can return anything, which will be saved as self.model, that can be used in e.g.
        the _run() method.model.

        This version just returns the model name as the model itself (useful for external APIs).
        """
        return self.model_name

    async def load(self, **kw) -> bool:
        """Loads the model if not already loaded.

        This handles caching, etc. The actual implementation is in `_load()`.

        - **kw: Additional keyword arguments for loading

        Returns `True` if the model was actually loaded, `False` if it was already cached.
        Uses a global model cache to avoid reloading the same model multiple times.
        """
        model_cache_key = (self.__class__, self.model_name)
        async with self.lock:
            if self.model is None:
                # first check if we have a cached version
                if model_cache_key in MODEL_CACHE:
                    self.model = MODEL_CACHE[model_cache_key]
                    logger.debug(f"Model {self.model_name} loaded from cache")
                    return False
                t0 = time.time()
                self.model = await self._load(**kw)
                t1 = time.time()
                MODEL_CACHE[model_cache_key] = self.model
                self.timing['load_time'] = t1 - t0
                self.timing['load_ts'] = t1
                logger.debug(f"Model {self.model_name} loaded in {t1-t0:.2f}s")
                return True
        return False

    def update_kw(self, input, **kw) -> dict:
        """Updates the keyword arguments with default parameters.

        - input: The input data being processed
        - **kw: Keyword arguments to update

        Override this method in subclasses to add model-specific defaults.
        """
        return kw

    @abstractmethod
    async def _get_cache_key(self, input: Any, **kw) -> str:
        """Returns the cache key for a given input and kw. Override this in your subclass"""
        ...

    @abstractmethod
    async def _run(self, input: Any, **kw) -> dict:
        """Run implementation for a single `input`. Override this in your subclass"""
        ...

    async def _run_batch(self, batch: list[dict]) -> list[dict]:
        """Process a batch of items individually but async-concurrently.

        Each item in the `batch` is a dict containing:
        - input: Input data
        - kw: Keyword arguments

        Override this to provide optimized batch processing.
        """
        async def process_single(input, kw):
            return await self._run(input, **kw)

        tasks = [process_single(item['input'], item['kw']) for item in batch]
        return await asyncio.gather(*tasks)


    async def run(self, input: Any, caller: str='', **kw) -> dict:
        """Runs the model with given `input` and `kw`.

        You should set a `caller` wherever possible, for logging/monitoring/prioritization.

        This calls either _single_run() or _batched_run() depending on whether auto-batching is
        enabled or not.

        This handles some of the common functionality like timing, caching, etc.
        """
        req_start_time = time.time()
        kw = self.update_kw(input, **kw)
        # get the cache key and load the model concurrently (only if needed)
        cache_key = None
        if self.use_cache or self.model is None:
            cache_key, self.timing['did_load'] = await asyncio.gather(
                self._get_cache_key(input, **kw) if self.use_cache else asyncio.sleep(0),
                self.load(**kw) if self.model is None else asyncio.sleep(0)
            )
        if cache_key and cache_key in self.cache: # cache hit
            ret = self.cache[cache_key].copy()
            self.timing['n_cache_hits'] += 1
            self.cache.move_to_end(cache_key)  # move to end to keep it fresh
            self.update_timings(ret, req_start_time=req_start_time, cache_key=cache_key)
            return ret
        else: # no caching, or cache miss
            if cache_key is not None: # cache miss
                # make sure no one else is running this exact input
                if cache_key in self.current:
                    async with self.condition:
                        await self.condition.wait_for(lambda: cache_key not in self.current)
                self.timing['n_cache_misses'] += 1
                self.current.add(cache_key)
            self.callers[caller] += 1
            try:
                # Route to appropriate implementation
                if self.enable_auto_batching:
                    ret = await self._batched_run(input, req_start_time=req_start_time, **kw)
                else:
                    ret = await self._single_run(input, **kw)
                    self.update_timings(ret, req_start_time=req_start_time, cache_key=cache_key)
                if cache_key is not None:
                    self.cache[cache_key] = ret.copy()
                return ret
            except Exception as e:
                logger.warning(f'Error during model {self.model_name} run on input {input}: {e}')
                return dict(
                    error=str(e),
                    error_type=type(e).__name__,
                    model=self.model_name, input=str(input)[:100],
                )
            finally:
                # Common cleanup
                self.callers[caller] -= 1
                # update cache
                if cache_key is not None:
                    while len(self.cache) > self.max_cache_entries:
                        self.cache.popitem(last=False)
                    if cache_key in self.current:
                        self.current.remove(cache_key)
                    async with self.condition:
                        self.condition.notify_all()

    async def _single_run(self, input: Any, **kw) -> dict:
        """Run a single input through the model, with caching and timing, etc.

        - input: Input data to process
        - kw: Keyword arguments (already processed)

        Handles caching, timing, and coordination between concurrent requests.
        """
        t2 = time.time()
        ret = await self._run(input, **kw)
        t3 = time.time()
        ret.setdefault('timing', {}).update(
            run_start=t2,
            run_end=t3,
            run_time=t3 - t2,
            from_batch=False,
        )
        self.timing['n_inferences'] += 1
        self.timing['inference_time'] += (t3 - t2)
        return ret

    async def _batched_run(self, input: Any, *, req_start_time: float, **kw) -> dict:
        """Runs the model with auto-batching.

        - input: Input data to process
        - kw: Keyword arguments
        - req_start_time: Start time

        Batches are flushed when full or after a timeout.
        """
        future = asyncio.Future() # type: ignore[var-annotated]
        batch_start_time = time.time()
        async with self.batch_lock:
            req = dict(
                input=input,
                kw=kw,
                future=future,
            )
            self.pending_requests.append(req)
            # start timer if this is the first request
            if len(self.pending_requests) == 1:
                self.batch_timer = asyncio.create_task(self._batch_timeout())
            # process batch if full
            if len(self.pending_requests) >= self.max_batch_size:
                await self._flush_batch()
        ret = await future
        total_time = time.time() - req_start_time
        ret['timing']['total_generate'] = total_time
        ret['timing']['wait_time'] = ret['timing'].get('batch_start', batch_start_time) - req_start_time
        return ret

    async def _batch_timeout(self):
        """Timer to flush partial batches.

        Waits for `max_wait_ms` milliseconds, then flushes any pending requests
        to avoid indefinite waiting for small batches.
        """
        await asyncio.sleep(self.max_wait_ms / 1000)
        async with self.batch_lock:
            if self.pending_requests:
                await self._flush_batch()

    async def _flush_batch(self):
        """Process the current batch of pending requests, handling caching, timing, etc.
        """
        if not self.pending_requests:
            return
        batch = self.pending_requests[:]
        self.pending_requests.clear()
        if self.batch_timer:
            self.batch_timer.cancel()
            self.batch_timer = None
        batch_start_time = time.time()
        results = await self._run_batch(batch)
        batch_end_time = time.time()
        batch_inference_time = batch_end_time - batch_start_time
        # Update global timing stats
        self.timing['n_batch_calls'] += 1
        self.timing['n_batch_inferences'] += len(batch)
        self.timing['inference_time'] += batch_inference_time
        if not results: # there was some error, so mark all as done with an error
            for item in batch:
                if not item['future'].done():
                    item['future'].set_exception(RuntimeError(f'Error during batch processing of model {self.model_name}'))
            return
        # update each item's timing and set the result
        for item, result in zip(batch, results):
            result.setdefault('timing', {}).update(
                batch_start=batch_start_time,
                batch_end=batch_end_time,
                batch_size=len(batch),
                avg_inference_time=batch_inference_time / len(batch),
                from_batch=True,
                found_cache=False,
            )
            self.timing['n_calls'] += 1
            if not item['future'].done():
                item['future'].set_result(result)

    def update_timings(self, ret: dict, *, cache_key:str|None, req_start_time: float) -> None:
        """Updates timing statistics after a cache hit or single model run.

        - ret: Result dict to augment with timing info
        - req_start_time: Start time of the request
        """
        t1 = time.time()
        self.timing['n_calls'] += 1
        self.timing['generate_all'] += t1 - req_start_time
        self.timing['load_elapsed'] = t1 - self.timing.get('load_ts', req_start_time)
        if self.timing['load_elapsed'] > 0:
            self.timing['throughput'] = self.timing['n_inferences'] / self.timing['load_elapsed']
        else:
            self.timing['throughput'] = 0
        self.timing['avg_generate'] = self.timing['generate_all'] / (self.timing['n_inferences']+1)
        ret.setdefault('timing', {}).update(
            self.timing,
            generate=t1-req_start_time,
            found_cache=cache_key and cache_key in self.cache,
        )
        logger.debug(f"Model {self.model_name} run in {t1-req_start_time:.2f}s")


class ChatModel(Model):
    """Base class for chat models.

    Provides common functionality for text generation models:
    - Sets default `max_tokens` parameter
    - Defines cache key based on `max_tokens` and input
    - Includes postprocessing for chat completion format

    Input messages can be provided as:
    - A single string (mapped to a single user message)
    - A list of `Msg` tuples (role, content)
    - A list of dicts in OpenAI API format:
        [{"role": role, "content": content}, ...]
    """
    def update_kw(self, input: Any, **kw) -> dict:
        """Updates keyword arguments with chat model defaults.

        - input: Input messages
        - **kw: Keyword arguments to update

        Sets default `max_tokens` if not provided.
        """
        if 'max_tokens' not in kw or not kw['max_tokens']:
            kw['max_tokens'] = self.max_tokens
        return kw

    async def _get_cache_key(self, input: Any, **kw) -> str:
        return f"{kw['max_tokens']}:{str(input)}"

    def postprocess(self, input, ret, **kw) -> dict:
        """Augments the output with chat completion metadata.

        - input: Original input messages
        - ret: Raw model output
        - **kw: Additional parameters used

        Adds standard OpenAI-compatible fields like model name, timestamp, etc.
        """
        ret['messages'] = input
        ret['model'] = self.model_name.split('/', 1)[-1]
        ret['max_tokens'] = kw['max_tokens']
        ret['object'] = 'chat.completion'
        ret['created'] = int(time.time())
        #print(f'Sending back response: {pformat(ret)}')
        return ret

    @classmethod
    def process_messages(cls, messages: list[Msg]|str) -> list[dict]:
        """Processes messages into OpenAI API format.

        - messages: List of `Msg` tuples or dicts, or a single string

        Converts:
        - Single string -> [{"role": "user", "content": string}]
        - List of (role, content) tuples -> List of {"role": role, "content": content} dicts
        - Passes through existing dict format unchanged
        """
        # if we have a single string -> map to a single user message
        if isinstance(messages, str):
            messages = [('user', messages)]
        # map list of Msg tuples to list of dicts
        fix_msg = lambda m: {'role': m[0], 'content': m[1]} if isinstance(m, tuple) else m
        return [fix_msg(m) for m in messages]


class LocalChatModel(ChatModel):
    """Model subclass for handling local chat models.

    Uses llama-cpp-python to run models locally. Supports GGUF format models.

    Note: This is extremely slow on machines without GPU acceleration.
    Consider using ollama or vllm for better performance.
    """
    async def _load(self, **kw) -> Any:
        from llama_cpp import Llama
        return Llama(
            model_path=self.model_name,
            n_ctx=8192,
            n_threads=8,
            n_gpu_layers=35,
        )

    async def _run(self, input: Any, **kw) -> dict:
        result = self.model(
            input,
            max_tokens=kw['max_tokens'],
            echo=False,
        )
        return self.postprocess(input, result, **kw)


@Singleton
class ExternalChatModel(ChatModel):
    """Model subclass for handling external chat models.

    Routes requests to external API providers (OpenAI, Anthropic, etc.) via the provider system.
    """
    async def _run(self, input: Any, **kw) -> dict:
        logger.debug(f'Running external model: {self.model_name} on input: {input} with kw {kw}')
        if self.model_name.startswith('models/'):
            model = self.model_name.split('/', 1)[-1]
        else:
            model = self.model_name
        kw = kw or {}
        kw['messages'] = self.process_messages(input)
        ret = await call_external(endpoint='/chat/completions',
                                  provider_name=kw.pop('provider', ''),
                                  model=model,
                                  **kw)
        return self.postprocess(input, ret, **kw)


@Singleton
class VLMModel(ChatModel):
    """Model subclass for handling Vision-Language Models.

    Supports multimodal chat with both text and images.
    The input should be `(image, messages)`. Images can be provided as URLs, local paths, or data
    URLs. The messages are processed as in `ChatModel`.
    """
    #TODO figure out image scaling options?
    async def _get_cache_key(self, input: Any, **kw) -> str:
        """Returns cache key based on max_tokens, image, and prompts."""
        image, messages = input
        return f"{kw['max_tokens']}:{image}:{str(messages)}"

    async def _run(self, input: Any, **kw) -> dict:
        logger.warning(f'Running VLM model: {self.model_name} on input: {input} with kw {kw}')
        image, messages = input
        kw = kw or {}
        kw['messages'] = self.process_messages(messages)
        # vlms are run externally and need a data url
        if not image.startswith('http') and not image.startswith('data:'):
            with open(image, 'rb') as f:
                image = data_url_from_file(f)
        # add the image to the first user message
        for msg in kw['messages']:
            if msg['role'] == 'user':
                cur_text = msg['content']
                msg['content'] = [{"type": "text", "text": cur_text},
                                  {"type": "image_url", "image_url": {"url": image}}]
                break
        ret = await call_external(endpoint='/chat/completions',
                                  provider_name=kw.get('provider', ''),
                                  model=self.model_name,
                                  **kw)
        self.postprocess(input, ret, **kw)
        return ret


class EmbeddingModel(Model):
    """Base class for embedding models.

    Provides common functionality for text and image embedding models:
    - Cache key based on input string
    - Postprocessing to OpenAI-compatible embedding format
    """
    def __init__(self, **kw):
        enable_auto_batching = kw.pop('enable_auto_batching', True)
        super().__init__(enable_auto_batching=enable_auto_batching, **kw)

    async def _get_cache_key(self, input: Any, **kw) -> str:
        assert isinstance(input, str)
        return input

    async def _run(self, input: Any, **kw) -> dict:
        """Embedding models run faster in batches, so this just calls that"""
        ret = await self._run_batch([dict(input=input, kw=kw)])
        return ret[0]

    async def _preprocess_single(self, input: Any) -> Any:
        raise NotImplementedError("_preprocess_single must be implemented in subclass")

    def _batch_inference(self, batch_inputs: list[Any]) -> list[Any]:
        raise NotImplementedError("_batch_inference must be implemented in subclass")

    async def _run_batch(self, batch: list[dict]) -> list[dict]:
        """Generic optimized batch processing for embedding models.

        - batch: List of dicts with 'input' and 'kw'

        Behavior:
        - Preprocess all items concurrently; capture per-item exceptions.
        - Run batch inference on successfully preprocessed subset.
        - If batch inference fails, fall back to per-item inference for all remaining items.
        - Always return a results list aligned to the input order, with per-item errors when needed.
        """
        logger.debug(f'Running {self} embeddings batch of size {len(batch)}')
        inputs = [item['input'] for item in batch]
        n = len(inputs)
        t0 = time.time()

        # 1) Preprocess concurrently, capturing exceptions per item.
        tasks = [self._preprocess_single(inp) for inp in inputs]
        preprocessed_or_exc = await asyncio.gather(*tasks, return_exceptions=True)
        t1 = time.time()

        # Prepare results placeholders and collect successful indices
        results: list[dict|None] = [None] * n
        ok_idx: list[int] = []
        ok_items: list[Any] = []
        for i, value in enumerate(preprocessed_or_exc):
            if isinstance(value, Exception):
                # Map preprocess error to per-item result
                results[i] = dict(
                    object='list',
                    data=[],
                    model=self.model_name,
                    n_dims=0,
                    error=str(value),
                    error_type=type(value).__name__,
                )
            else:
                ok_idx.append(i)
                ok_items.append(value)

        # Early exit if nothing to infer
        if not ok_idx:
            # Fill timing for all items (errors only)
            for i in range(n):
                assert results[i] is not None
                results[i].setdefault('timing', {}).update(
                    model=self.model_name,
                    generate=t1 - t0,
                    preprocess_time=(t1 - t0) / max(n, 1),
                    inference_time=0.0,
                )
            return results  # type: ignore[return-value]

        # 2) Try batch inference for successful subset
        embeddings_ok: list[Any] | None = None
        t2_start = time.time()
        try:
            embeddings_ok = await asyncio.to_thread(self._batch_inference, ok_items)
            t2_end = time.time()
            # Map embeddings back to original indices
            for j, orig_i in enumerate(ok_idx):
                embedding = embeddings_ok[j]
                results[orig_i] = self.postprocess(embedding)
        except Exception as e:
            logger.warning(f'Batch inference failed; falling back to per-item inference: {e}')
            # 3) Fallback: per-item inference for all remaining items
            # Note: we use _batch_inference on single-item lists to keep code paths consistent.
            per_item_results: list[tuple[int, dict]] = []
            async def run_single(i: int, item: Any):
                try:
                    emb_single = await asyncio.to_thread(self._batch_inference, [item])
                    return i, self.postprocess(emb_single[0])
                except Exception as ex:
                    return i, dict(
                        object='list',
                        data=[],
                        model=self.model_name,
                        n_dims=0,
                        error=str(ex),
                        error_type=type(ex).__name__,
                    )

            per_item = await asyncio.gather(*[run_single(i, itm) for i, itm in zip(ok_idx, ok_items)])
            t2_end = time.time()
            for i, res in per_item:
                results[i] = res

        # 4) Finalize: add shared timing to all items and return aligned list
        preprocess_time_avg = (t1 - t0) / max(n, 1)
        inference_time_total = max(0.0, t2_end - t2_start)
        for i in range(n):
            assert results[i] is not None
            results[i].setdefault('timing', {}).update(
                model=self.model_name,
                generate=(t2_end - t0),
                preprocess_time=preprocess_time_avg,
                inference_time=inference_time_total,
            )

        logger.debug(f'Processed batch of {n} in {t2_end - t0:.2f}s (ok: {len(ok_idx)}, err: {n - len(ok_idx)})')
        return results  # type: ignore[return-value]

    def postprocess(self, embedding: nparray1d) -> dict:
        """Converts embedding to OpenAI-compatible format.

        Returns dict with OpenAI embeddings API structure.
        """
        return dict(
            object='list',
            data=[dict(object='embedding', index=0, embedding=embedding.tolist())],
            model=self.model_name,
            n_dims=len(embedding),
        )


@Singleton
class MobileNetEmbeddingModel(EmbeddingModel):
    """Model subclass for MobileNet image embeddings."""
    async def _load(self, **kw) -> Any:
        """Loads MobileNetV3 model for image embeddings.

        The model has its classification head removed to output feature embeddings.
        Returns tuple of `(model, preprocess_transform, device)`.
        """
        from torchvision import models, transforms # type: ignore
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.eval()
        # Remove classification head
        # model.features -> conv backbone
        # model.avgpool -> global average pooling
        # model.classifier -> remove
        model.classifier = torch.nn.Identity()
        model.to(device)
        preprocess_image = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        return (model, preprocess_image, device)

    async def _preprocess_single(self, input_data):
        """Preprocesses a single image input asynchronously."""
        _, preprocess_image, _ = self.model
        image = await pil_image_from_input(input_data, force_rgb=True)
        return await asyncio.to_thread(preprocess_image, image)

    def _batch_inference(self, batch_inputs: list[Any]):
        """Runs batch inference on preprocessed image tensors."""
        model, _, device = self.model
        batch_tensor = torch.stack(batch_inputs).to(device)
        with torch.no_grad():
            return model(batch_tensor).cpu().numpy()


class ClipEmbeddingModel(EmbeddingModel):
    """Model subclass for CLIP text or image embeddings"""
    async def _load(self, **kw) -> Any:
        from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoTokenizer # type: ignore
        model = CLIPModel.from_pretrained(self.model_name)
        torch.compiler.is_compiling = lambda: False # type: ignore # temporary hack needed for `use_fast`
        processor = CLIPProcessor.from_pretrained(self.model_name, use_fast=True)
        return (model, processor)

    def feature_func(self, **kw):
        """Returns our model's feature extraction function"""
        raise NotImplementedError("feature_func must be implemented in subclass")

    def _batch_inference(self, batch_inputs: Any):
        batch_dict = {}
        for key in batch_inputs[0]:
            batch_dict[key] = torch.cat([item[key] for item in batch_inputs], dim=0)
        with torch.no_grad():
            return self.feature_func(**batch_dict).cpu().numpy()


@Singleton
class ClipTextEmbeddingModel(ClipEmbeddingModel):
    """Generate CLIP text embeddings"""
    async def _preprocess_single(self, input_data):
        _, processor = self.model
        with torch.no_grad():
            return processor(text=input_data, return_tensors="pt")

    def feature_func(self, **kw):
        model, _ = self.model
        return model.get_text_features(**kw)


@Singleton
class ClipImageEmbeddingModel(ClipEmbeddingModel):
    """Generate CLIP image embeddings"""
    async def _preprocess_single(self, input_data):
        _, processor = self.model
        image = await pil_image_from_input(input_data)
        with torch.no_grad():
            return processor(images=image, return_tensors="pt")

    def feature_func(self, **kw):
        model, _ = self.model
        return model.get_image_features(**kw)


@Singleton
class SentenceTransformerModel(EmbeddingModel):
    """Model subclass for SentenceTransformer text embeddings.

    Uses the sentence-transformers library for high-quality text embeddings.
    Default model is 'BAAI/bge-large-en-v1.5' which performs well on benchmarks.
    """
    async def _load(self, **kw) -> Any:
        from sentence_transformers import SentenceTransformer # type: ignore
        return SentenceTransformer(self.model_name)

    async def _run_batch(self, batch: list[dict]) -> list[dict]:
        inputs = [item['input'] for item in batch]
        embeddings = self.model.encode(inputs, normalize_embeddings=True)
        ret = [self.postprocess(embedding) for embedding in embeddings]
        return ret


@Singleton
class ExternalEmbeddingModel(EmbeddingModel):
    """Model subclass for external API text embeddings.

    Routes embedding requests to external providers (OpenAI, etc.)
    via the provider system.
    """
    def __init__(self, **kw):
        super().__init__(enable_auto_batching=False, **kw)

    async def _run(self, input: Any, **kw) -> dict:
        ret = await call_external(endpoint='/embeddings', provider_name=kw.get('provider', ''), model=self.model_name, input=input)
        ret['input'] = input
        return ret


@Singleton
class TextExtractionModel(Model):
    """Model subclass for extracting text from various sources.

    Supports text extraction from:
    - PDF files
    - Images (via OCR)
    - Web pages
    - Plain text files
    """
    async def _get_cache_key(self, input: Any, **kw) -> str:
        assert isinstance(input, str)
        return input

    async def _run(self, input: Any, **kw) -> dict:
        from nkpylib.ml.text import get_text
        try:
            ret = await asyncio.to_thread(get_text, input, **kw)
            return dict(url=input, text=ret)
        except Exception as e:
            logger.error(f"Error extracting text from {input}: {e}")
            return dict(url=input, text='', error=str(e))


class TranscriptionModel(Model):
    """Base class for speech transcription models.

    Provides common functionality for audio transcription:
    - Cache key based on file hash, language, and chunk level
    - Support for various audio formats
    - Language detection and specification
    """
    async def _get_cache_key(self, input: Any, **kw) -> str:
        with open(input, 'rb') as f:
            sha = sha256(f.read()).hexdigest()
        return f"{sha}:{kw.get('language', 'en')}:{kw.get('chunk_level', 'segment')}"


class LocalTranscriptionModel(TranscriptionModel):
    """Local speech transcription using faster-whisper.

    Runs Whisper models locally using the faster-whisper library
    for improved performance over the original OpenAI implementation.
    """
    async def _load(self, n_threads=12, **kw) -> Any: # type: ignore[override]
        from faster_whisper import WhisperModel # type: ignore
        model_name = 'large-v3'
        logger.debug(f'Loading model {model_name} with {n_threads} threads')
        return WhisperModel(model_name, device="cpu", compute_type='int8', cpu_threads=n_threads)

    async def _run(self, input: Any, language='en', beam_size=5, **kw) -> dict:
        assert isinstance(input, str)
        segments, info = self.model.transcribe(input, beam_size=beam_size, language=language)
        ret = dict(**info._asdict())
        ret['transcription_options'] = info.transcription_options._asdict()
        ret['segments'] = []
        for segment in tqdm(segments):
            await asyncio.sleep(0)  # yield thread
            seg = segment._asdict()
            if seg['words'] is not None:
                seg['words'] = [word._asdict() for word in seg['words']]
            ret['segments'].append(seg)
            logger.debug(f'Processed segment {len(ret["segments"])}: {seg}')
        ret['text'] = ''.join([seg['text'] for seg in ret['segments']])
        return ret


@Singleton
class ExternalTranscriptionModel(TranscriptionModel):
    """External speech transcription via API providers.

    Routes transcription requests to external providers
    that support Whisper or similar models.
    """
    async def _run(self, input: Any, **kw) -> dict:
        logger.debug(f'Running transcription model: {self.model_name} with {input}, {kw}')
        ret = await call_provider(
            provider_name='deepinfra',
            endpoint=f'https://api.deepinfra.com/v1/inference/{self.model_name}',
            files=dict(audio=open(input, 'rb')),
            **kw
        )
        return ret


ALL_SINGLETON_MODELS = [
    ExternalChatModel,
    VLMModel,
    MobileNetEmbeddingModel,
    ClipTextEmbeddingModel,
    ClipImageEmbeddingModel,
    SentenceTransformerModel,
    ExternalEmbeddingModel,
    TextExtractionModel,
    ExternalTranscriptionModel,
]

EMOJI_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
    <text y=".9em" font-size="90">🚀</text>
</svg>
"""

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=EMOJI_SVG, media_type="image/svg+xml")

@app.get("/v1/status")
async def status():
    """Returns server status and model information.

    Provides information about:
    - Loaded models and their cache status
    - Timing statistics for each model instance
    - Current active requests and callers
    - Cache sizes and hit rates
    """
    ret = dict(
        ts=time.time(),
        MODEL_CACHE=[str(k) for k in MODEL_CACHE],
        RESULTS_CACHE={str(k): len(v) for k, v, in RESULTS_CACHE.items()},
        SEMAPHORES={k: sem._value for k, sem in SEMAPHORES.items()},
    )
    for model in ALL_SINGLETON_MODELS:
        instances = model._instances
        if not instances:
            continue
        name = model._cls.__name__
        for key, el in instances.items():
            ret[f'{name}: {key}'] = {
                'class': name,
                'instance_id': key,
                'model_name': el.model_name,
                'loaded': el.model is not None,
                'timing': el.timing,
                'current': list(el.current),
                'callers': dict(el.callers),
            }
    return ret


class BaseRequest(BaseModel):
    """Base request model for all API requests.

    This provides a common structure for requests:
    - model: a string representing the model name to use, or a short string that gets looked up in
      DEFAULT_MODELS
    - use_cache: a boolean indicating whether to use the cache for this request (by default no)
    - q_timeout: if >0, max seconds to wait in the queue before erroring out
    - proc_timeout: if >0, max seconds to allow for processing before erroring out
    - kwargs: a dictionary of additional keyword arguments to pass to the model
      (not actually used by all models)
    - provider: a string representing the external provider to use (e.g. 'deepinfra')
    - caller: a string representing the caller of the API (e.g. 'nkpylib')
    """
    model: str
    use_cache: bool=True
    q_timeout: float=-1
    proc_timeout: float=-1
    kwargs: dict={}
    provider: str=''
    caller: str=''

async def run_with_processing_timeout(coro, timeout: float):
    """Run `coro` with a processing timeout.

    Cancels the root task and all subtasks if timeout occurs.
    Returns the result if completed in time.
    Raises asyncio.TimeoutError if processing timed out.
    """
    loop = asyncio.get_running_loop()
    main_task = loop.create_task(coro)
    try:
        return await asyncio.wait_for(main_task, timeout=timeout)
    except asyncio.TimeoutError:
        # Cancel the main task (all subtasks inherit the cancellation)
        main_task.cancel()
        try:
            await main_task
        except asyncio.CancelledError:
            pass  # expected
        raise

def concurrency_endpoint(tier: str):
    """Decorator for FastAPI endpoints to handle concurrency with semaphores and timeouts.

    - tier: A string representing the concurrency tier

    We read the following from the request object from the function call
    - q_timeout: Timeout for waiting to acquire the semaphore
    - proc_timeout: Timeout for processing the request after acquiring the semaphore
    """
    sem = SEMAPHORES[tier]
    def decorator(func: Callable[[BaseRequest], asyncio.Future]):
        @wraps(func)
        async def wrapper(req: BaseRequest):
            q_timeout = getattr(req, 'q_timeout', -1)
            proc_timeout = getattr(req, 'proc_timeout', -1)
            if q_timeout <= 0:
                q_timeout = 100000000 # effectively infinite
            if proc_timeout <= 0:
                proc_timeout = 100000000
            async def gen():
                # --- queue admission with timeout ---
                try:
                    await asyncio.wait_for(sem.acquire(), timeout=q_timeout)
                except asyncio.TimeoutError:
                    raise HTTPException(status_code=429, detail="queue timeout")
                try:
                    # --- processing starts: emit sentinel ---
                    #yield b"\n"
                    # --- execute the actual endpoint logic with processing timeout ---
                    result = await run_with_processing_timeout(func(req), timeout=proc_timeout)
                    return result
                    #yield json.dumps(result).encode("utf-8")
                except asyncio.TimeoutError:
                    raise HTTPException(status_code=503, detail="processing timeout")
                finally:
                    sem.release()

            # for streaming:
            #return StreamingResponse(gen(), media_type="application/json")
            # for non-streaming:
            result = await gen()
            return result


        return wrapper
    return decorator

# setup fastapi chat endpoint
class ChatRequest(BaseRequest):
    messages: str|list[Msg]|list[dict[str,str]]
    model: str='chat'
    max_tokens: int=0


async def chat_impl(req: ChatRequest):
    """Implementation for chat completion endpoints.

    - req: `ChatRequest` containing messages and parameters

    Routes to either local or external chat models based on model name.
    Supports both `/v1/chat` and `/v1/chat/completions` endpoints.
    """
    logger.debug(f'running chat model')
    # note that we only need to look up the default model to know whether to use local or external
    if req.model in DEFAULT_MODELS:
        req.model = DEFAULT_MODELS[req.model].name
    model: ChatModel
    if req.model in LOCAL_MODELS:
        model = LocalChatModel(model_name=req.model, device='cpu', use_cache=req.use_cache)
    else:
        model = ExternalChatModel(model_name=req.model, use_cache=req.use_cache)
    ret = await model.run(
        input=req.messages,
        max_tokens=req.max_tokens,
        provider=req.provider,
        caller=req.caller,
        **req.kwargs
    )
    return ret

@app.post("/v1/chat")
@concurrency_endpoint(tier='extfast')
async def chat(req: ChatRequest):
    """Chat completion endpoint.

    - req: `ChatRequest` with messages and model parameters

    Returns OpenAI-compatible chat completion response.
    """
    return await chat_impl(req)

@app.post("/v1/chat/completions")
@concurrency_endpoint(tier='extfast')
async def chat_completion(req: ChatRequest):
    """OpenAI-compatible chat completions endpoint.

    - req: `ChatRequest` with messages and model parameters

    Returns OpenAI-compatible chat completion response.
    """
    return await chat_impl(req)

# setup fastapi VLM endpoint
class VLMRequest(BaseRequest):
    image: str # path/url/image directly?
    messages: str|list[Msg]|list[dict[str,str]]
    model: str='vlm' # this will get mapped based on DEFAULT_MODELS['vlm']
    max_tokens: int=0


@app.post("/v1/vlm")
@concurrency_endpoint(tier='extslow')
async def vlm(req: VLMRequest):
    """Vision-Language Model endpoint for multimodal chat.

    - req: `VLMRequest` with image, messages, and model parameters

    Processes both image and text inputs to generate contextual responses.
    Images can be URLs, local paths, or data URLs.
    """
    # note that we don't need to look up the default model, since it's always external
    model = VLMModel(model_name=req.model, use_cache=req.use_cache)
    print(f'Running VLM model {req.model} on image {req.image}')
    #print(f'Running VLM model {req.model} on image {req.image} and messages {req.messages}')
    ret = await model.run(
        input=(req.image, req.messages),
        max_tokens=req.max_tokens,
        provider=req.provider,
        caller=req.caller,
        **req.kwargs
    )
    return ret


class TextEmbeddingRequest(BaseRequest):
    input: str
    model: str=DEFAULT_MODELS['st'].name


@app.post("/v1/embeddings")
@concurrency_endpoint(tier='medium')
async def text_embeddings(req: TextEmbeddingRequest):
    """Text embedding endpoint.

    - req: `TextEmbeddingRequest` with input text and model name

    Returns OpenAI-compatible embedding response with vector data.
    Supports CLIP, SentenceTransformers, and external models.
    """
    req.model = _default(req.model)
    model_class_by_name = {
        _default('clip'): lambda **kw: ClipTextEmbeddingModel(**kw),
        _default('sentence'): SentenceTransformerModel,
    }
    ModelClass: Any = model_class_by_name.get(req.model, ExternalEmbeddingModel)
    model = ModelClass(model_name=req.model, use_cache=req.use_cache)
    ret = await model.run(input=req.input, provider=req.provider, caller=req.caller)
    return ret


class ImageEmbeddingRequest(BaseRequest):
    url: str
    model: str=_default('image')


@app.post("/v1/image_embeddings")
@concurrency_endpoint(tier='fast')
async def image_embeddings(req: ImageEmbeddingRequest):
    """Image embedding endpoint.

    - req: `ImageEmbeddingRequest` with image URL/path and model name

    Returns embedding response with vector data for the input image.
    Supports CLIP and MobileNet models.
    """
    req.model = _default(req.model)
    Cls: type[EmbeddingModel]
    if req.model == _default('clip'):
        Cls = ClipImageEmbeddingModel
    elif req.model == _default('mobilenet'):
        Cls = MobileNetEmbeddingModel
    else:
        raise NotImplementedError(f"Model {req.model} not supported for image embeddings")
    model = Cls(model_name=req.model, use_cache=req.use_cache)
    ret = await model.run(input=req.url, caller=req.caller)
    return ret


class StrSimRequest(BaseRequest):
    a: str
    b: str
    model: str=_default('clip')

@app.post("/v1/strsim")
@concurrency_endpoint(tier='medium')
async def strsim(req: StrSimRequest):
    """String similarity endpoint using embedding cosine similarity.

    - req: `StrSimRequest` with two strings and model name

    Embeds both strings and computes cosine similarity.
    Returns similarity score (higher = more similar) and timing information.
    """
    req.model = _default(req.model)
    # embed both texts by calling the text_embeddings function
    timings: dict[str, float] = {}
    async def call(s):
        ret = await text_embeddings(TextEmbeddingRequest(input=s, model=req.model, use_cache=req.use_cache))
        # update timings by adding all numbers, and taking "or" of boolean values
        for k, v in ret['timing'].items():
            if isinstance(v, bool):
                timings[k] = timings.get(k, False) or v
            elif isinstance(v, (int, float)):
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


class GetTextRequest(BaseRequest):
    url: str
    model: str='get_text' # this is not actually used right now

@app.post("/v1/get_text")
@concurrency_endpoint(tier='medium')
async def get_text_api(req: GetTextRequest):
    """Text extraction endpoint.

    - req: `GetTextRequest` with URL or file path

    Extracts text from various sources including PDFs, images, and web pages.
    Returns extracted text or error information.
    """
    model = TextExtractionModel(model_name='text', use_cache=req.use_cache)
    ret = await model.run(input=req.url, caller=req.caller, **(req.kwargs or {}))
    return ret


class TranscriptionRequest(BaseRequest):
    url: str # path/url to file or base64 encoded audio bytes
    model: str='transcription' # this will get mapped based on DEFAULT_MODELS['Transcription']
    language: str='en'
    chunk_level: str='segment'
    provider: str='deepinfra'

@app.post("/v1/transcription")
@concurrency_endpoint(tier='slow')
async def speech_transcription(req: TranscriptionRequest):
    """Speech transcription endpoint.

    - req: `TranscriptionRequest` with audio URL/path and parameters

    Transcribes audio to text using Whisper models (local or external).
    Supports language specification and different chunk levels.
    """
    ModelClass = LocalTranscriptionModel if req.model == 'local-transcription' else ExternalTranscriptionModel
    logger.info(f'In speech transcription, got model {req.model} and cls {ModelClass}')
    model = ModelClass(model_name=req.model, use_cache=req.use_cache)
    async with dl_temp_file(req.url) as path:
        kw = req.kwargs or {}
        kw.update(
            language=req.language,
            chunk_level=req.chunk_level,
            provider=req.provider,
        )
        ret = await model.run(input=path, caller=req.caller, **kw)
    return ret


@app.get("/test")
async def test_api():
    """Simple test endpoint for server health checks.

    Returns a basic "Hello world" response to verify the server is running.
    """
    print(f'got request for test: {time.time()}')
    #await asyncio.sleep(10)
    return "Hello world\n"


def cleanup_executors():
    """Clean up all process executors on server shutdown.

    Shuts down all tracked `ProcessPoolExecutor` instances to prevent
    hanging processes when the server restarts or shuts down.
    """
    print(f'cleaning up executors')
    for executor in _EXECUTORS:
        try:
            executor.shutdown(wait=False)
        except Exception as e:
            logger.debug(f"Error during executor cleanup: {e}")
    _EXECUTORS.clear()

# Register cleanup for multiple signals
atexit.register(cleanup_executors)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
