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
  - `jina`: The jina-clip-v2 model, which can embed both text and images. This is a better version
    of CLIP.
- String similarity:
  - This embeds two strings using any of the text embedding models, and then computes the cosine
    similarity between the two embeddings. Higher is more similar.
- Image embeddings:
  - `clip`: The OpenAI CLIP model, which can embed both text and images (in the same space).
  - `jina`: The jina-clip-v2 model, which can embed both text and images (in the same space). Prefer
    this to clip, except that it's quite a bit slower.
"""

from __future__ import annotations

#TODO have some kind of context manager for deciding where to run llm functions from? (local, replicate, openai)
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
from hashlib import sha256
from pprint import pformat
from asyncio import Lock, Condition
from typing import Any, Callable, Literal, Optional, Union
from urllib.request import urlretrieve

import anyio
import fastapi
import numpy as np
import requests

from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm

from nkpylib.ml.constants import DEFAULT_MAX_TOKENS, DEFAULT_MODELS, LOCAL_MODELS, Role, Msg, data_url_from_file
from nkpylib.ml.providers import call_external, call_provider
from nkpylib.ml.text import get_text
from nkpylib.thread_utils import Singleton
from nkpylib.web_utils import make_request_async, dl_temp_file
from nkpylib.utils import is_instance_of_type

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # Startup
    yield
    # Shutdown - this will be called on reload
    cleanup_executors()

app = fastapi.FastAPI(lifespan=lifespan)

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
    """Returns the default model full name"""
    if name not in DEFAULT_MODELS:
        return name
    return DEFAULT_MODELS[name].name

@functools.cache
def load_clip(model_name=_default('clip')):
    """Loads clip and returns two embedding functions: one for text, one for images"""
    import torch
    from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoTokenizer # type: ignore
    model = CLIPModel.from_pretrained(model_name)
    torch.compiler.is_compiling = lambda: False
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)

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

@functools.cache
def load_jina(model_name=_default('jina'), dims:int =DEFAULT_MODELS['jina'].default_dims):
    """Loads jina and returns two embedding functions: one for text, one for images.

    Based on the research paper, going from the max dims of 1024 to 768 doesn't hurt performance at
    all (and might even slightly improve it for some tasks).
    """
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
    """Base class for models, providing a common interface for loading and running models."""
    def __init__(self,
                 model_name: str='',
                 use_cache: bool=True,
                 enable_auto_batching: bool=False,
                 max_batch_size: int=4,
                 max_wait_ms: float=50,
                 **kw):
        self.model_cfg = DEFAULT_MODELS.get(model_name)
        if self.model_cfg:
            model_name = self.model_cfg.name
            self.max_tokens = self.model_cfg.max_tokens
        else:
            self.max_tokens = DEFAULT_MAX_TOKENS
        self.model_name = model_name
        self.lock = Lock()
        self.condition = Condition()
        self.use_cache = use_cache
        self.model: Any = None
        self.cache = RESULTS_CACHE.setdefault(self.__class__, OrderedDict())
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


    async def _load(self, **kw) -> Any:
        """Load implementation.

        This version just returns the model name as the model itself (useful for external APIs).
        """
        return self.model_name

    async def load(self, **kw) -> bool:
        """Loads our model if not already loaded, returning if we actually loaded it"""
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
        """Updates the `kw` input dict with default parameters, etc."""
        return kw

    @abstractmethod
    async def _get_cache_key(self, input: Any, **kw) -> str:
        """Returns the cache key for a given input and kw. Override this in your subclass"""
        ...

    @abstractmethod
    async def _run(self, input: Any, **kw) -> dict:
        """Run implementation. Override this in your subclass"""
        ...

    async def run(self, input: Any, caller: str='', **kw) -> dict:
        """Runs the model with given `input` - automatically batches if enabled"""
        if self.enable_auto_batching:
            return await self._auto_batched_run(input, caller, **kw)
        else:
            return await self._original_run(input, caller, **kw)
    
    async def _original_run(self, input: Any, caller: str='', **kw) -> dict:
        """Original run implementation"""
        t0 = time.time()
        cache_key = None
        kw = self.update_kw(input, **kw)
        self.callers[caller] += 1
        if self.use_cache or self.model is None:
            cache_key, self.timing['did_load'] = await asyncio.gather(
                self._get_cache_key(input, **kw) if self.use_cache else asyncio.sleep(0),
                self.load(**kw) if self.model is None else asyncio.sleep(0)
            )
        if cache_key and cache_key in self.cache:
            ret = self.cache[cache_key].copy()
            self.timing['n_cache_hits'] += 1
            self.cache.move_to_end(cache_key)  # move to end to keep it fresh
        else:
            if cache_key is not None:
                if cache_key in self.current:
                    async with self.condition:
                        await self.condition.wait_for(lambda: cache_key not in self.current)
                self.timing['n_cache_misses'] += 1
                self.current.add(cache_key)
            t2 = time.time()
            ret = await self._run(input, **kw)
            t3 = time.time()
            self.timing['n_inferences'] += 1
            self.timing['inference_time'] += (t3 - t2)
            if cache_key is not None:
                self.cache[cache_key] = ret.copy()
                while len(self.cache) > RESULTS_CACHE_LIMIT:
                    self.cache.popitem(last=False)
                self.current.remove(cache_key)
                async with self.condition:
                    self.condition.notify_all()
        t1 = time.time()
        self.callers[caller] -= 1
        self.timing['n_calls'] += 1
        self.timing['generate_all'] += t1 - t0
        self.timing['load_elapsed'] = t1 - self.timing.get('load_ts', t0)
        self.timing['throughput'] = self.timing['n_inferences'] / self.timing['load_elapsed'] if self.timing['load_elapsed'] > 0 else 0
        self.timing['avg_generate'] = self.timing['generate_all'] / (self.timing['n_inferences']+1)
        ret['timing'] = dict(self.timing, generate=t1-t0, found_cache=cache_key and cache_key in self.cache)
        logger.debug(f"Model {self.model_name} run in {t1-t0:.2f}s")
        return ret
    
    async def _auto_batched_run(self, input: Any, caller: str='', **kw) -> dict:
        """Auto-batching version of run()"""
        t0 = time.time()
        kw = self.update_kw(input, **kw)
        
        # Check cache first (preserve caching behavior)
        cache_key = None
        if self.use_cache:
            cache_key = await self._get_cache_key(input, **kw)
            if cache_key in self.cache:
                ret = self.cache[cache_key].copy()
                ret['timing'] = dict(self.timing, generate=time.time() - t0, found_cache=True, from_batch=False)
                self.timing['n_cache_hits'] += 1
                self.callers[caller] += 1
                self.cache.move_to_end(cache_key)
                return ret
        
        # Ensure model is loaded
        await self.load(**kw)
        
        # Batch uncached items
        future = asyncio.Future()
        batch_start_time = time.time()
        
        async with self.batch_lock:
            self.pending_requests.append((input, caller, kw, future, cache_key, batch_start_time))
            
            # Start timer if this is the first request
            if len(self.pending_requests) == 1:
                self.batch_timer = asyncio.create_task(self._batch_timeout())
            
            # Process batch if full
            if len(self.pending_requests) >= self.max_batch_size:
                await self._flush_batch()
        
        result = await future
        
        # Add individual timing
        total_time = time.time() - t0
        result['timing']['total_generate'] = total_time
        result['timing']['wait_time'] = result['timing'].get('batch_start', batch_start_time) - t0
        
        return result
    
    async def _batch_timeout(self):
        """Timer to flush partial batches"""
        await asyncio.sleep(self.max_wait_ms / 1000)
        async with self.batch_lock:
            if self.pending_requests:
                await self._flush_batch()
    
    async def _flush_batch(self):
        """Process the current batch"""
        if not self.pending_requests:
            return
            
        batch = self.pending_requests[:]
        self.pending_requests.clear()
        if self.batch_timer:
            self.batch_timer.cancel()
            self.batch_timer = None
        
        batch_start_time = time.time()
        
        # Use optimized batch processing if available
        try:
            if hasattr(self, '_run_batch_optimized'):
                results = await self._run_batch_optimized(batch)
            else:
                # Fallback: process individually but concurrently
                results = await self._process_batch_individually(batch)
        except Exception as e:
            logger.warning(f"Batch processing failed: {e}, falling back to individual")
            results = await self._process_batch_individually(batch)
        
        batch_end_time = time.time()
        batch_inference_time = batch_end_time - batch_start_time
        
        # Update global timing stats
        self.timing['n_batch_calls'] += 1
        self.timing['n_batch_inferences'] += len(batch)
        self.timing['inference_time'] += batch_inference_time
        
        # Cache and distribute results with proper timing
        for (input_data, caller, kw, future, cache_key, start_time), result in zip(batch, results):
            # Cache result
            if cache_key and self.use_cache:
                self.cache[cache_key] = result.copy()
                while len(self.cache) > RESULTS_CACHE_LIMIT:
                    self.cache.popitem(last=False)
            
            # Add timing metadata
            result['timing'].update(dict(
                batch_start=batch_start_time,
                batch_end=batch_end_time,
                batch_size=len(batch),
                batch_inference_time=batch_inference_time / len(batch),
                from_batch=True,
                found_cache=False,
            ))
            
            self.callers[caller] += 1
            self.timing['n_calls'] += 1
            if not future.done():
                future.set_result(result)
    
    async def _process_batch_individually(self, batch):
        """Fallback: process batch items individually but concurrently"""
        async def process_single(input_data, caller, kw, future, cache_key, start_time):
            return await self._run(input_data, **kw)
        
        tasks = [process_single(*item) for item in batch]
        return await asyncio.gather(*tasks)


class ChatModel(Model):
    """Base class for chat models.

    This currently just sets a default value for `max_tokens`, and defines the cache key as
    the `max_tokens` and the input string(s).
    """
    def update_kw(self, input: Any, **kw) -> dict:
        """Updates the `kw` input dict with default parameters, etc."""
        if 'max_tokens' not in kw or not kw['max_tokens']:
            kw['max_tokens'] = self.max_tokens
        return kw

    async def _get_cache_key(self, input: Any, **kw) -> str:
        return f"{kw['max_tokens']}:{str(input)}"

    def postprocess(self, input, ret, **kw) -> dict:
        """Augments the output with additional information."""
        ret['messages'] = input
        ret['model'] = self.model_name.split('/', 1)[-1]
        ret['max_tokens'] = kw['max_tokens']
        ret['object'] = 'chat.completion'
        ret['created'] = int(time.time())
        #print(f'Sending back response: {pformat(ret)}')
        return ret


class LocalChatModel(ChatModel):
    """Model subclass for handling local chat models.

    This runs llama locally using llama-cpp-python.

    Note this is extremely slow on my current machine (mini-pc, no gpu), and in general I think
    people recommend using ollama or vllm instead these days.
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
    """Model subclass for handling external chat models."""
    async def _run(self, input: Any, **kw) -> dict:
        logger.debug(f'Running external model: {self.model_name} on input: {input} with kw {kw}')
        if self.model_name.startswith('models/'):
            model = self.model_name.split('/', 1)[-1]
        else:
            model = self.model_name
        kw = kw or {}
        kw['messages'] = process_messages(input)
        ret = await call_external(endpoint='/chat/completions', provider_name=kw.pop('provider', ''), model=model, **kw)
        return self.postprocess(input, ret, **kw)

def process_messages(messages: list[Msg]) -> list[dict]:
    """Processes a list of Msg tuples into the format expected by the API."""
    # if we have a single string -> map to a single user message
    if isinstance(messages, str):
        messages = [('user', messages)]
    # map list of Msg tuples to list of dicts
    fix_msg = lambda m: {'role': m[0], 'content': m[1]} if isinstance(m, tuple) else m
    return [fix_msg(m) for m in messages]


@Singleton
class VLMModel(ChatModel):
    """Model subclass for handling VLM models."""
    async def _get_cache_key(self, input: Any, **kw) -> str:
        image, prompts = input
        return f"{kw['max_tokens']}:{image}:{str(prompts)}"

    async def _run(self, input: Any, **kw) -> dict:
        logger.debug(f'Running VLM model: {self.model_name} on input: {input} with kw {kw}')
        image, messages = input
        kw = kw or {}
        kw['messages'] = process_messages(messages)
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
        ret = await call_external(endpoint='/chat/completions', provider_name=kw.get('provider', ''), model=self.model_name, **kw)
        self.postprocess(input, ret, **kw)
        return ret


class EmbeddingModel(Model):
    """Base class for text embeddings.

    This includes a postprocess() function.
    """
    async def _get_cache_key(self, input: Any, **kw) -> str:
        assert isinstance(input, str)
        return input

    def postprocess(self, embedding) -> dict:
        return dict(
            object='list',
            data=[dict(
                object='embedding',
                index=0,
                embedding=embedding.tolist(),
            )],
            model=self.model_name,
            n_dims=len(embedding),
        )

def load_mobilenet():
    """Loads mobilenetv2 and returns (model, preprocess_image, device)"""
    import torch
    from torchvision import models, transforms
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

def run_mobilenet(input, model, preprocess_image, device: str) -> np.ndarray:
    import torch
    times = [time.time()]
    if isinstance(input, str):
        if input.startswith('http'):
            image = Image.open(requests.get(input, stream=True).raw).convert('RGB')
        else:
            image = Image.open(input).convert('RGB')
    else:
        image = input.convert('RGB')
    times.append(time.time())
    input_tensor = preprocess_image(image).unsqueeze(0).to(device)  # create a mini-batch as expected by the model
    times.append(time.time())
    with torch.no_grad():
        embedding = model(input_tensor).squeeze(0).cpu().numpy()
    times.append(time.time())
    #print(f'Mobilenet embedding run in {[t1-t0 for t0, t1 in zip(times, times[1:])]}')
    return embedding


def image_text_embedding_worker(model_name: str, mode: str, input_data: Any):
    """Class method that handles both loading and execution"""
    global PROC_MODELS
    times = [time.time()]
    if model_name not in PROC_MODELS:
        logger.info(f"Loading {model_name} in process {os.getpid()}")
        if 'clip' in model_name:
            PROC_MODELS[model_name] = load_clip(model_name)
        elif 'jina' in model_name:
            PROC_MODELS[model_name] = load_jina(model_name)
        elif 'mobilenet' in model_name:
            PROC_MODELS[model_name] = load_mobilenet()
            assert mode == 'image', "MobileNet only supports image embeddings"
        else:
            raise NotImplementedError(f"Unsupported ImageTextEmbeddingModel: {model_name}")
    times.append(time.time())
    if 'mobilenet' in model_name:
        # special case for mobilenet
        model, preprocess_image, device = PROC_MODELS[model_name]
        embedding = run_mobilenet(input_data, model, preprocess_image, device)
        ret = embedding
    else:
        text_func, image_func = PROC_MODELS[model_name]
        func = text_func if mode == 'text' else image_func
        ret = func(input_data)
    times.append(time.time())
    #logger.info(f"Process {os.getpid()} ran {model_name} ({mode}) in {[t1-t0 for t0, t1 in zip(times, times[1:])]}")
    #print(f"Process {os.getpid()} ran {model_name} ({mode}) in {[t1-t0 for t0, t1 in zip(times, times[1:])]}")
    return ret


@Singleton
class OldMobileNetEmbeddingModel(EmbeddingModel):
    """Model subclass for handling MobileNet image embeddings."""
    def __init__(self, enable_auto_batching: bool=False, **kw):
        super().__init__(enable_auto_batching=enable_auto_batching, **kw)
    
    async def _load(self, **kw) -> Any:
        return load_mobilenet()

    async def _run(self, input: Any, **kw) -> dict:
        model, preprocess_image, device = self.model
        embedding = run_mobilenet(input, model, preprocess_image, device)
        return self.postprocess(embedding)
    
    async def _run_batch_optimized(self, batch):
        """Optimized batch processing for MobileNet"""
        import torch
        
        # Extract inputs and metadata
        inputs = [item[0] for item in batch]
        
        # Parallel preprocessing
        async def preprocess_single(input_data):
            model, preprocess_image, device = self.model
            
            def sync_preprocess():
                if isinstance(input_data, str):
                    if input_data.startswith('http'):
                        image = Image.open(requests.get(input_data, stream=True).raw).convert('RGB')
                    else:
                        image = Image.open(input_data).convert('RGB')
                else:
                    image = input_data.convert('RGB')
                return preprocess_image(image)
            
            return await asyncio.to_thread(sync_preprocess)
        
        # Preprocess all images concurrently
        t0 = time.time()
        preprocess_tasks = [preprocess_single(inp) for inp in inputs]
        preprocessed_tensors = await asyncio.gather(*preprocess_tasks)
        t1 = time.time()
        
        # Batch inference
        model, _, device = self.model
        
        def batch_inference():
            batch_tensor = torch.stack(preprocessed_tensors).to(device)
            with torch.no_grad():
                return model(batch_tensor).cpu().numpy()
        
        batch_embeddings = await asyncio.to_thread(batch_inference)
        t2 = time.time()
        # Create individual results with proper timing/metadata
        results = []
        for i, input_data in enumerate(inputs):
            embedding = batch_embeddings[i]
            result = self.postprocess(embedding)
            # Add timing info (shared across batch)
            result['timing'] = dict(
                model=self.model_name,
                generate=t2 - t0,
                preprocess_time=(t1 - t0) / len(inputs),
                inference_time=(t2 - t1),
                batch_size=len(inputs),
            )
            results.append(result)
        logger.debug(f"Processed batch of {len(inputs)} in {t2-t0:.2f}s")
        return results


class ImageTextEmbeddingModel(EmbeddingModel):
    """Model subclass for handling joint text/image embeddings."""
    def __init__(self, mode='text', model_name: str='', use_cache: bool=True, n_procs: int=8, **kw):
        super().__init__(model_name=model_name, use_cache=use_cache, **kw)
        assert mode in ('text', 'image')
        self.mode = mode
        self.executor = ProcessPoolExecutor(max_workers=n_procs)
        #self.executor = ThreadPoolExecutor(max_workers=n_procs)
        _EXECUTORS.append(self.executor)  # Track for cleanup

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

    async def load_feature_funcs(self):
        """Returns two functions: one for text features, one for image features."""
        raise NotImplementedError("This method should be overridden in subclasses")

    async def _load(self, **kw) -> Any:
        return None #TODO remove following code once we test process version
        get_text_features, get_image_features = await self.load_feature_funcs()
        if self.mode == 'text':
            return get_text_features
        elif self.mode == 'image':
            return get_image_features
        raise NotImplementedError(f"Unsupported mode: {self.mode}")

    async def _run(self, input: Any, **kw) -> dict:
        try:
            times = [time.time()]
            loop = asyncio.get_event_loop()
            times.append(time.time())
            ret = await loop.run_in_executor(
                self.executor,
                image_text_embedding_worker,
                self.model_name,
                self.mode,
                input
            )
            times.append(time.time())
            # Use thread-based execution (original behavior)
            #ret = await asyncio.to_thread(self.model, input)
            if not isinstance(ret, np.ndarray):
                ret = ret.numpy()
            times.append(time.time())
            ret = self.postprocess(ret)
            times.append(time.time())
            #logger.warning(f"Ran {self.model_name} ({self.mode}) in {[t1-t0 for t0, t1 in zip(times, times[1:])]}")
            return ret
        except Exception as e:
            logger.error(f"Error running {self.model_name} embedding model: {e}")
            return {}

@Singleton
class MobileNetEmbeddingModel(ImageTextEmbeddingModel):
    """Model subclass for handling MobileNet image embeddings."""
    def __init__(self, model_name: str='', use_cache: bool=True, n_procs: int=8, **kw):
        super().__init__(mode='image', model_name=model_name, use_cache=use_cache, n_procs=n_procs, **kw)

    async def load_feature_funcs(self):
        """Returns two functions: one for text features, one for image features."""
        raise NotImplementedError()

@Singleton
class ClipTextEmbeddingModel(ImageTextEmbeddingModel):
    """Model subclass for handling CLIP text embeddings."""
    def __init__(self, model_name: str='', use_cache: bool=True, n_procs: int=1, **kw):
        super().__init__(mode='text', model_name=model_name, use_cache=use_cache, n_procs=n_procs, **kw)

    async def load_feature_funcs(self):
        """Returns two functions: one for text features, one for image features."""
        return load_clip()

@Singleton
class ClipImageEmbeddingModel(ImageTextEmbeddingModel):
    """Model subclass for handling CLIP text/image embeddings."""
    def __init__(self, model_name: str='', use_cache: bool=True, n_procs: int=1, **kw):
        super().__init__(mode='image', model_name=model_name, use_cache=use_cache, n_procs=n_procs, **kw)

    async def load_feature_funcs(self):
        """Returns two functions: one for text features, one for image features."""
        return load_clip()


@Singleton
class JinaTextEmbeddingModel(ImageTextEmbeddingModel):
    """Model subclass for handling Jina text embeddings."""
    def __init__(self, model_name: str='', use_cache: bool=True, n_procs: int=8, **kw):
        super().__init__(mode='text', model_name=model_name, use_cache=use_cache, n_procs=n_procs, **kw)

    async def load_feature_funcs(self):
        """Returns two functions: one for text features, one for image features."""
        return load_jina()


@Singleton
class JinaImageEmbeddingModel(ImageTextEmbeddingModel):
    """Model subclass for handling Jina image embeddings."""
    def __init__(self, model_name: str='', use_cache: bool=True, n_procs: int=8, **kw):
        super().__init__(mode='image', model_name=model_name, use_cache=use_cache, n_procs=n_procs, **kw)

    async def load_feature_funcs(self):
        """Returns two functions: one for text features, one for image features."""
        return load_jina()


@Singleton
class SentenceTransformerModel(EmbeddingModel):
    """Model subclass for handling SentenceTransformer embeddings."""
    async def _load(self, **kw) -> Any:
        from sentence_transformers import SentenceTransformer # type: ignore
        return SentenceTransformer(self.model_name)

    async def _run(self, input: Any, **kw) -> dict:
        embedding = self.model.encode([input], normalize_embeddings=True)[0]
        return self.postprocess(embedding)


@Singleton
class ExternalEmbeddingModel(EmbeddingModel):
    """Model subclass for handling external API text embeddings."""
    async def _run(self, input: Any, **kw) -> dict:
        ret = await call_external(endpoint='/embeddings', provider_name=kw.get('provider', ''), model=self.model_name, input=input)
        ret['input'] = input
        return ret


@Singleton
class TextExtractionModel(Model):
    """Model subclass for extracting text from URLs or file paths."""
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
    """Base class for transcription models.

    This checks that the input is a valid path, and uses the hash of the file as the cache key.
    It also includes the language (default 'en') and 'chunk_level' (default 'segment') in the cache
    key.
    """
    async def _get_cache_key(self, input: Any, **kw) -> str:
        with open(input, 'rb') as f:
            sha = sha256(f.read()).hexdigest()
        return f"{sha}:{kw.get('language', 'en')}:{kw.get('chunk_level', 'segment')}"


class LocalTranscriptionModel(TranscriptionModel):
    """A local transcription model using faster-whisper."""
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
    """Model subclass for handling speech transcription."""
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
    ClipTextEmbeddingModel,
    ClipImageEmbeddingModel,
    JinaTextEmbeddingModel,
    JinaImageEmbeddingModel,
    SentenceTransformerModel,
    ExternalEmbeddingModel,
    TextExtractionModel,
    ExternalTranscriptionModel,
]

@app.get("/v1/status")
async def status():
    """Returns various kinds of status"""
    ret = dict(
        ts=time.time(),
        MODEL_CACHE=[str(k) for k in MODEL_CACHE],
        RESULTS_CACHE={str(k): len(v) for k, v, in RESULTS_CACHE.items()},
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
    - kwargs: a dictionary of additional keyword arguments to pass to the model
      (not actually used by all models)
    - provider: a string representing the external provider to use (e.g. 'deepinfra')
    - caller: a string representing the caller of the API (e.g. 'nkpylib')
    """
    model: str
    use_cache: bool=False
    kwargs: dict={}
    provider: str=''
    caller: str=''

# setup fastapi chat endpoint
class ChatRequest(BaseRequest):
    messages: str|list[Msg]|list[dict[str,str]]
    model: str='chat' # this will get mapped based on DEFAULT_MODELS['chat']
    max_tokens: int=0



async def chat_impl(req: ChatRequest):
    """Generates chat response for the given messages using the given model."""
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
async def chat(req: ChatRequest):
    """Generates chat response for the given messages using the given model."""
    return await chat_impl(req)

@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest):
    """Generates chat response for the given messages using the given model."""
    return await chat_impl(req)

# setup fastapi VLM endpoint
class VLMRequest(BaseRequest):
    image: str # path/url/image directly?
    messages: str|list[Msg]|list[dict[str,str]]
    model: str='vlm' # this will get mapped based on DEFAULT_MODELS['vlm']
    max_tokens: int=0


@app.post("/v1/vlm")
async def vlm(req: VLMRequest):
    """Generates VLM chat response for the given image and messages using the given model."""
    # note that we don't need to look up the default model, since it's always external
    model = VLMModel(model_name=req.model, use_cache=req.use_cache)
    print(f'Running VLM model {req.model} on image {req.image} and messages {req.messages}')
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
    model: str=DEFAULT_MODELS['st']


@app.post("/v1/embeddings")
async def text_embeddings(req: TextEmbeddingRequest):
    """Generates embeddings for the given text using the given model."""
    req.model = _default(req.model)
    model_class_by_name = {
        _default('clip'): lambda **kw: ClipTextEmbeddingModel(**kw),
        _default('jina'): lambda **kw: JinaTextEmbeddingModel(**kw),
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
async def image_embeddings(req: ImageEmbeddingRequest):
    """Generates embeddings for the given image url (or local path) using the given model."""
    req.model = _default(req.model)
    if req.model == _default('clip'):
        Cls = ClipImageEmbeddingModel
    elif req.model == _default('jina'):
        Cls = JinaImageEmbeddingModel
    elif req.model == _default('mobilenet'):
        Cls = MobileNetEmbeddingModel
    elif req.model == _default('oldmobilenet'):
        Cls = OldMobileNetEmbeddingModel
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
async def strsim(req: StrSimRequest):
    """Computes the strsim between `a` and `b` (higher is more similar)"""
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
async def get_text_api(req: GetTextRequest):
    """Gets the text from the given URL or path of pdf, image, or text."""
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
async def speech_transcription(req: TranscriptionRequest):
    """Generates a transcription object for the given audio (path, url, or bytes)."""
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
    print(f'got request for test: {time.time()}')
    #await asyncio.sleep(10)
    return "Hello world\n"


def cleanup_executors():
    """Clean up all executors more aggressively"""
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
