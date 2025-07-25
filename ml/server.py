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
import functools
import json
import logging
import tempfile
import time
import uuid

from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
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

from nkpylib.ml.constants import DEFAULT_MODELS, LOCAL_MODELS, Role, Msg, data_url_from_file
from nkpylib.ml.providers import call_external, call_provider
from nkpylib.ml.text import get_text
from nkpylib.thread_utils import Singleton
from nkpylib.web_utils import make_request_async, dl_temp_file
from nkpylib.utils import is_instance_of_type

logger = logging.getLogger(__name__)

app = fastapi.FastAPI()

MODEL_CACHE: dict = {}
RESULTS_CACHE: dict = {}

RESULTS_CACHE_LIMIT = 90000

# load func takes model name and **kw, and returns the loaded model
LoadFuncT = Callable[[Any], Any]
RunFuncT = Callable[[Any, Any], dict]

@functools.cache
def load_clip(model_name=DEFAULT_MODELS['clip']):
    """Loads clip and returns two embedding functions: one for text, one for images"""
    import torch
    from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoTokenizer # type: ignore
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

@functools.cache
def load_jina(model_name=DEFAULT_MODELS['jina'], dims:int =768):
    """Loads jina and returns two embedding functions: one for text, one for images.

    Based on the research paper, going from the max dims of 1024 to 768 doesn't hurt performance at
    all (and might even slightly improve it for some tasks).
    """
    from sentence_transformers import SentenceTransformer
    assert dims <= 1024, "Jina-clip-v2 only supports up to 1024 dimensions"
    model = SentenceTransformer(model_name, trust_remote_code=True, truncate_dim=dims)

    def get_image_features(image_or_path):
        return model.encode([image_or_path], normalize_embeddings=True)[0]

    def get_text_features(text):
        # note that in the example code, for "query" embeddings,
        # they also set prompt_name='retrieval.query'...
        return model.encode([text], normalize_embeddings=True)[0]

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


class Model(ABC):
    """Base class for models, providing a common interface for loading and running models."""
    def __init__(self, model_name: str='', use_cache: bool=True, **kw):
        if model_name in DEFAULT_MODELS:
            model_name = DEFAULT_MODELS[model_name]
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
        )
        self.current: set[Any] = set() # what's currently running
        self.callers: Counter = Counter() # current set of callers


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
        """Runs the model with given `input`"""
        t0 = time.time()
        cache_key = None
        kw = self.update_kw(input, **kw)
        self.callers[caller] += 1
        if self.use_cache or self.model is None:
            cache_key, self.timing['did_load'] = await asyncio.gather(
                self._get_cache_key(input, **kw) if self.use_cache else asyncio.sleep(0),
                self.load(**kw) if self.model is None else asyncio.sleep(0)
            )
        if cache_key in self.cache:
            ret = self.cache[cache_key]
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
                self.cache[cache_key] = ret
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
        self.timing['throughput'] = self.timing['n_inferences'] / self.timing['load_elapsed']
        self.timing['avg_generate'] = self.timing['generate_all'] / (self.timing['n_inferences']+1)
        ret['timing'] = dict(self.timing, generate=t1-t0, found_cache=cache_key in self.cache)
        logger.debug(f"Model {self.model_name} run in {t1-t0:.2f}s")
        return ret


class ChatModel(Model):
    """Base class for chat models.

    This currently just sets a default value for `max_tokens`, and defines the cache key as
    the `max_tokens` and the input string(s).
    """
    def update_kw(self, input: Any, **kw) -> dict:
        """Updates the `kw` input dict with default parameters, etc."""
        if 'max_tokens' not in kw:
            kw['max_tokens'] = 1024
        return kw

    async def _get_cache_key(self, input: Any, **kw) -> str:
        return f"{kw['max_tokens']}:{str(input)}"

    def postprocess(self, input, ret, **kw) -> dict:
        """Augments the output with additional information."""
        ret['prompts'] = input
        ret['model'] = self.model_name.split('/', 1)[-1]
        ret['max_tokens'] = kw['max_tokens']
        return ret


class LocalChatModel(ChatModel):
    """Model subclass for handling local chat models."""
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
        prompts = input
        if isinstance(prompts, str):
            prompts = [('user', prompts)]
        assert is_instance_of_type(prompts, list[Msg]), f"Prompts should be of type {list[Msg]}, actually: {prompts}"
        if not kw:
            kw = {}
        kw['messages'] = [{'role': role, 'content': text} for role, text in prompts]
        ret = await call_external(endpoint='/chat/completions', provider_name=kw.pop('provider', ''), model=model, **kw)
        return self.postprocess(input, ret, **kw)


@Singleton
class VLMModel(ChatModel):
    """Model subclass for handling VLM models."""
    async def _get_cache_key(self, input: Any, **kw) -> str:
        image, prompts = input
        return f"{kw['max_tokens']}:{image}:{str(prompts)}"

    async def _run(self, input: Any, **kw) -> dict:
        logger.debug(f'Running VLM model: {self.model_name} on input: {input} with kw {kw}')
        image, prompts = input
        if isinstance(prompts, str):
            prompts = [('user', prompts)]
        assert is_instance_of_type(prompts, list[Msg]), f"Prompts should be of type {list[Msg]}, actually: {prompts}"
        kw['messages'] = [{'role': role, 'content': text} for role, text in prompts]
        if not image.startswith('http') and not image.startswith('data:'):
            with open(image, 'rb') as f:
                image = data_url_from_file(f)
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

class ImageTextEmbeddingModel(EmbeddingModel):
    """Model subclass for handling joint text/image embeddings."""
    def __init__(self, mode='text', model_name: str='', use_cache: bool=True, **kw):
        super().__init__(model_name=model_name, use_cache=use_cache, **kw)
        assert mode in ('text', 'image')
        self.mode = mode

    async def load_feature_funcs(self):
        """Returns two functions: one for text features, one for image features."""
        raise NotImplementedError("This method should be overridden in subclasses")

    async def _load(self, **kw) -> Any:
        get_text_features, get_image_features = await self.load_feature_funcs()
        if self.mode == 'text':
            return get_text_features
        elif self.mode == 'image':
            return get_image_features
        raise NotImplementedError(f"Unsupported mode: {self.mode}")

    async def _run(self, input: Any, **kw) -> dict:
        try:
            ret = await asyncio.to_thread(self.model, input) # no kw options at runtime
            if not isinstance(ret, np.ndarray):
                ret = ret.numpy()
            return self.postprocess(ret)
        except Exception as e:
            logger.error(f"Error running {self.model_name} embedding model: {e}")
            return {}

@Singleton
class ClipEmbeddingModel(ImageTextEmbeddingModel):
    """Model subclass for handling CLIP text/image embeddings."""
    async def load_feature_funcs(self):
        """Returns two functions: one for text features, one for image features."""
        return load_clip()


@Singleton
class JinaEmbeddingModel(ImageTextEmbeddingModel):
    """Model subclass for handling Jina text/image embeddings."""
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

    This checks that the input is a valid path, and uses the hash of the path as the cache key"""
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
    ClipEmbeddingModel,
    JinaEmbeddingModel,
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
    prompts: str|list[Msg]
    model: str='chat' # this will get mapped based on DEFAULT_MODELS['chat']
    max_tokens: int=1024

@app.post("/v1/chat")
async def chat(req: ChatRequest):
    """Generates chat response for the given prompts using the given model."""
    logger.debug(f'running chat model')
    if req.model in DEFAULT_MODELS:
        req.model = DEFAULT_MODELS[req.model]
    model: ChatModel
    if req.model in LOCAL_MODELS:
        model = LocalChatModel(model_name=req.model, device='cpu', use_cache=req.use_cache)
    else:
        model = ExternalChatModel(model_name=req.model, use_cache=req.use_cache)
    ret = await model.run(
        input=req.prompts,
        max_tokens=req.max_tokens,
        provider=req.provider,
        caller=req.caller,
        **req.kwargs
    )
    return ret

# setup fastapi VLM endpoint
class VLMRequest(BaseRequest):
    image: str # path/url/image directly?
    prompts: str|list[Msg]
    model: str='vlm' # this will get mapped based on DEFAULT_MODELS['vlm']
    max_tokens: int=1024

@app.post("/v1/vlm")
async def vlm(req: VLMRequest):
    """Generates VLM chat response for the given image and prompts using the given model."""
    model = VLMModel(model_name=req.model, use_cache=req.use_cache)
    ret = await model.run(
        input=(req.image, req.prompts),
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
    if req.model in DEFAULT_MODELS:
        req.model = DEFAULT_MODELS[req.model]
    model_class_by_name = {
        DEFAULT_MODELS['clip']: lambda **kw: ClipEmbeddingModel(mode='text', **kw),
        DEFAULT_MODELS['jina']: lambda **kw: JinaEmbeddingModel(mode='text', **kw),
        DEFAULT_MODELS['sentence']: SentenceTransformerModel,
    }
    ModelClass: Any = model_class_by_name.get(req.model, ExternalEmbeddingModel)
    model = ModelClass(model_name=req.model, use_cache=req.use_cache)
    if 0: #FIXME what was this code even doing??
        async with dl_temp_file(req.input) as path:
            ret = await model.run(input=path, provider=req.provider, caller=req.caller)
    else:
        ret = await model.run(input=req.input, provider=req.provider, caller=req.caller)
    return ret


class ImageEmbeddingRequest(BaseRequest):
    url: str
    model: str=DEFAULT_MODELS['image']


@app.post("/v1/image_embeddings")
async def image_embeddings(req: ImageEmbeddingRequest):
    """Generates embeddings for the given image url (or local path) using the given model."""
    req.model = DEFAULT_MODELS.get(req.model, req.model)
    if req.model == DEFAULT_MODELS['clip']:
        Cls = ClipEmbeddingModel
    elif req.model == DEFAULT_MODELS['jina']:
        Cls = JinaEmbeddingModel
    else:
        raise NotImplementedError(f"Model {req.model} not supported for image embeddings")
    model = Cls(model_name=req.model, mode='image', use_cache=req.use_cache)
    ret = await model.run(input=req.url, caller=req.caller)
    return ret


class StrSimRequest(BaseRequest):
    a: str
    b: str
    model: str=DEFAULT_MODELS['clip']

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

@app.post("/v1/get_text")
async def get_text_api(req: GetTextRequest, cache={}):
    """Gets the text from the given URL or path of pdf, image, or text."""
    model = TextExtractionModel(model_name='text', use_cache=req.use_cache)
    ret = await model.run(input=req.url, caller=req.caller, **(req.kwargs or {}))
    if req.use_cache:
        cache[req.url] = ret
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
    print(f'In speech transcription, got model {req.model} and cls {ModelClass}')
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
    test()
