from __future__ import annotations

import asyncio
import json
import logging
import time

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from nkpylib.utils import is_instance_of_type
from nkpylib.ml.providers import call_external
from nkpylib.ml.constants import DEFAULT_MODELS, LOCAL_MODELS, Role, Msg, data_url_from_file

logger = logging.getLogger(__name__)

MODEL_CACHE: dict = {}
RESULTS_CACHE: dict = {}

class Model(ABC):
    """Base class for models, providing a common interface for loading and running models."""
    def __init__(self, model_name: str=None, use_cache: bool=True, **kw):
        if model_name in DEFAULT_MODELS:
            model_name = DEFAULT_MODELS[model_name]
        self.model_name = model_name
        self.use_cache = use_cache
        self.model = None
        self.cache = RESULTS_CACHE.setdefault(self.__class__, {})
        self.timing: dict[str, Any] = dict(model=model_name)


    async def _load(self, **kw) -> Any:
        """Load implementation.

        This version just returns the model name as the model itself (useful for external APIs).
        """
        return self.model_name

    async def load(self, **kw) -> bool:
        """Loads our model if not already loaded, returning if we actually loaded it"""
        model_cache_key = (self.__class__, self.model_name)
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

    async def run(self, input: Any, **kw) -> dict:
        """Runs the model with given `input`"""
        cache_key = None
        kw = self.update_kw(input, **kw)
        if self.use_cache or self.model is None:
            cache_key, self.timing['did_load'] = await asyncio.gather(
                self._get_cache_key(input, **kw) if self.use_cache else asyncio.sleep(0),
                self.load(**kw) if self.model is None else asyncio.sleep(0)
            )
            assert cache_key is not None
        t0 = time.time()
        if cache_key in self.cache:
            ret = self.cache[cache_key]
            self.cache[cache_key] = ret
            self.timing['found_cache'] = True
        else:
            ret = await self._run(input, **kw)
            self.cache[cache_key] = ret
            self.timing['found_cache'] = False
        t1 = time.time()
        self.timing['generate'] = t1 - t0
        ret['timing'] = dict(self.timing)
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


class ClipEmbeddingModel(EmbeddingModel):
    """Model subclass for handling CLIP text/image embeddings."""
    def __init__(self, mode='text', model_name: str=None, use_cache: bool=True, **kw):
        super().__init__(model_name=model_name, use_cache=use_cache, **kw)
        assert mode in ('text', 'image')
        assert self.model_name == DEFAULT_MODELS['clip']
        self.mode = mode

    async def _load(self, **kw) -> Any:
        from nkpylib.ml.server import load_clip
        get_text_features, get_image_features = load_clip()
        if self.mode == 'text':
            return get_text_features
        elif self.mode == 'image':
            return get_image_features
        raise NotImplementedError(f"Unsupported mode: {self.mode}")

    async def _run(self, input: Any, **kw) -> dict:
        ret = await asyncio.to_thread(self.model, input) # clip doesn't use any kw
        ret = ret.numpy()
        return self.postprocess(ret)


class SentenceTransformerModel(EmbeddingModel):
    """Model subclass for handling SentenceTransformer embeddings."""
    async def _load(self, **kw) -> Any:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.model_name)

    async def _run(self, input: Any, **kw) -> dict:
        embedding = self.model.encode([input], normalize_embeddings=True)[0]
        return self.postprocess(embedding)


class ExternalEmbeddingModel(EmbeddingModel):
    """Model subclass for handling external API text embeddings."""
    async def _run(self, input: Any, **kw) -> dict:
        ret = await call_external(endpoint='/embeddings', provider_name=kw.get('provider', ''), model=self.model_name, input=input)
        ret['input'] = input
        return ret


async def test():
    if 0:
        for i in range(2):
            m = ExternalChatModel('chat')
            r = await m.run('what is the capital of france?')
            print(json.dumps(r, indent=2))
        m = ExternalChatModel('chat')
        r = await m.run('what is the capital of india?')
        print(json.dumps(r, indent=2))
    m = VLMModel('vlm')
    r = await m.run(('./radiohead.jpg', 'what is this?'))
    print(json.dumps(r, indent=2))


logging.basicConfig(level=logging.INFO)
#asyncio.run(test())
