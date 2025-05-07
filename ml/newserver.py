from __future__ import annotations

import asyncio
import logging
import time

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

ModelCache: dict = {}
ResultsCache: dict = {}

class Model(ABC):
    """Base class for models, providing a common interface for loading and running models."""
    def __init__(self, model_name: str, use_cache: bool=True, **kw):
        self.model_name = model_name
        self.use_cache = use_cache
        self.model = None
        self.cache = ResultsCache.get(self.__class__, {})
        self.timing: dict[str, Any] = {}

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
            if self.model_name in ModelCache:
                self.model = ModelCache[model_cache_key]
                logger.debug(f"Model {self.model_name} loaded from cache")
                return False
            t0 = time.time()
            self.model = await self._load(**kw)
            t1 = time.time()
            ModelCache[model_cache_key] = self.model
            self.timing['load'] = t1 - t0
            logger.debug(f"Model {self.model_name} loaded in {t1-t0:.2f}s")
            return True
        return False

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
            self.found_cache = True
        else:
            ret = await self._run(input, **kw)
            self.found_cache = False
        t1 = time.time()
        self.timing['generate'] = t1 - t0
        ret.timing = dict(self.timing)
        logger.debug(f"Model {self.model_name} run in {t1-t0:.2f}s")
        return ret


if __name__ == '__main__':
    m = Model('test')

class LocalChatModel(Model):
    """Model subclass for handling local chat models."""
    
    async def _load(self, **kw) -> Any:
        from llama_cpp import Llama
        return Llama(
            model_path=self.model_name,
            n_ctx=8192,
            n_threads=8,
            n_gpu_layers=35,
        )

    async def _get_cache_key(self, input: Any, **kw) -> str:
        return f"{kw.get('max_tokens', 1024)}:{str(input)}"

    async def _run(self, input: Any, **kw) -> dict:
        model = self.model
        result = model(
            input,
            max_tokens=kw.get('max_tokens', 1024),
            echo=False,
        )
        return result


class ExternalChatModel(Model):
    """Model subclass for handling external chat models."""
    
    async def _load(self, **kw) -> Any:
        return self.model_name

    async def _get_cache_key(self, input: Any, **kw) -> str:
        return f"{kw.get('max_tokens', 1024)}:{str(input)}"

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
        ret = await call_external(endpoint='/chat/completions', provider_name=kw.get('provider', ''), model=model, **kw)
        return ret
