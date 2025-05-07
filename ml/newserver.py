import asyncio
import logging
import time

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

ModelCache = {}
ResultsCache = {}

class Model(ABC):
    """Base class for models, providing a common interface for loading and running models."""
    def __init__(self, model_name: str, use_cache: bool=True, **kw):
        self.model_name = model_name
        self.use_cache = use_cache
        self.model = None
        self.cache = ResultsCache.get(self.__class__, {})

    async def _load(self, **kw) -> Any:
        """Load implementation.

        This version just returns the model name as the model itself (useful for external APIs).
        """
        return self.model_name

    async def load(self, **kw) -> None:
        """Loads our model if not already loaded"""
        if self.model is None:
            t0 = time.time()
            self.model = await self._load(**kw)
            t1 = time.time()
            logger.debug(f"Model {self.model_name} loaded in {t1-t0:.2f}s")
        else:
            print(f"Model {self.model_name} already loaded")

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
            cache_key, _ = await asyncio.gather(
                self._get_cache_key(input, **kw) if self.use_cache else asyncio.sleep(0),
                self.load(**kw) if self.model is None else asyncio.sleep(0)
            )

        t0 = time.time()
        result = await asyncio.to_thread(self.run_model, input, self.model, **kw)
        t1 = time.time()
        print(f"Model {self.model_name} run in {t1-t0:.2f}s")

        if cache_key:
            self.cache[cache_key] = result

        return result

# Example subclasses for each function in server.py
class ClipModel(Model):
    def __init__(self):
        super().__init__("clip", self.load_clip, self.run_clip)

    def load_clip(self, model_name: str, **kw) -> Any:
        # Implement the specific loading logic for the clip model
        pass

    def run_clip(self, input: Any, model: Any, **kw) -> dict:
        # Implement the specific running logic for the clip model
        pass

class LlamaModel(Model):
    def __init__(self):
        super().__init__("llama", self.load_llama, self.run_llama)

    def load_llama(self, model_name: str, **kw) -> Any:
        # Implement the specific loading logic for the llama model
        pass

    def run_llama(self, input: Any, model: Any, **kw) -> dict:
        # Implement the specific running logic for the llama model
        pass

if __name__ == '__main__':
    m = Model('test')

