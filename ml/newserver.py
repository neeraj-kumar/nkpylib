import asyncio
import time
from typing import Any, Callable, Optional

class Model:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.cache = {}
        self.did_load = False

    async def load(self, **kw) -> None:
        """Loads the model using the provided load function."""
        if self.model is None:
            t0 = time.time()
            self.model = await asyncio.to_thread(self.load_model, self.model_name, **kw)
            t1 = time.time()
            self.did_load = True
            print(f"Model {self.model_name} loaded in {t1-t0:.2f}s")
        else:
            print(f"Model {self.model_name} already loaded")

    async def run(self, input: Any, cache_key: Optional[str] = None, **kw) -> dict:
        """Runs the model using the provided run function, with optional caching."""
        if cache_key and cache_key in self.cache:
            print(f"Using cached result for {cache_key}")
            return self.cache[cache_key]

        if self.model is None:
            await self.load(**kw)

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

# Additional subclasses can be created for other models as needed

