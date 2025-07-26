from __future__ import annotations

import json

from abc import ABC, abstractmethod
from typing import Any

class CacheFormatter(ABC):
    """Base class for serialization formats."""
    @abstractmethod
    def dumps(self, obj: Any) -> bytes:
        """Serialize object to bytes."""
        pass

    @abstractmethod
    def loads(self, data: bytes) -> Any:
        """Deserialize bytes to object."""
        pass

class JsonFormatter(CacheFormatter):
    """JSON serialization format."""
    def __init__(self, EncoderCls=json.JSONEncoder, DecoderCls=json.JSONDecoder):
        self.EncoderCls = EncoderCls
        self.DecoderCls = DecoderCls

    def dumps(self, obj: Any) -> bytes:
        return json.dumps(obj, cls=self.EncoderCls, ensure_ascii=False).encode('utf-8')

    def loads(self, data: bytes) -> Any:
        return json.loads(data.decode('utf-8'), cls=self.DecoderCls)
