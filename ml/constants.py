"""Various constants used throughout the application (both for server and client)."""

from __future__ import annotations

import base64
import mimetypes
import os

from dataclasses import dataclass
from os.path import dirname, join
from typing import Literal

SERVER_BASE_URL = "http://localhost:8234"

SERVER_API_VERSION = "1"

PROVIDERS_PATH = 'providers.json'

@dataclass
class ModelConfig:
    name: str
    max_tokens: int=10240


DEFAULT_MODELS = dict(
    # sentence embedding models for lots of text
    st=ModelConfig('BAAI/bge-large-en-v1.5'),
    sentence=ModelConfig('BAAI/bge-large-en-v1.5'),
    # clip for image <-> text embeddings in same space
    clip=ModelConfig("openai/clip-vit-large-patch14"),
    image=ModelConfig('openai/clip-vit-large-patch14'),
    # jina-clip-v2 for better image <-> text embeddings in the same space
    jina=ModelConfig("jinaai/jina-clip-v2"),
    # llama4 (scout) for top-line perf on text tasks
    llama4=ModelConfig('meta-llama/Llama-4-Scout-17B-16E-Instruct', 131072),
    # qwen, latest baidu model
    qwen=ModelConfig('Qwen/Qwen3-30B-A3B', 32768),
    qwen_large=ModelConfig('Qwen/Qwen3-235B-A22B', 32768),
    # llama3 for default good perf on various text tasks
    llama3=ModelConfig('meta-llama/Llama-3.3-70B-Instruct', 131072),
    text=ModelConfig('meta-llama/Llama-3.3-70B-Instruct', 131072),
    # llama3 turbo as all-purpose default fast model
    llama3_turbo=ModelConfig('meta-llama/Llama-3.3-70B-Instruct-Turbo', 131072),
    turbo=ModelConfig('meta-llama/Llama-3.3-70B-Instruct-Turbo', 131072),
    # faster llama3 for chat
    chat=ModelConfig('meta-llama/Llama-3.3-70B-Instruct-Turbo', 131072),
    # generic vlm model for vision+language tasks
    vlm=ModelConfig('meta-llama/Llama-3.2-90B-Vision-Instruct', 131072),
    # model for better vlm performance on extracting doc data
    docimage=ModelConfig('accounts/fireworks/models/qwen2-vl-72b-instruct', 32768),
    # a fast model for text tasks
    fast=ModelConfig('mistralai/Mistral-Small-24B-Instruct-2501', 32768),
    # a good model for manipulating json
    json=ModelConfig('Qwen/Qwen2.5-72B-Instruct', 32768),
    # a good model for manipulating html (llama3)
    html=ModelConfig('meta-llama/Llama-3.3-70B-Instruct', 131072),
    # nomic is a good model for text embeddings
    nomic=ModelConfig('nomic-ai/nomic-embed-text-v1.5'),
    # suitable for generating code
    code=ModelConfig('Qwen/Qwen2.5-Coder-32B-Instruct', 32768),
    # speech transcription
    whisper=ModelConfig('openai/whisper-large-v3'),
    speech=ModelConfig('openai/whisper-large-v3'),
    transcription=ModelConfig('openai/whisper-large-v3'),
    # small fast ada text embedding model
    ada=ModelConfig('text-embedding-ada-002'),
    # e5 embeddings with 1024 output dims
    e5=ModelConfig('intfloat/e5-large-v2'),
    # qwen3 embeddings (large and small)
    qwen_emb=ModelConfig('Qwen/Qwen3-Embedding-8B'),
    qwen_emb_small=ModelConfig('Qwen/Qwen3-Embedding-0.6B'),
    # groq fastest
    groq=ModelConfig('openai/gpt-oss-20b', 8192),

)

LOCAL_MODELS = os.listdir(join(dirname(__file__), 'models/')) + ['openai/clip-vit-large-patch14']

REPLICATE_MODELS = dict(
    face_detection=dict(
        model_name='chigozienri/mediapipe-face',
        version='b52b4833a810a8b8d835d6339b72536d63590918b185588be2def78a89e7ca7b',
        docker_port=5005),
)

# typedefs for message roles and text
Role = Literal['user', 'assistant', 'system']
Msg = tuple[Role, str]


def get_model_name(key: str) -> str:
    """Get the model name from DEFAULT_MODELS, maintaining backward compatibility."""
    model_config = DEFAULT_MODELS.get(key, key)
    if isinstance(model_config, ModelConfig):
        return model_config.name
    return model_config

def get_model_max_tokens(key: str) -> int:
    """Get the max_tokens for a model, with sensible default."""
    model_config = DEFAULT_MODELS.get(key)
    if isinstance(model_config, ModelConfig):
        return model_config.max_tokens
    return 10240  # default fallback

def data_url_from_file(file_obj, mimetype='') -> str:
    """Converts a file object to a data URL. You can optionally provide the explicit mimetype"""
    data = file_obj.read()
    if not mimetype:
        mimetype, _ = mimetypes.guess_type(file_obj.name)
    return f"data:{mimetype or ''};base64,{base64.b64encode(data).decode()}"
