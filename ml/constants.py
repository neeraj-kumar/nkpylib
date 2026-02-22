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

DEFAULT_MAX_TOKENS = 32768


@dataclass
class ModelConfig:
    name: str
    max_tokens: int=DEFAULT_MAX_TOKENS
    max_dims: int=0 # for embeddings
    default_dims: int=0 # same as max if not given

DEFAULT_MODELS = dict(
    # sentence embedding models for lots of text
    st=ModelConfig('BAAI/bge-large-en-v1.5', max_dims=1024, max_tokens=512),
    sentence=ModelConfig('BAAI/bge-large-en-v1.5', max_dims=1024, max_tokens=512),
    # clip for image <-> text embeddings in same space
    clip=ModelConfig("openai/clip-vit-large-patch14", max_dims=768),
    image=ModelConfig('openai/clip-vit-large-patch14', max_dims=768),
    # fast image model is mobilenet
    mobilenet=ModelConfig('mobilenet_v3', max_dims=576),
    fast_image=ModelConfig('mobilenet_v3', max_dims=576),
    # jina-clip-v2 for better image <-> text embeddings in the same space
    # - Based on the research paper, going from the max dims of 1024 to 768 doesn't hurt performance
    #   at all (and might even slightly improve it for some tasks).
    jina=ModelConfig("jinaai/jina-clip-v2", max_dims=1024, default_dims=768),
    # llama4 (scout) for top-line perf on text tasks
    llama4=ModelConfig('meta-llama/Llama-4-Scout-17B-16E-Instruct', max_tokens=131072),
    # qwen, latest baidu model
    qwen=ModelConfig('Qwen/Qwen3-30B-A3B', max_tokens=32768),
    qwen_large=ModelConfig('Qwen/Qwen3-235B-A22B', max_tokens=32768),
    # llama3 for default good perf on various text tasks
    llama3=ModelConfig('meta-llama/Llama-3.3-70B-Instruct-Turbo', max_tokens=131072),
    text=ModelConfig('meta-llama/Llama-3.3-70B-Instruct-Turbo', max_tokens=131072),
    # llama3 turbo as all-purpose default fast model
    llama3_turbo=ModelConfig('meta-llama/Llama-3.3-70B-Instruct-Turbo', max_tokens=131072),
    turbo=ModelConfig('meta-llama/Llama-3.3-70B-Instruct-Turbo', max_tokens=131072),
    # faster llama3 for chat
    chat=ModelConfig('meta-llama/Llama-3.3-70B-Instruct-Turbo', max_tokens=131072),
    # generic vlm model for vision+language tasks
    vlm=ModelConfig('zai-org/GLM-4.6V', max_tokens=131072),
    # faster/cheaper vlm
    fastvlm=ModelConfig('google/gemma-3-4b-it', max_tokens=131072),
    oldvlm=ModelConfig('meta-llama/Llama-3.2-90B-Vision-Instruct', max_tokens=131072),
    # model for better vlm performance on extracting doc data
    docimage=ModelConfig('accounts/fireworks/models/qwen2-vl-72b-instruct', max_tokens=32768),
    # a fast model for text tasks
    fast=ModelConfig('mistralai/Mistral-Small-24B-Instruct-2501', max_tokens=32768),
    # a good model for manipulating json
    json=ModelConfig('Qwen/Qwen2.5-72B-Instruct', max_tokens=32768),
    # a good model for manipulating html (llama3)
    html=ModelConfig('moonshotai/Kimi-K2-Instruct-0905', max_tokens=262144),
    oldhtml=ModelConfig('meta-llama/Llama-3.3-70B-Instruct', max_tokens=131072),
    # nomic is a good model for text embeddings
    nomic=ModelConfig('nomic-ai/nomic-embed-text-v1.5', max_dims=768, max_tokens=8192),
    # suitable for generating code
    code=ModelConfig('Qwen/Qwen2.5-Coder-32B-Instruct', max_tokens=32768),
    # speech transcription
    whisper=ModelConfig('openai/whisper-large-v3'),
    speech=ModelConfig('openai/whisper-large-v3'),
    transcription=ModelConfig('openai/whisper-large-v3'),
    # e5 embeddings with 1024 output dims
    e5=ModelConfig('intfloat/e5-large-v2', max_dims=1024, max_tokens=514),
    # qwen3 embeddings (large and small)
    qwen_emb=ModelConfig('Qwen/Qwen3-Embedding-8B', max_dims=4096, max_tokens=32768),
    qwen_emb_small=ModelConfig('Qwen/Qwen3-Embedding-0.6B', max_dims=1024, max_tokens=32768),
    # groq fastest
    groq=ModelConfig('openai/gpt-oss-20b', max_tokens=8192),
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


def data_url_from_file(file_obj, mimetype='') -> str:
    """Converts a file object to a data URL. You can optionally provide the explicit mimetype"""
    data = file_obj.read()
    if not mimetype:
        mimetype, _ = mimetypes.guess_type(file_obj.name)
    return f"data:{mimetype or ''};base64,{base64.b64encode(data).decode()}"
