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
    st='BAAI/bge-large-en-v1.5',
    sentence='BAAI/bge-large-en-v1.5',
    # clip for image <-> text embeddings in same space
    clip="openai/clip-vit-large-patch14",
    image='openai/clip-vit-large-patch14',
    # jina-clip-v2 for better image <-> text embeddings in the same space
    jina="jinaai/jina-clip-v2",
    # llama4 (scout) for top-line perf on text tasks
    llama4='meta-llama/Llama-4-Scout-17B-16E-Instruct',
    # qwen, latest baidu model
    qwen='Qwen/Qwen3-30B-A3B',
    qwen_large='Qwen/Qwen3-235B-A22B',
    # llama3 for default good perf on various text tasks
    llama3='meta-llama/Llama-3.3-70B-Instruct',
    text='meta-llama/Llama-3.3-70B-Instruct',
    # llama3 turbo as all-purpose default fast model
    llama3_turbo='meta-llama/Llama-3.3-70B-Instruct-Turbo',
    turbo='meta-llama/Llama-3.3-70B-Instruct-Turbo',
    # faster llama3 for chat
    chat='meta-llama/Llama-3.3-70B-Instruct-Turbo',
    # generic vlm model for vision+language tasks
    vlm='meta-llama/Llama-3.2-90B-Vision-Instruct',
    # model for better vlm performance on extracting doc data
    docimage='accounts/fireworks/models/qwen2-vl-72b-instruct',
    # a fast model for text tasks
    fast='mistralai/Mistral-Small-24B-Instruct-2501',
    # a good model for manipulating json
    json='Qwen/Qwen2.5-72B-Instruct',
    # a good model for manipulating html (llama3)
    html='meta-llama/Llama-3.3-70B-Instruct',
    # nomic is a good model for text embeddings
    nomic='nomic-ai/nomic-embed-text-v1.5',
    # suitable for generating code
    code='Qwen/Qwen2.5-Coder-32B-Instruct',
    # speech transcription
    whisper='openai/whisper-large-v3',
    speech='openai/whisper-large-v3',
    transcription='openai/whisper-large-v3',
    # small fast ada text embedding model
    ada='text-embedding-ada-002',
    # e5 embeddings with 1024 output dims
    e5='intfloat/e5-large-v2',
    # qwen3 embeddings (large and small)
    qwen_emb='Qwen/Qwen3-Embedding-8B',
    qwen_emb_small='Qwen/Qwen3-Embedding-0.6B',
    # groq fastest
    groq='openai/gpt-oss-20b',

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
