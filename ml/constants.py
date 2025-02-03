"""Various constants used throughout the application (both for server and client)."""

from __future__ import annotations

import os

from typing import Literal

SERVER_BASE_URL = "http://aphex.local:8234"

SERVER_API_VERSION = "1"

PROVIDERS_PATH = 'providers.json'

DEFAULT_MODELS = dict(
    st='BAAI/bge-large-en-v1.5',
    sentence='BAAI/bge-large-en-v1.5',
    clip="openai/clip-vit-large-patch14",
    image='openai/clip-vit-large-patch14',
    llama3='meta-llama/Llama-3.3-70B-Instruct',
    chat='meta-llama/Llama-3.3-70B-Instruct-Turbo',
    text='meta-llama/Llama-3.3-70B-Instruct',
)

LOCAL_MODELS = os.listdir('models/') + ['openai/clip-vit-large-patch14']

REPLICATE_MODELS = dict(
    face_detection=dict(
        model_name='chigozienri/mediapipe-face',
        version='b52b4833a810a8b8d835d6339b72536d63590918b185588be2def78a89e7ca7b',
        docker_port=5005),
)

# typedefs for message roles and text
Role = Literal['user', 'assistant', 'system']
Msg = tuple[Role, str]
