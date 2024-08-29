"""Various constants used throughout the application (both for server and client)."""

SERVER_BASE_URL = "http://aphex.local:8234"

SERVER_API_VERSION = "1"

DEFAULT_MODELS = dict(
    st='BAAI/bge-large-en-v1.5',
    sentence='BAAI/bge-large-en-v1.5',
    clip="openai/clip-vit-large-patch14",
    image='openai/clip-vit-large-patch14',
    llama3='meta/meta-llama-3-70b-instruct'
)

REPLICATE_MODELS = dict(
    face_detection=dict(
        model_name='chigozienri/mediapipe-face',
        version='b52b4833a810a8b8d835d6339b72536d63590918b185588be2def78a89e7ca7b',
        docker_port=5005),
)
