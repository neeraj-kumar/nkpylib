"""Some clip utilities"""

import functools
import logging
import os
import time

from argparse import ArgumentParser
from typing import Any, Callable, Sequence, Type, Union

import numpy as np
import requests
import torch

from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoTokenizer
from ..stringutils import generate_random_sentences

logger = logging.getLogger(__name__)

EmbeddingT = Union[np.ndarray, torch.Tensor, Sequence[float]]
TextEmbedderT = Callable[[str], EmbeddingT]
ImageEmbedderT = Callable[[Union[str, Image.Image]], EmbeddingT]

@functools.cache
def load_clip(model_name: str='openai/clip-vit-large-patch14') -> tuple[TextEmbedderT, ImageEmbedderT]:
    """Loads clip and returns two embedding functions: one for text, one for images.

    - Both take a single input and return a 768-dim vector.
    - The text embedder takes text directly.
    - The image embedder takes either a PIL image or a local path to an image or a url to an image.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    def get_image_features(image_or_path: Union[str, Image.Image]) -> EmbeddingT:
        if isinstance(image_or_path, str):
            if image_or_path.startswith('http'):
                image = Image.open(requests.get(image_or_path, stream=True).raw)
            else:
                image = Image.open(image_or_path)
        else:
            image = image_or_path
        with torch.no_grad():
            return model.get_image_features(**processor(images=image, return_tensors="pt"))[0]

    def get_text_features(text: str) -> EmbeddingT:
        with torch.no_grad():
            return model.get_text_features(**processor(text=text, return_tensors="pt"))[0]

    return get_text_features, get_image_features

def match_embedding(needle: EmbeddingT, haystack: np.ndarray, k:int=5, largest=True) -> Any:
    """Given a needle and haystack, find the `k` closest items in haystack.

    - The needle is an embedding
    - The haystack is a numpy array of embeddings
    - By default returns the largest items, but you can flip that with `largest=False`
    - The returned object has fields `indices` and `values` of the matches.
    """
    similarity = np.dot(needle, haystack.T)
    topk = torch.topk(torch.tensor(similarity), k, largest=largest)
    return topk


def benchmark() -> None:
    """Does some basic benchmarking.

    Note that clip outputs are 768 dims.

    On my desktop on Dec 19, 2023, it's looking like:
        text: 100 sentences in 6.842s
        images with loading: 10 images (~10MP) in 11.254s
        images without loading: 10 images (~10MP) in 10.917s
        images with loading: 0.3 images (~0.3MP) in 10.085s
        images without loading: 0.3 images (~0.3MP) in 10.105s

    Note that paths were relative to ~/dp/projects/cooking
    """
    get_text_features, get_image_features = load_clip()

    sentences = list(generate_random_sentences(100))
    logger.info(f'Got {len(sentences)} sentences: {sentences[:5]}')
    t0 = time.time()
    _feats = []
    for sentence in sentences:
        _feats.append(get_text_features(sentence))
    t1 = time.time()
    feats = np.array(_feats)
    logger.info(f'Got {feats.shape} text features in {t1-t0:.3f}s: {feats}')

    im_dir = '../../Camera Uploads'
    im_dir = '../../randomimages/'
    images = [f for f in os.listdir(im_dir) if f.lower().endswith('.jpg')][:10]
    logger.info(f'Got {len(images)} images: {images[:5]}')
    t0 = time.time()
    _feats = []
    for path in images:
        _feats.append(get_image_features(os.path.join(im_dir, path)))
    t1 = time.time()
    feats = np.array(_feats)
    logger.info(f'Got {feats.shape} image features (with loading) in {t1-t0:.3f}s: {feats}')

    _feats = []
    elapsed = 0.0
    sizes = [] # in megapixels
    for path in images:
        # load the image
        image = Image.open(os.path.join(im_dir, path))
        sizes.append(image.size[0] * image.size[1] / 1e6)
        t0 = time.time()
        _feats.append(get_image_features(image))
        t1 = time.time()
        elapsed += t1 - t0
    feats = np.array(_feats)
    logging.info(f'Got {feats.shape} image features (without loading) in {elapsed:.3f}s: {feats}\n{sizes}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()
    funcs = {f.__name__: f for f in [benchmark]}
    parser.add_argument('func', choices=funcs.keys())
    args = parser.parse_args()
    funcs[args.func](**vars(args))
