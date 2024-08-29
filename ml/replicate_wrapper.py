"""Replicate wrappers for large hosted ML models."""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import tempfile
import time

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, cache
from os.path import dirname, exists
from typing import NamedTuple, Optional, Iterable, Sequence, Any
from urllib.request import urlretrieve

import replicate
import requests

from PIL import Image

from nkpylib.ml.constants import REPLICATE_MODELS

logger = logging.getLogger(__name__)


def data_url_from_file(file_obj, mimetype='') -> str:
    """Converts a file object to a data URL. You can optionally provide the explicit mimetype"""
    data = file_obj.read()
    if not mimetype:
        mimetype, _ = mimetypes.guess_type(file_obj.name)
    return f"data:{mimetype or ''};base64,{base64.b64encode(data).decode()}"


class Prediction:
    """Simple version of Replicate's Prediction object."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __iter__(self):
        return iter(self.__dict__.items())

def convert_files_to_data_urls(obj: dict) -> dict:
    """Recursively converts any file objects in the given dict to data URLs."""
    for k, v in obj.items():
        if isinstance(v, dict):
            obj[k] = convert_files_to_data_urls(v)
        elif hasattr(v, 'read'):
            obj[k] = data_url_from_file(v)
        # if it's a Sequence, recurse
        elif isinstance(v, Sequence) and not isinstance(v, str):
            obj[k] = [convert_files_to_data_urls(x) if hasattr(x, 'read') else x for x in v]
    return obj

def run_replicate_model(model_name: str,
                        version: str,
                        docker_port: Optional[int]=None,
                        **model_inputs: Any):
    """Runs a replicate model with the given `model_inputs` and returns the prediction object once done.

    If you specify a `docker_port` > 0, this will use the locally-running dockerized version of the
    model. In that case, we will generate our own Prediction object with all the returned outputs.
    """
    if docker_port and docker_port > 0:
        # convert any input fields that are open files to b64 data objects
        for k, v in model_inputs.items():
            if hasattr(v, 'read'):
                model_inputs[k] = data_url_from_file(v)
        # make the request
        while True:
            resp = requests.post(f'http://localhost:{docker_port}/predictions',
                                 json=dict(input=model_inputs))
            # 409 errors mean we should wait and retry
            if resp.status_code == 409:
                time.sleep(0.5)
                continue
            break
        ret = resp.json()
        return Prediction(**ret)
    else:
        model = replicate.models.get(model_name)
        version = model.versions.get(version)
        prediction = replicate.predictions.create(version=version, input=model_inputs)
        prediction.wait()
        return prediction

def face_detection(img: str) -> dict(str, Any):
    """
    Runs face detection on the given image.

    Returns a dict with at least:
    - `boxes`: list of bounding boxes for each face detected in the image
    - `mask_urls`: list of URLs to the face masks generated for each face detected

    Note that the bounding boxes are in absolute coordinates on the image, not normalized.
    The urls are only guaranteed to exist for upto an hour.

    This is a low-level function; you probably want to use the wrapped version in faces.py
    """
    # disable other loggers for the duration of this function
    orig_loggers = {}
    for name in ['httpx']:
        orig_loggers[name] = logging.getLogger(name).getEffectiveLevel()
        logging.getLogger(name).setLevel(logging.WARNING)

    model_inputs = dict(
        bias=0,
        images=img if img.startswith('http') else open(img, 'rb'),
        blur_amount=1,
        output_transparent_image=False,
    )
    prediction = run_replicate_model(**REPLICATE_MODELS['face_detection'], **model_inputs)
    logger.debug(f'Facedet output: {dict(prediction)}')
    # parse out the bounding boxes from the logs, which look like:
    #  '(137, 162, 232, 232)\n(260, 361, 216, 215)\n'
    # note that there might be junk lines in the logs
    boxes = []
    for line in prediction.logs.split('\n'):
        line = line.strip()
        try:
            box = tuple(map(int, line[1:-1].split(', ')))
            boxes.append(box)
        except Exception:
            pass
    ret = dict(boxes=[], mask_urls=[])
    for box, url in zip(boxes, prediction.output):
        ret['boxes'].append(box)
        ret['mask_urls'].append(url)

    # reset logging levels
    for name, level in orig_loggers.items():
        logging.getLogger(name).setLevel(level)
    return ret

def llm_complete(prompt: str, model_name: str='meta/meta-llama-3-70b-instruct', **kw):
    """Runs the LLM model on the given prompt and returns the completion."""
    ret = ''.join(replicate.run(model_name, input=dict(prompt=prompt, **kw)))
    return ret

TEST_DRIVERS = dict(
    face_detection=lambda:[face_detection('radiohead.jpg')],
    llm=lambda:[llm_complete('What is Dropbox? Explain in 1 sentence.')],
)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('func', choices=TEST_DRIVERS,
                        help=f'Function to run ({", ".join(TEST_DRIVERS)})')
    args = parser.parse_args()
    outputs = TEST_DRIVERS[args.func]()
    for o in outputs:
        print(o)
