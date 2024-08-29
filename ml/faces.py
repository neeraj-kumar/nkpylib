"""Face-related utilities."""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import tempfile
import time
import traceback

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from functools import cache, lru_cache
from os.path import dirname, exists
from typing import Any, Iterable, NamedTuple, Optional, Sequence, Union
from urllib.request import urlretrieve

import numpy as np
import requests

from PIL import Image

logger = logging.getLogger(__name__)

def dl_image(url: str, out_path: str='') -> str:
    """Downloads an image `url` to local path if needed, returning the local path.

    If the url doesn't start with 'http' or 'data:', we assume it's a local path and return it as is
    (we do NOT copy it to `out_path`).

    If `out_path` is not provided, we created a named temporary file, but we do NOT delete it, so
    you have to do it yourself.
    """
    if not url.startswith('http') or not url.startswith('data:'):
        return url
    ext = url.split('.')[-1]
    if not out_path:
        with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as f:
            out_path = f.name
    urlretrieve(img, out_path)
    return out_path

class FaceDetectionResult(NamedTuple):
    """Result of a face detection operation."""
    orig_w: int # original image width
    orig_h: int # original image height
    resized_w: int # resized image width
    resized_h: int # resized image height
    boxes: list[tuple[float, float, float, float]]
    mask_urls: list[str]
    elapsed: float=0.0
    mask_paths: Optional[list[str]] = None
    crop_paths: Optional[list[str]] = None


class FaceSystem(ABC):
    """Generic face system interface, encompassing detection, alignment, embeddings, etc.

    For convenience, we allow image inputs to either be local paths or urls, but in practice if they
    are urls, we always download them first (to a temporary file) before processing.

    """
    def __init__(self, batch_size: int=10, max_size: int=-1, **kw: Any):
        """Set some parameters:

        - `batch_size` - batch size for face detection.
        - `max_size` - maximum size for input images (thumbnail down if larger).
        - `kw` - additional keyword arguments are stored as instance attributes.

        This class maintains some state:
        - `pool` - a ThreadPoolExecutor if `batch_size` > 1, otherwise None.
        - `temp_files` - a list of temporary files that will be deleted when the object is deleted.
        """
        self.batch_size = batch_size
        #self.pool = ThreadPoolExecutor(max_workers=batch_size) if batch_size > 1 else None
        self.pool = ThreadPoolExecutor(max_workers=max(batch_size, 1))
        self.max_size = max_size
        self.__dict__.update(kw)
        self.temp_files: list[str] = []

    def __del__(self):
        """Various cleanup"""
        # delete temp files
        for f in self.temp_files:
            try:
                os.remove(f)
            except Exception as e:
                logger.error(f"Error deleting temp file {f}: {e}")
        # kill our pool if we have one
        if self.pool:
            self.pool.shutdown(wait=False)

    @abstractmethod
    def detect_faces(self, images: list[str], **kw: Any) -> Iterable[FaceDetectionResult]:
        """Detect faces in images, and optionally crop/etc.

        The `images` can be a list of image urls or local paths.

        This method is a wrapper around the implementation `_detect_faces()`, which is the one you
        should override in your subclass.
        """
        pass

    def preprocess_image(self, url: str) -> dict[str, Any]:
        """This takes an image url (which might be a local path) and does various preprocessing:

        - downloads the image if it's a url
        - resizes the image if `max_size` is set

        This will update self.temp_files with the temporary files created during processing.

        Returns a dict with:
        - `img_path` - final image path to use for real processing
        - `orig_img_path` - original (downloaded) image path
        - `orig_w` - original image width
        - `orig_h` - original image height
        - `resized_w` - resized image width
        - `resized_h` - resized image height
        """
        t0 = time.time()
        path = dl_image(url)
        if path != url:
            self.temp_files.append(path)
        # at this point, path is a local path
        img = Image.open(path)
        ret = dict(orig_img_path=path, orig_w=img.size[0], orig_h=img.size[1])
        # thumbnailing if requested
        if self.max_size > 0:
            img.thumbnail((self.max_size, self.max_size))
            with tempfile.NamedTemporaryFile(suffix=f'.{path.split(".")[-1]}', delete=False) as f:
                path = f.name
                img.save(path)
                self.temp_files.append(path)
        del img
        # at this point, path is a local path that's been thumbnailed if requested
        W, H = Image.open(path).size
        ret.update(img_path=path, resized_w=W, resized_h=H)
        t1 = time.time()
        logger.info(f'Preprocessed {url} to {ret} in {t1-t0:.2f}s')
        return ret

    def make_and_add_path(self, fmt, i, key, ret):
        """Helper to make a path from `fmt` for index `i` and append it to `ret[key]`"""
        path = fmt % i
        try:
            os.makedirs(dirname(path), exist_ok=True)
        except Exception:
            pass
        ret.setdefault(key, []).append(path)
        return path


class ReplicateFaceSystem(FaceSystem):
    """Replicate face system interface."""
    def detect_faces(self,
                     images: list[str],
                     mask_fmts: Optional[list[str]]=None,
                     crop_fmts: Optional[list[str]]=None,
                     crop_size: int=-1,
                     ) -> Iterable[Union[FaceDetectionResult, Exception]]:
        """Detect faces in images, and optionally crop/etc.

        - `mask_fmts` - list of mask format strings to save to, with a %d for face number
        - `crop_fmts` - list of crop format strings to save to, with a %d for face number
        - `crop_size` - size to crop faces to if > 0. Note that this will upscale as well as

        Note that if the input images are urls, then the images will be downloaded locally, as we
        need their size to normalize the output bounding box coordinates.

        If there's more than 1 input and `batch_size` is greater than 1, this will call things in
        parallel using a ThreadPoolExecutor with `batch_size` threads.

        If `max_size` is > 0, it will thumbnail the input images to be at most that size before running
        the face detection.

        The underlying Replicate method returns a list of URLs to the face masks, as well as bounding
        boxes for the faces detected in the image. In contrast, this function normalizes bounding boxes
        to (x0, y0, x1, y1) where each coordinate is specified in fractional coordinates.

        This function lets you specify `mask_fmts`: string to download the masks to. The fmt string
        should contain a %-compatible placeholder for the face number within that image (0-indexed).
        Parent dirs will be created. If `None`, masks are not downloaded. Note that the mask image will
        be the size of the cropped input.

        Similarly, you can specify `crop_fmts`: strings to save cropped faces from the original images
        (before cropping), using the returned bounding box. This should contain a %-compatible
        placeholder for the face number. Parent dirs will be created. If `crop_size` is > 0, the crops
        will be resized to that size. Note that unlike the `max_size` parameter, this is not only
        thumbnailing down to that size, but also upscaling smaller images to that size, as well as
        making it square (i.e., not preserving aspect ratio). If `None`, crops are not saved.

        This function yields results in the same order as inputs, where each result is a
        `FaceDetectionResult` tuple containing the bounding boxes and mask URLs for each face detected,
        as well as the paths to the downloaded masks and crops if they were requested. If there was any
        error processing an image, its result will be an Exception object.
        """
        if mask_fmts is None:
            mask_fmts = [''] * len(images)
        if crop_fmts is None:
            crop_fmts = [''] * len(images)

        # run as futures, so we can put exceptions in the output if needed
        futures = [self.pool.submit(self._detect_face, img=img, mask_fmt=mask_fmt,
                                    crop_fmt=crop_fmt, crop_size=crop_size)
                       for img, mask_fmt, crop_fmt in zip(images, mask_fmts, crop_fmts)]
        for future in futures:
            try:
                yield future.result()
            except Exception as e:
                yield e

    def _detect_face(self, img: str, mask_fmt: str, crop_fmt: str, crop_size: int) -> FaceDetectionResult:
        """Actual processing"""
        print(f'Starting detect face with {img}, {mask_fmt}, {crop_fmt}, {crop_size}')
        from llm.replicate_wrapper import face_detection
        input = self.preprocess_image(img)
        raw = face_detection(input['img_path'])
        boxes, mask_urls = raw['boxes'], raw['mask_urls']
        # normalize the boxes to fractions of the resized input
        W, H = input['resized_w'], input['resized_h']
        boxes = [(x/W, y/H, (x+w)/W, (y+h)/H) for x, y, w, h in boxes]
        ret = dict(boxes=boxes, mask_urls=mask_urls, orig_w=input['orig_w'], orig_h=input['orig_h'],
                   resized_w=W, resized_h=H)
        for i, (box, mask_url) in enumerate(zip(boxes, mask_urls)):
            if mask_fmt:
                mask_path = self.make_and_add_path(mask_fmt, i, 'mask_paths', ret)
                dl_image(mask_url, mask_path)
                logger.debug(f'Downloaded mask {i} to {mask_path}')
            if crop_fmt:
                crop_path = self.make_and_add_path(crop_fmt, i, 'crop_paths', ret)
                with Image.open(input['orig_img_path']) as image:
                    W, H = image.size
                    int_box = tuple(int(coord*dim) for coord, dim in zip(box, [W, H, W, H]))
                    assert len(int_box) == 4
                    crop = image.crop(int_box)
                    if crop_size > 0:
                        crop = crop.resize((crop_size, crop_size), Image.LANCZOS)
                    crop.save(crop_path)
                    logger.debug(f'Saved crop {i} to {crop_path} with size {crop.size}')
        return FaceDetectionResult(**ret)


class RetinaFaceSystem(FaceSystem):
    """RetinaFace face detection and alignment system."""
    def __init__(self, threshold: float=0.9, **kw: Any) -> None:
        """Set some parameters:

        `threshold` - face detection threshold.
        """
        super().__init__(threshold=threshold, **kw)

    def retina_extract_faces(
        self,
        img_path: Union[str, np.ndarray],
        threshold: float = 0.9,
        model: Any = None, #TODO this was originally Optional[Model], but we don't have it
        align: bool = True,
        allow_upscaling: bool = True,
        expand_face_area: int = 0,
        ) -> list[dict[str, object]]:
        """
        Detect, crop, and optionally align faces.

        Note that this is code copied from the retinaface library, with some modifications to reduce
        duplicate work, and return all outputs.

        Args:
        - img_path (str or numpy): given image
        - threshold (float): detection threshold
        - model (Model): pre-trained model can be passed to the function
        - align (bool): enable or disable alignment
        - allow_upscaling (bool): allowing up-scaling
        - expand_face_area (int): expand detected facial area with a percentage
        """
        from retinaface.RetinaFace import detect_faces, postprocess, preprocess
        resp: list[dict[str, object]] = []

        # ---------------------------

        img = preprocess.get_image(img_path)

        # ---------------------------

        obj = detect_faces(
            img_path=img, threshold=threshold, model=model, allow_upscaling=allow_upscaling
        )

        if not isinstance(obj, dict):
            return resp

        for _, identity in obj.items():
            facial_area = identity["facial_area"]
            rotate_angle = 0
            rotate_direction = 1

            x = facial_area[0]
            y = facial_area[1]
            w = facial_area[2] - x
            h = facial_area[3] - y

            if expand_face_area > 0:
                expanded_w = w + int(w * expand_face_area / 100)
                expanded_h = h + int(h * expand_face_area / 100)

                # overwrite facial area
                x = max(0, x - int((expanded_w - w) / 2))
                y = max(0, y - int((expanded_h - h) / 2))
                w = min(img.shape[1] - x, expanded_w)
                h = min(img.shape[0] - y, expanded_h)

            unaligned = img[y : y + h, x : x + w]
            resp.append(dict(cropped=unaligned, w=img.shape[1], h=img.shape[0], **identity))
            if align is True:
                facial_img = unaligned.copy()
                landmarks = identity["landmarks"]
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                nose = landmarks["nose"]
                mouth_right = landmarks["mouth_right"]
                mouth_left = landmarks["mouth_left"]

                # notice that left eye of one is seen on the right from your perspective
                aligned_img, rotate_angle, rotate_direction = postprocess.alignment_procedure(
                    img=img, left_eye=right_eye, right_eye=left_eye, nose=nose
                )

                # find new facial area coordinates after alignment
                rotated_x1, rotated_y1, rotated_x2, rotated_y2 = postprocess.rotate_facial_area(
                    (x, y, x + w, y + h), rotate_angle, rotate_direction, (img.shape[0], img.shape[1])
                )
                facial_img = aligned_img[
                    int(rotated_y1) : int(rotated_y2), int(rotated_x1) : int(rotated_x2)
                ]
                resp[-1]['aligned'] = facial_img
        return resp

    def detect_faces(self, images: list[str]) -> Iterable[FaceDetectionResult]:
        """Detect faces in images, and optionally crop/etc."""
        for img in images:
            input = self.preprocess_image(img)
            raw = self.retina_extract_faces(input['img_path'], threshold=self.threshold, align=True)
            # normalize the boxes to fractions of the resized input

FACE_DETECTORS = dict(
    replicate=ReplicateFaceSystem,
    retina=RetinaFaceSystem,
)

if __name__ == '__main__':
    parser = ArgumentParser(description="Test face detection on images")
    parser.add_argument('images', nargs='+', help='Images to run face detection on')
    parser.add_argument('-d', '--detector', default='retina', choices=FACE_DETECTORS.keys(),
                        help=f'Which face detector to use [{", ".join(FACE_DETECTORS)}]')
    args = parser.parse_args()
    kwargs = {}
    if args.detector == 'replicate':
        kwargs.update(dict(mask_fmts=[f'{i}-mask_%d.png' for i, _ in enumerate(args.images)],
                           crop_fmts=[f'{i}-crop_%d.png' for i, _ in enumerate(args.images)],
                           crop_size=256))

    detector = FACE_DETECTORS[args.detector]()
    for img, result in zip(args.images, detector.detect_faces(images=args.images, **kwargs)):
        if isinstance(result, Exception):
            print(f'Error processing {img}: {result}')
            # print traceback
            traceback.print_exc()
        else:
            print(f'Found {len(result)} faces in {img} using {args.detector}')
            print(result)
