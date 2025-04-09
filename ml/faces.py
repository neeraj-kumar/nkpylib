"""Face-related utilities.

The main thing in this module is the `FaceSystem` class, which is an abstract class that defines
a face detector + related functionality. There are a few implementations of this class, including
one using mediaface (running either on Replicate or locally via docker at a given port); and a
much better one (RetinaFace), also running locally (within this process).

You initialize the system once, and then you can call `detect_faces()` on a list of images (urls or
local paths), and it will return a list of `FaceDetectionResult` objects, which contain the face
boxes and other info.

You can also optionally specify the output formats for different types of outputs (masks, crops,
etc.) using the `all_output_fmts` parameter. This should be a list of the same length as `images`,
each containing a mapping from output type to an output format string. The output types are usually
things like 'mask', 'crop', etc. The format string should contain a %-compatible placeholder for
the face number within that.

Note that different implementations might support different kinds of outputs, but we provide
`crop` with all methods, which simply uses the detected face boxes.

The face systems can also all do downscaling of inputs (mainly for speed).
"""

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

# define a type for input url, which can be a string, PIL Image, or numpy array
ImageT = Union[str, Image.Image, np.ndarray]

def dl_image(url: ImageT, out_path: str='') -> str:
    """Downloads an image `url` to local path if needed, returning the local path.

    This can handle urls in the following formats/sources:
    - url (starting with 'http')
    - data url (starting with 'data:')
    - local path (other strings)
    - PIL Image
    - numpy array

    If the url is a local path, we return it as is (we do NOT copy it to `out_path`).

    If `out_path` is not provided, we created a named temporary file, but we do NOT delete it, so
    you have to do it yourself.

    If it's a PIL image or numpy array, we save it as a png.
    """
    def get_out_path(ext: str) -> str:
        """Gets the output path with given ext"""
        nonlocal out_path
        ext = ext.lstrip('.')
        if not out_path:
            with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as f:
                out_path = f.name
        return out_path

    if isinstance(url, str):
        if not url.startswith('http') or not url.startswith('data:'): # local path
            return url
        ext = url.split('.')[-1]
        out_path = get_out_path(ext)
        urlretrieve(img, out_path)
    elif isinstance(url, Image.Image):
        out_path = get_out_path('png')
        url.save(out_path)
    elif isinstance(url, np.ndarray):
        assert len(url.shape) in (2, 3), f'Expected 2D or 3D array, got shape {url.shape}'
        assert url.dtype == np.uint8, f'Expected uint8 array, got dtype {url.dtype}'
        out_path = get_out_path('png')
        Image.fromarray(url).save(out_path)
    return out_path


class FaceDetectionResult(NamedTuple):
    """Result of a face detection operation."""
    input_url: str # input image url or path (or '' if it was a raw image)
    orig_w: int # original image width
    orig_h: int # original image height
    resized_w: int # resized image width
    resized_h: int # resized image height
    boxes: list[tuple[float, float, float, float]] # list of face boxes in fractional (x0, y0, x1, y1) coords
    timings: dict[str, float] # timings for different parts of the process
    output_paths: Optional[dict[str, list[str]]] = None # output paths by type (masks, crops, etc.)


class FaceSystem(ABC):
    """Generic face system interface, encompassing detection, alignment, embeddings, etc.

    For convenience, we allow image inputs to either be local paths or urls, but in practice if they
    are urls, we always download them first (to a temporary file) before processing.

    """
    def __init__(self, batch_size: int=10, input_max_size: int=-1, crop_size: int=-1, **kw: Any):
        """Set some parameters:

        - `batch_size` - batch size for face detection.
        - `input_max_size` - maximum size for input images (thumbnail down if larger).
        - `crop_size` - size to crop faces to if > 0 (upscale as well as down, square outputs).
        - `kw` - additional keyword arguments are stored as instance attributes.

        This class maintains some state:
        - `pool` - a ThreadPoolExecutor if `batch_size` > 1, otherwise None.
        - `temp_files` - a list of temporary files that will be deleted when the object is deleted.
        """
        self.batch_size = batch_size
        self.pool = ThreadPoolExecutor(max_workers=batch_size) if batch_size > 1 else None
        #self.pool = ThreadPoolExecutor(max_workers=max(batch_size, 1))
        self.input_max_size = input_max_size
        self.crop_size = crop_size
        self.__dict__.update(kw)
        self.temp_files: list[str] = []

    def __del__(self):
        """Various cleanup"""
        # delete temp files
        for f in self.temp_files:
            try:
                os.remove(f)
            except Exception:
                pass
        # kill our pool if we have one
        if self.pool:
            self.pool.shutdown(wait=False)

    def detect_faces(self,
                     images: list[ImageT],
                     all_output_fmts: Optional[list[Optional[dict[str, str]]]]=None,
                     ) -> Iterable[Union[FaceDetectionResult, Exception]]:
        """Detect faces in images, and optionally crop/etc.

        Yields a `FaceDetectionResult` for each image in `images`, unless there's an error (for that
        image), in which case it yields the exception.

        The `images` can be a list of image urls, data urls, local paths, PIL images, or np arrays.
        If not already local paths, then they are downloaded/saved to temp local paths.

        You can specify the output formats for different types of outputs (masks, crops, etc.) using
        the `all_output_fmts` parameter. This should be a list of the same length as `images`, each
        containing a mapping from output type to an output format string. The output types are
        usually things like 'mask', 'crop', etc. The format string should contain a %-compatible
        placeholder for the face number within that.

        Note that the crops are generated from the original (not resized) image for maximum quality.
        If self.crop_size > 0, the crops will be resized to a square image of that size (including
        down- and up-scaling). Other outputs must be generated from the underlying implementation
        and will usually be the same as the resized inputs.

        This method is a generic wrapper around the implementation `_detect_face()`, which is the
        one you should override in your subclass. This wrapper handles preprocessing,
        postprocessing, and parallelization if needed.

        If there's more than 1 input and `self.batch_size` is greater than 1, this will call things
        in parallel using our thread pool.
        """
        if not all_output_fmts:
            all_output_fmts = [None] * len(images)
        inputs = zip(images, all_output_fmts)
        if self.pool and len(images) > 1:
            # run as futures, so we can put exceptions in the output if needed
            futures = [self.pool.submit(self._detect_face, *args) for args in inputs]
            for future in futures:
                try:
                    yield future.result()
                except Exception as e:
                    yield e
        else:
            for args in inputs:
                try:
                    yield self._detect_face(*args)
                except Exception as e:
                    yield e

    @abstractmethod
    def _detect_face(self, img: ImageT, output_fmts: Optional[dict[str, str]]=None) -> FaceDetectionResult:
        """Detects faces in a single `img`, saving outputs as needed based on `output_fmts`"""
        pass

    def preprocess_image(self, url: ImageT) -> dict[str, Any]:
        """This takes an image input url and does various preprocessing:

        - downloads the image if it's a url or a raw image (PIL/numpy)
        - resizes the image if `input_max_size` is set

        This will update self.temp_files with the temporary files created during processing.

        Returns a dict with:
        - `input_url` - original image path/url
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
        ret['input_url'] = url if isinstance(url, str) else ''
        # thumbnailing if requested
        if self.input_max_size > 0:
            img.thumbnail((self.input_max_size, self.input_max_size))
            with tempfile.NamedTemporaryFile(suffix=f'.{path.split(".")[-1]}', delete=False) as f:
                path = f.name
                img.save(path)
                self.temp_files.append(path)
        del img
        # at this point, path is a local path that's been thumbnailed if requested
        W, H = Image.open(path).size
        ret.update(img_path=path, resized_w=W, resized_h=H)
        t1 = time.time()
        logger.debug(f'Preprocessed {url} to {ret} in {t1-t0:.2f}s')
        return ret

    def postprocess_faces(self,
                          input: dict[str, Any],
                          boxes: list[tuple[int, int, int, int]],
                          output_fmts: Optional[dict[str, str]]=None,
                          output_urls: Optional[dict[str, list[str]]]=None) -> FaceDetectionResult:
        """Does various postprocessing on faces and returns a dict of values.

        Args:
        - input: dict with input image info (from `preprocess_image()`)
        - boxes: list of face boxes in absolute coords (x0, y0, x1, y1) format
        - output_fmts: dict of output format strings for different types of outputs
          - Output types are 'mask', 'crop', 'aligned'
          - Each value should be a format string with a %d for the face number
        - output_urls: list of URLs for the outputs of different types.
          - These are saved to disk using the `output_fmts` given for that output type (if given)
          - These can be anything `dl_image()` can process

        The output is a `FaceDetectionResult`.
        """
        ret: dict[str, Any] = dict(output_paths={}, timings={})
        for field in 'input_url orig_w orig_h resized_w resized_h'.split():
            ret[field] = input[field]
        # normalize the boxes to fractions of the resized input
        W, H = input['resized_w'], input['resized_h']
        ret['boxes'] = boxes = [(x0/W, y0/H, x1/W, y1/H) for x0, y0, x1, y1 in boxes]
        # if we don't have any output formats we're done
        if not output_fmts:
            return FaceDetectionResult(**ret) # type: ignore
        if not output_urls:
            output_urls = {}
        for i, box in enumerate(boxes):
            for output_type, fmt in output_fmts.items():
                t0 = time.time()
                if not fmt:
                    continue
                url: Optional[ImageT] = output_urls.get(output_type, [''] * len(boxes))[i]
                if url is None or (isinstance(url, str) and not url):
                    # special case for `crop`: do the actual cropping
                    if output_type == 'crop':
                        with Image.open(input['orig_img_path']) as image:
                            W, H = image.size
                            int_box = tuple(int(coord*dim) for coord, dim in zip(box, [W, H, W, H]))
                            assert len(int_box) == 4
                            crop = image.crop(int_box)
                            if self.crop_size > 0: # forcibly resize crop
                                crop = crop.resize((self.crop_size, self.crop_size), Image.LANCZOS)
                            url = crop
                    else:
                        continue
                # at this point we definitely have a fmt and a url
                path = fmt % i
                try:
                    os.makedirs(dirname(path), exist_ok=True)
                except Exception:
                    pass
                dl_image(url, path)
                t1 = time.time()
                ret['output_paths'].setdefault(output_type, []).append(path)
                ret['timings'].setdefault(output_type, 0)
                ret['timings'][output_type] += t1 - t0
                logger.debug(f'Downloaded {output_type} {i} to {path}')
        return FaceDetectionResult(**ret)


class ReplicateFaceSystem(FaceSystem):
    """Replicate face system interface.

    Notes:
    - The only extra outputs are mask images, but these are the same size as the resized images, not
      originals.
    """
    def _detect_face(self, img: ImageT, output_fmts: Optional[dict[str, str]]=None) -> FaceDetectionResult:
        """Actual face processing for a single face `img`."""
        from nkpylib.ml.replicate_wrapper import face_detection
        assert isinstance(img, str), f'Expected string, got {type(img)}' # type: ignore[operator]
        logger.info(f'Starting detect face with {img}, {output_fmts}')
        t0 = time.time()
        input = self.preprocess_image(img)
        t1 = time.time()
        raw = face_detection(input['img_path'])
        t2 = time.time()
        ret = self.postprocess_faces(input=input,
                                     boxes=[(x, y, x+w, y+h) for x, y, w, h in raw['boxes']],
                                     output_fmts=output_fmts,
                                     output_urls=dict(mask=raw['mask_urls']))
        t3 = time.time()
        ret.timings.update(preprocess=t1-t0, detect=t2-t1, postprocess=t3-t2, total=t3-t0)
        return ret


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
        ) -> list[dict[str, Any]]:
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
        from retinaface.RetinaFace import detect_faces, postprocess, preprocess # type: ignore
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
                mouth_right = landmarks["mouth_right"] #TODO unused
                mouth_left = landmarks["mouth_left"] #TODO unused

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

    def _detect_face(self, img: ImageT, output_fmts: Optional[dict[str, str]]=None) -> FaceDetectionResult:
        """Actual face processing for a single face `img`."""
        assert isinstance(img, str), f'Expected string, got {type(img)}' # type: ignore[operator]
        logger.debug(f'Starting detect face with {img}, {output_fmts}')
        t0 = time.time()
        input = self.preprocess_image(img)
        t1 = time.time()
        faces = self.retina_extract_faces(input['img_path'], threshold=self.threshold, align=True) # type: ignore
        t2 = time.time()
        # the aligned images are in bgr format, so fix them to rgb
        for f in faces:
            if 'aligned' in f:
                f['aligned'] = f['aligned'][:, :, ::-1]
        ret = self.postprocess_faces(input=input,
                                     boxes=[f['facial_area'] for f in faces],
                                     output_fmts=output_fmts,
                                     output_urls=dict(align=[f['aligned'] for f in faces]))
        t3 = time.time()
        ret.timings.update(preprocess=t1-t0, detect=t2-t1, postprocess=t3-t2, total=t3-t0)
        return ret



FACE_DETECTORS = dict(
    replicate=ReplicateFaceSystem,
    retina=RetinaFaceSystem,
)

if __name__ == '__main__':
    # init logging with fmt including ts, func, name, level, msg
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(funcName)s %(name)s %(levelname)s: %(message)s')
    parser = ArgumentParser(description="Test face detection on images")
    parser.add_argument('images', nargs='+', help='Images to run face detection on')
    parser.add_argument('-d', '--detector', default='retina', choices=FACE_DETECTORS.keys(),
                        help=f'Which face detector to use [{", ".join(FACE_DETECTORS)}]')
    parser.add_argument('-m', '--max_input_size', type=int, default=1000, help='Max input size [1000]')
    parser.add_argument('-c', '--crop_size', type=int, default=256, help='Crop size [256]')
    args = parser.parse_args()
    detector = FACE_DETECTORS[args.detector](input_max_size=args.max_input_size, crop_size=args.crop_size)
    all_output_fmts = []
    for i in range(len(args.images)):
        fmts = {}
        for type in 'mask crop align'.split():
            fmts[type] = f'{args.detector}-{i}-{type}-%d.png'
        all_output_fmts.append(fmts)
    for img, result in zip(args.images, detector.detect_faces(images=args.images,
                                                              all_output_fmts=all_output_fmts)):
        if isinstance(result, Exception):
            print(f'\nError processing {img}: {result}')
            # print the traceback of the exception, which is in the `result` object
            traceback.print_tb(result.__traceback__)
        else:
            print(f'\nFound {len(result.boxes)} faces in {img} using {args.detector}:\n  {result}')
