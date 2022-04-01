"""Image feature extraction.

There are a few main classes:
    - ImageNetFeatureExtractor is the main class for imagenet-trained classifiers. You specify a
      specific model name and list of named layers to extract from that model.
    - HEDFeatureExtractor is for extracting edge-based features using the Holistically-nested Edge
      Detection (HED) model. This requires no parameters.
"""

import json
import logging
import os
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from os.path import exists
from random import sample, shuffle
from typing import Any, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

import numpy as np  # type: ignore
import psutil  # type: ignore
import torch
import torchHED.hed  # type: ignore
import torchvision.transforms as T  # type: ignore
from gensim.models import KeyedVectors  # type: ignore
from PIL import Image  # type: ignore
from sklearn.exceptions import NotFittedError  # type: ignore
from sklearn.random_projection import SparseRandomProjection  # type: ignore
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models  # type: ignore
from tqdm import tqdm  # type: ignore

#: disable gensim log messages
logging.getLogger("gensim").setLevel(logging.WARNING)

IMAGE_DIMS = 4096
N_WORKERS = os.cpu_count() or 1  # in case we get None
BATCH_SIZE = psutil.virtual_memory().total // (1024 * 1024 * 1024)  # assume 1 gig needed per batch
SAVE_INCR = 10  # number of batches to save after


def transform_image(img):
    """Transforms `img` suitably for classifiers"""
    transforms = [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if 0:
        vgg16_transform = torch.nn.Sequential(*transforms)
    else:
        vgg16_transform = T.Compose(transforms)
    return vgg16_transform(img)


def load_image(path: str) -> Tuple[str, Any]:
    """Returns `(path, transformed_image)`"""
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        logging.error("Got %s when loading %s: %s", type(e), path, e)
        return (path, None)
    ret = transform_image(img)
    logging.debug("Returning image of shape %s -> %s", img.size, ret.shape)
    return (path, ret)


class ImageListDataset(Dataset):
    """A simple dataset that reads a list of image paths as inputs"""

    def __init__(self, image_list_path: str, to_skip: Optional[Sequence[str]] = None):
        super().__init__()
        if image_list_path == "-":
            self.images = [l.strip() for l in sys.stdin]
        else:
            with open(image_list_path) as f:
                self.images = [l.strip() for l in f if exists(l.strip())]
        orig_length = len(self.images)
        skip_set = set()
        if to_skip is not None:
            skip_set = set(to_skip)
            self.images = [im for im in self.images if im not in skip_set]
        logging.info(
            "Read %d image paths from %s, down to %d after skipping %d",
            orig_length,
            image_list_path,
            len(self.images),
            len(skip_set),
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return load_image(self.images[idx])


#: mapping from model names to initializers
MODELS_BY_NAME = dict(vgg16=models.vgg16)


class ImageNetFeatureExtractor(nn.Module):
    """Extracts high-level image features using a network trained on ImageNet.

    You specify a model or model name, as well as output layer names when you initialize this. When
    you run `forward()` on the input data, it will store the features after each named output layer,
    and finally concatenate them all and return the result.

    VGG16 consists of 3 main chunks:
    1. features - which is repeated conv -> relu -> maxpool
    2. avgpool - adaptive avg pooling of output size 7x7 (and 512 feature layers)
    3. classifier - 2 fully connected layers with relu and dropout (out dims=4096),
                    followed by final classification (1000 dims)

    """

    def __init__(self, out_layers=["fc6"], model_name="vgg16", projection_length=None):
        """Initialize this extractor with given list of `out_layers`.

        You must pass in a `model_name` (one of ['vgg16'] for now).

        You can optionally pass in a `projection_length`. If given, we construct a random matrix
        from original output size (after concatenating all output layers) to given length. This is
        used to project the output to this length. This is most useful for dimensionality reduction.
        """
        super(ImageNetFeatureExtractor, self).__init__()
        self.out_layers = out_layers
        self.model_name = model_name
        self.model = MODELS_BY_NAME[model_name](pretrained=True)
        for layer in out_layers:
            self.layer_by_name(layer).register_forward_hook(self.make_hook(layer))
        # flattener to reuse in our feature extractor
        self.flatten = nn.Flatten()
        # where we'll store our outputs (rewritten in every forward pass)
        self.features_by_layer = defaultdict(list)
        # initialize projection
        if projection_length is not None:
            self.projection = SparseRandomProjection(
                n_components=projection_length, density=1 / 3.0, dense_output=True, random_state=0
            )
        else:
            self.projection = None

    def layer_by_name(self, name):
        """Returns the layer from our model for the given layer `name`"""
        model = self.model
        if self.model_name == "vgg16":
            return dict(
                conv1=model.features[2],
                conv2=model.features[7],
                conv3=model.features[14],
                conv4=model.features[21],
                conv5=model.features[28],
                maxpool=model.features[30],
                avgpool=model.avgpool,
                fc6=model.classifier[0],
                fc7=model.classifier[3],
                fc8=model.classifier[6],
            )[name]
        else:
            raise NotImplementedError(f"Cannot extract from model name {self.model_name}")

    def make_hook(self, name):
        """Makes a hook to store features from given layer `name`"""

        def hook(model, input, output):
            flat = self.flatten(output)
            self.features_by_layer[name].extend(flat)

        return hook

    def forward(self, x):
        """We take our input `x` (image or images) and return a feature vector (or matrix)"""
        self.features_by_layer = defaultdict(list)
        out = self.model(x)
        ret = torch.hstack(
            [torch.vstack(self.features_by_layer[layer]) for layer in self.out_layers]
        )
        # print(ret, ret.shape)
        assert len(ret.shape) == 2
        n, n_dims = ret.shape
        assert len(x) == n
        if self.projection:
            try:
                ret = self.projection.transform(ret)
            except NotFittedError:
                ret = self.projection.fit_transform(ret)
            # print(ret, ret.shape)
        return out


class HEDFeatureExtractor(nn.Module):
    """Extracts low-level image edge features using the HED technique

    This runs the HED edge detector on input images, resizes the output to a fixed size, and returns
    a flattened version of that image as the features.

    Note that the current HED implementation requires the input be of size 480 x 320, so we first
    forcibly resize our input to that size.
    """

    def __init__(self, output_size=(75, 75)):
        super(HEDFeatureExtractor, self).__init__()
        self.net = torchHED.hed.Network()
        if torch.cuda.is_available():
            self.net.cuda()
        self.net.eval()
        self.output_size = output_size

    def process_img(self, img):
        """Actual extract implementation on a single `img`"""
        img = T.Resize((320, 480))(img)
        out = torchHED.hed._estimate(img, use_cuda=True, net=self.net)
        return torch.flatten(T.Resize(self.output_size)(out))

    def forward(self, input):
        """Processes `input` to get HED features.

        The input should either be a single image (3 channels) or list of images (4 channels), in
        pytorch format.
        """
        if len(input.shape) == 3:
            return torch.tensor([self.process_img(input)])
        elif len(input.shape) == 4:
            return torch.stack([self.process_img(img) for img in input])
        else:
            print(input.shape, type(input))
            raise ValueError("Input must be either 3 or 4-dimensional!")


def extract_image_features_impl(
    model: nn.Module,
    dataset: Dataset,
    output_path: str = "",
    n_workers: int = N_WORKERS,
    batch_size: int = BATCH_SIZE,
    show_progress: bool = False,
    save_incr: int = SAVE_INCR,
) -> np.ndarray:
    """Extracts image features for all images in `dataset`"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=n_workers)
    try:
        kv = KeyedVectors.load(output_path)
    except Exception:
        kv = None

    def update_and_save(paths: List[str], features: List[np.ndarray]) -> None:
        """Updates and saves the given list of `paths` and `features`, clearing them after"""
        if not paths:
            return
        assert len(paths) == len(features)
        nonlocal kv
        np_features = np.array(features)
        if kv is None:
            kv = KeyedVectors(vector_size=np_features.shape[1], count=0, dtype=np_features.dtype)
        kv[paths] = np_features
        if output_path:
            kv.save(output_path)
        del paths[:]
        del features[:]

    all_paths = []
    cur_paths, cur_features = [], []
    batches_done = 0
    tqdm_kwargs = dict(disable=not show_progress)
    for i, (paths, batch) in enumerate(tqdm(loader, **tqdm_kwargs)):
        batch = batch.to(device)
        logging.debug("got batch of shape %s, %s", batch.shape)
        all_paths.extend(paths)
        cur_paths.extend(paths)
        with torch.no_grad():
            cur = model(batch)
            logging.debug("got features of shape %s", cur.shape)
            # Convert to NumPy Array, Reshape it, and save it to features variable
            features = cur.cpu().detach().numpy()
            cur_features.extend(features)
        batches_done += 1
        if batches_done == save_incr:
            update_and_save(cur_paths, cur_features)
            batches_done = 0
    update_and_save(cur_paths, cur_features)
    if not all_paths:
        return np.array([])
    ret = kv[all_paths]
    logging.info("Got final features of shape %s", ret.shape)
    return ret


if __name__ == "__main__":
    model_names = sorted(MODELS_BY_NAME) + ["hed"]
    LOG_FORMAT = "%(asctime)s.%(msecs)03d\t%(filename)s:%(lineno)s\t%(funcName)s\t%(message)s"
    logging.basicConfig(format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    parser = ArgumentParser(description="Image feature extractor")
    parser.add_argument("model_name", choices=model_names, help="which model type to create")
    parser.add_argument(
        "image_list_path", help="path to read list of images from, or - to read from stdin"
    )
    parser.add_argument("output_path", help="path to write outputs, in gensim format")
    parser.add_argument("-l", "--layers", help="comma-separated list of layers to extract")
    parser.add_argument(
        "-d",
        "--output_size",
        type=int,
        default=0,
        help="either the random projection output size for imagenet-based features, or one side of the square output of HED",
    )
    parser.add_argument(
        "-s", "--skip_existing", action="store_true", help="if set, then skip existing paths"
    )
    parser.add_argument(
        "-n", "--n_workers", type=int, default=N_WORKERS, help=f"number of workers [{N_WORKERS}]"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=BATCH_SIZE, help=f"batch_size [{BATCH_SIZE}]"
    )
    parser.add_argument(
        "-i",
        "--save_incr",
        type=int,
        default=SAVE_INCR,
        help=f"number of batches to save after, or <=0 to disable [{SAVE_INCR}]",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help=f"if set, then show progress")
    args = parser.parse_args()
    # initialize model class
    model: nn.Module
    if args.model_name == "hed":
        output_size = args.output_size or 75
        model = HEDFeatureExtractor(output_size=(output_size, output_size))
    else:
        kwargs = dict(out_layers=args.layers.split(","), model_name=args.model_name)
        if args.output_size:
            kwargs["projection_length"] = args.output_size
        model = ImageNetFeatureExtractor(**kwargs)
    # get list of done paths if we want to skip them
    if args.skip_existing:
        try:
            kv = KeyedVectors.load(args.output_path, mmap="r")
            to_skip = kv.index_to_key
        except Exception:
            to_skip = None
    else:
        to_skip = None
    # main extraction function
    extract_image_features_impl(
        model=model,
        dataset=ImageListDataset(args.image_list_path, to_skip=to_skip),
        output_path=args.output_path,
        n_workers=args.n_workers,
        batch_size=args.batch_size,
        save_incr=args.save_incr,
        show_progress=args.verbose,
    )
