"""Image feature extraction"""

import json
import logging
import os
import sys
import time
from argparse import ArgumentParser
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


class VGGFeatureExtractor(nn.Module):
    """Extracts high-level image features using the VGG network.

    This broadly follows:
    https://towardsdatascience.com/image-feature-extraction-using-pytorch-e3b327c3607a
    """

    def __init__(self, model=None):
        super(VGGFeatureExtractor, self).__init__()
        if model is None:
            model = models.vgg16(pretrained=True)
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]

    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


class HEDFeatureExtractor(nn.Module):
    """Extracts low-level image edge features using the HED technique"""

    def __init__(self):
        super(HEDFeatureExtractor, self).__init__()
        self.net = torchHED.hed.Network()
        if torch.cuda.is_available():
            self.net.cuda()
        self.net.eval()

    def forward(self, img):
        """Processes an image in pytorch format"""

        def process_img(img):
            img = T.Resize((320, 480))(img)
            out = torchHED.hed._estimate(img, use_cuda=True, net=self.net)
            return torch.flatten(T.Resize((75, 75))(out))

        if len(img.shape) == 3:
            return torch.tensor([process_img(img)])
        elif len(img.shape) == 4:
            return torch.stack([process_img(_img) for _img in img])
        else:
            print(img.shape, type(img))
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


#: mapping from model type string to initializer
MODEL_TYPES = {
    "vgg": VGGFeatureExtractor,
    "hed": HEDFeatureExtractor,
}


if __name__ == "__main__":
    LOG_FORMAT = "%(asctime)s.%(msecs)03d\t%(filename)s:%(lineno)s\t%(funcName)s\t%(message)s"
    logging.basicConfig(format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    parser = ArgumentParser(description="Image feature extractor")
    parser.add_argument("model_type", choices=MODEL_TYPES, help="which model type to create")
    parser.add_argument(
        "image_list_path", help="path to read list of images from, or - to read from stdin"
    )
    parser.add_argument("output_path", help="path to write outputs, in gensim format")
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
    model = MODEL_TYPES[args.model_type]()
    if args.skip_existing:
        try:
            kv = KeyedVectors.load(args.output_path, mmap="r")
            to_skip = kv.index_to_key
        except Exception:
            to_skip = None
    else:
        to_skip = None
    extract_image_features_impl(
        model,
        ImageListDataset(args.image_list_path, to_skip=to_skip),
        args.output_path,
        n_workers=args.n_workers,
        batch_size=args.batch_size,
        save_incr=args.save_incr,
        show_progress=args.verbose,
    )
