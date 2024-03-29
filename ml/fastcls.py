"""Generic feature utils for various purposes.

There are a few main things in this module:
- `read_and_join_features()`: a function to read various feature files and concatenate them
- `FastClassifier`: a class to do fast interactive classification of items based on exemplar SVMs
- If you run this module, it starts a server that provides a simple API to do fast interactive
  classification. You should run it in a directory which has a `static` subdirectory, containing css
  and js files with certain names (see command line args for more).

  The API has the following endpoints:
  - / : Loads a simple blank html page with a div with id "main", react, and given css and js files
  - /classify : You can GET this with args 'pos' and 'neg', which should be comma-separated keys of
    the items to use for training the classifier (negatives are optional). It returns a JSON object
    with 'status' (either 'ok' or 'error', and if ok, then 'cls' which contains pairs of (key,
    score) of matching items. Nothing is filtered out in the matches (e.g. the positives used to
    train it, so you have to do that yourself).
  - /static : A simple static file handler for everything in the "static" directory

"""

import json
import logging
import multiprocessing as mp
import os
import re
import time
from argparse import ArgumentParser
from collections import Counter, defaultdict
from os.path import exists
from random import sample
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore
import tornado.ioloop
import tornado.web
from tqdm import tqdm  # type: ignore
from gensim.models import KeyedVectors  # type: ignore
from numpy.random import default_rng
from scipy.spatial.distance import cdist, euclidean

# from sklearn.utils.testing import ignore_warnings
from PIL import Image  # type: ignore
from sklearn.exceptions import ConvergenceWarning  # type: ignore
from sklearn.linear_model import SGDClassifier  # type: ignore
from sklearn.preprocessing import normalize  # type: ignore
from tornado.web import HTTPError, RequestHandler, StaticFileHandler


def show_times(times: List[float]) -> None:
    """Shows times in a pretty-printed way"""
    logging.info(
        "times: %s = %0.3fs",
        "+".join("%0.2fs" % (t1 - t0) for t0, t1 in zip(times, times[1:])),
        times[-1] - times[0],
    )


def read_and_join_features(
    feature_paths: Sequence[str],
    key_func=lambda s: s.strip().replace("\\n", ""),
    max_features=-1,
) -> Tuple[List[str], np.ndarray, Dict[str, List[str]]]:
    """Reads multiple `feature_paths` and joins them together.

    Returns `(keys, features, key_to_item)`, where:
    - `keys` is a list of keys
    - `features` is a 2-d numpy array, where each row corresponds to a `key`
    - `key_to_item` is a dict from key to an item dict. This dict contains:
      - `paths`: the extracted original path for each key, for each feature path
      - any other attributes in the input data files:
        - for gensim inputs, we use the `get_vecattr()` interface to get attributes
        - for npz files, we look for any other arrays of the same length as `features` in the input,
          and use those

    The extension is used to determine what format to read:
        `.wv` or `.kv`: Assumed to be in gensim's KeyedVector format
        `.npz`: Assumed to be an npz file, with 'paths' and 'features' fields.

    All features are concatenated together, and only those keys where all inputs gave a vector are
    used.

    The paths (i.e., keys in .wv files, or the 'path' fields in .npz files) are converted to keys
    using the given `key_func`. These should be unique per path!

    If max_features > 0, then we limit to that many features
    """
    key_to_row = defaultdict(list)
    key_to_item = defaultdict(dict)
    for n, feature_path in enumerate(feature_paths):
        logging.info(
            "Reading features from file %d/%d: %s", n + 1, len(feature_paths), feature_path
        )

        def add_row(path, row, attrs):
            """Adds the given feature `row` for `path`, with optional `attrs`"""
            key = key_func(path)
            # if we already have one set of features, then this key must already be in there
            if n > 0 and key not in key_to_row:
                return
            key_to_row[key].append(row)
            item = key_to_item[key]
            if "paths" not in item:
                item["paths"] = []
            item["paths"].append(path)
            for attr, value in attrs.items():
                if attr == "id":
                    attr = "_id"
                item[attr] = value

        if feature_path.endswith(".wv") or feature_path.endswith(".kv"):
            wv = KeyedVectors.load(feature_path, mmap="r")
            attr_fields = sorted(wv.expandos)
            logging.info(
                "  Read %d wv, attrs: %s, %s", len(wv), sorted(wv.expandos), wv.index_to_key[:10]
            )
            for path in wv.index_to_key:
                attrs = {field: wv.get_vecattr(path, field) for field in attr_fields}
                add_row(path, wv[path], attrs)
                if max_features > 0 and len(key_to_row) >= max_features:
                    break
        elif feature_path.endswith(".npz"):
            data = np.load(feature_path)
            paths = [str(path) for path in data["paths"]]
            features = data["features"]
            attrs_by_field = {}
            for field in data:
                if field in ("paths", "features"):
                    continue
                try:
                    if len(data[field]) == len(features):
                        attrs_by_field[field] = data[field]
                except Exception:  # field that doesn't have len()
                    pass
            for idx, (path, row) in enumerate(zip(paths, features)):
                attrs = {field: attrs_by_field[field][idx] for field in attrs_by_field}
                add_row(path, row, attrs)
                if max_features > 0 and len(key_to_row) >= max_features:
                    break
        else:
            raise NotImplementedError(
                "Do not know how to deal with this filetype: %s" % (feature_path)
            )
    # merge all features together
    features = []
    for key, lst in key_to_row.items():
        if len(lst) == len(feature_paths):
            features.append((key, np.hstack(lst)))
    if not features:
        logging.warning("No valid features found!")
        return None
    keys, features = zip(*features)
    features = np.vstack(features)
    logging.info("Got %d keys and features of shape %s", len(keys), features.shape)
    key_to_item = dict(key_to_item)
    for key, item in key_to_item.items():
        key_to_item[key] = dict(item)
    return keys, features, key_to_item


class FastClassifier:
    """Wrapper class for a fast classifier that uses pre-computed features"""

    def __init__(
        self,
        feature_paths: List[str],
        sqrt_normalize=False,
        l2_normalize=False,
        n_models=4,
        n_top=500,
        n_negatives=50,
        key_func_str=None,
        max_features=-1,
        filter_regexps=None,
    ):
        """Loads the data and preprocesses it.

        Reads all `feature_paths` and concatenates features from each.
        The features are then optionally run through `sqrt_normalize` and/or `l2_normalize` if requested.

        The workings of the classifier are based on the following parameters:
            - n_models: The number of individual models to train
            - n_top: The number of top results from each individual classifier to use
            - n_negatives: The number of random negatives to use for each classifier

        You can optionally pass in a 'key_func_str', which is eval'ed to get the key func
        """
        t0 = time.time()
        if not key_func_str:
            key_func_str = "path"

        def key_func(path):
            return eval(key_func_str)

        keys, features, key_to_item = read_and_join_features(
            feature_paths,
            key_func=key_func,
            max_features=max_features,
        )
        # apply filter regexps
        if filter_regexps:
            logging.info(
                "Initially had %d keys, %s features, %d items",
                len(keys),
                features.shape,
                len(key_to_item),
            )
            to_keep = set()
            # check each item key and each item's values for each filter regexp
            for key, item in key_to_item.items():
                for regexp in filter_regexps:
                    regexp = re.compile(regexp)
                    if regexp.search(key):
                        break
                    matched_field = False
                    for field, value in item.items():
                        if regexp.search(str(value)):
                            logging.debug(f"matched {key} {regexp} {field}={value}")
                            matched_field = True
                            break
                    if matched_field:
                        break
                else:  # none of the regexps matched, so keep it
                    to_keep.add(key)
            # now do the filtering
            key_to_item = {key: item for key, item in key_to_item.items() if key in to_keep}
            keys, features = zip(
                *[(key, vec) for key, vec in zip(keys, features) if key in to_keep]
            )
            features = np.array(features)
        logging.info(
            "Left with %d keys, %s features, %d items",
            len(keys),
            features.shape,
            len(key_to_item),
        )
        self.paths = [key_to_item[key]["paths"][0] for key in keys]
        if sqrt_normalize:
            logging.info("Applying SQRT norm")
            features = np.sqrt(features)
        if l2_normalize:
            logging.info("Applying L2 normalization")
            features = normalize(features, norm="l2", axis=1)
        self.features_by_key = {key: feature for key, feature in zip(keys, features)}
        # in our full list of features, we add padding dimension for fast dot products
        self.features = np.hstack([features, np.ones((len(keys), 1))])
        logging.debug(
            "Pre: %s, post: %s, %s, %s",
            features.shape,
            self.features.shape,
            features,
            self.features,
        )
        logging.info(
            "Loaded fast classifier from %d feature paths with %d keys in %0.2fs",
            len(feature_paths),
            len(keys),
            time.time() - t0,
        )
        # now save other key variables
        self.keys, self.key_to_item = keys, key_to_item
        self.n_models = n_models
        self.n_top = n_top
        self.n_negatives = n_negatives
        self.rng = default_rng(0)

    def __len__(self):
        """Returns number of items in our dataset"""
        return len(self.keys)

    # @ignore_warnings(category=ConvergenceWarning)
    def train_single_model_impl(self, pos_features, neg_features, neg_weights):
        """Trains a single model and returns a single column array with the coefficients + intercept."""
        # make various lookups
        times = [time.time()]
        train_features = np.vstack((pos_features, neg_features))
        labels = np.array([1] * len(pos_features) + [-1] * len(neg_features))
        weights = np.array(([1] * len(pos_features)) + list(neg_weights))
        times.append(time.time())
        logging.debug(
            "Got training features of shape %s and labels of len %d+%d=%d, %d weights, in %0.3fs: %s",
            train_features.shape,
            len(pos_features),
            len(neg_features),
            len(labels),
            len(weights),
            times[-1] - times[0],
            np.array(neg_weights),
        )
        assert len(labels) == len(weights)
        # create model and train
        model = SGDClassifier(max_iter=20, shuffle=True, class_weight="balanced", tol=1e-3)
        times.append(time.time())
        model.fit(train_features, labels, sample_weight=weights)
        times.append(time.time())
        logging.debug(
            "Created model in %0.3fs and trained in %0.3fs",
            times[-2] - times[-3],
            times[-1] - times[-2],
        )
        # convert to output format
        times.append(time.time())
        ret = np.zeros((train_features.shape[1] + 1, 1), dtype=np.float32)
        ret[:-1, 0] = model.coef_[0]
        ret[-1, 0] = model.intercept_
        return ret

    def train_single_model(self, pos_keys, neg_keys, neg_weights=None):
        # make various lookups
        times = [time.time()]
        f_by_key = self.features_by_key
        # construct training data (skipping bad keys)
        pos_features = [f_by_key[key] for key in pos_keys if key in f_by_key]
        if neg_weights is None:
            neg_weights = [1] * len(neg_keys)
        neg = [
            (f_by_key[key], weight) for key, weight in zip(neg_keys, neg_weights) if key in f_by_key
        ]
        if len(neg) == 0:
            return None
        neg_features, neg_weights = zip(*neg)
        logging.debug("Got pos %s, neg %s, %s", pos_features, neg_features, neg_weights)
        return self.train_single_model_impl(pos_features, neg_features, neg_weights)

    def classify_many(self, models):
        """Classifies all images of using given `models`.

        The models should be an np array of coefficients and intercept per column.
        Returns a Counter mapping from key to score.
        """
        times = [time.time()]
        out = np.dot(self.features, models)
        times.append(time.time())
        # iterate over each column to find top matches and aggregate into scores
        keys = self.keys
        scores = Counter()
        for col in out.T:
            for i in col.argsort()[::-1][: self.n_top]:
                if col[i] > 0:
                    if i < len(
                        keys
                    ):  # sometimes we're in an inconsistent state, so this is a sanity check
                        scores[keys[i]] += col[i]
        times.append(time.time())
        logging.info(
            "Classified models of shape %s and got %d results in %0.3fs",
            models.shape,
            len(scores),
            times[-1] - times[0],
        )
        return scores

    def old_classify_many_rel(self, models):
        """Classifies all pairs of items of using given `models`.

        Since our classifiers are linear, we can do the following:

            outputs = max((features1 - features2) . model)
            outputs = max((features1 . model) - (features2 . model))

        The models should be an np array of coefficients and intercept per column.
        Returns a Counter mapping from pairs of keys to score.
        """
        times = [time.time()]
        out = np.dot(self.features, models)
        times.append(time.time())
        # iterate over each column (model) to find top matches and aggregate into scores
        keys = self.keys
        scores = Counter()
        n_close = int(np.sqrt(self.n_top))
        for col in out.T:
            ordered = sorted([proj, i] for i, proj in enumerate(col))
            for (proj_i, i), (proj_j, j) in zip(ordered[:n_close], ordered[-n_close:]):
                key = (keys[i], keys[j])
                scores[key] += proj_i - proj_j
        times.append(time.time())
        logging.info(
            "Classified models of shape %s and got %d results in %0.3fs",
            models.shape,
            len(scores),
            times[-1] - times[0],
        )
        return scores

    def classify_many_rel(self, models):
        """Classifies all pairs of items of using given `models`.

        In this case, we assume that `models` is just a direction, and we add that direction to
        every item, look for the closest matching items, and order by distance to the matches.

        The models should be an np array of coefficients and intercept per column.
        Returns a Counter mapping from pairs of keys to score.
        """
        keys = self.keys
        scores = Counter()
        dir = models
        target = self.features + dir
        logging.debug('got dir %s: %s, %s -> %s', dir.shape, dir, self.features, target)
        for i, key in tqdm(enumerate(self.keys)):
            row = cdist([target[i]], self.features)
            assert len(row) == 1
            row = row[0]
            idx = np.argmin(row)
            match = self.keys[idx]
            if key == match:
                continue
            min_dist = row[idx]
            scores[(key, match)] = 1.0 if min_dist == 0 else 1.0/min_dist
            logging.debug('For key %s: %s, %s', key, match, min_dist)
        return scores

    def train_and_classify_many_plus_minus(
        self,
        pos_keys: List[str],
        neg_keys: List[str] = [],
    ) -> Tuple[Counter, np.ndarray]:
        """Trains several models and classifies all items in this dataset.

        This does a "plus-minus" style of training: positives and negatives are added with their
        respective labels, additional "background" negatives are added, and then several classifiers
        are trained and evaluated.

        Returns `(scores, models)`. `scores` is a counter from key to score, and `models` is a numpy
        array of the models trained.
        """
        logging.info("Training %d exemplar classifiers", self.n_models)
        times = [time.time()]
        neg_options = set(self.features_by_key) - set(pos_keys) - set(neg_keys)
        times.append(time.time())
        models = []
        # aggregate results from different models
        for i in range(self.n_models):
            # sample some more negatives randomly, but give them lower weight
            more_neg_keys = sample(neg_options, min(len(neg_options), self.n_negatives))
            neg_weights = [1.0] * len(neg_keys) + [0.2] * len(more_neg_keys)
            all_neg = neg_keys + more_neg_keys
            logging.debug("Got %d neg weights, %d neg", len(neg_weights), len(all_neg))
            logging.info("got keys pos %s, neg %s + %s", pos_keys, neg_keys, more_neg_keys)
            cls = self.train_single_model(pos_keys, all_neg, neg_weights=neg_weights)
            models.append(cls)
        times.append(time.time())
        models = np.hstack(models)
        times.append(time.time())
        scores = self.classify_many(models)
        times.append(time.time())
        show_times(times)
        return (scores, models)

    def old_train_and_classify_many_rel(
        self,
        pos_keys: List[str],
        neg_keys: List[str] = [],
    ) -> Tuple[Counter, np.ndarray]:
        """Trains several models and classifies all items in this dataset.

        This does a "relative" style of training: positives and negatives are added with their
        respective labels, additional "background" negatives are added, and then several classifiers
        are trained and evaluated.

        Returns `(scores, models)`. `scores` is a counter from key to score, and `models` is a numpy
        array of the models trained.
        """
        logging.info("Training %d rel exemplar classifiers", self.n_models)
        times = [time.time()]
        models = []
        f_by_key = self.features_by_key
        neg_options = sorted(set(f_by_key) - set(pos_keys) - set(neg_keys))
        # aggregate results from different models
        for i in range(self.n_models):
            pos_features = np.array([f_by_key[neg_keys[0]] - f_by_key[pos_keys[0]]])
            n_neg = min(len(neg_options) // 2, self.n_negatives)
            neg_pairs = self.rng.choice(neg_options, (n_neg, 2), replace=False)
            neg_features = np.array([f_by_key[n1] - f_by_key[n2] for n1, n2 in neg_pairs[:]])
            neg_weights = np.ones(len(neg_features)) / len(neg_features)
            logging.info(
                "Got pos %s, neg pairs %s, features %s: %s, %s",
                pos_features,
                neg_pairs,
                neg_features.shape,
                neg_features,
                neg_weights,
            )
            assert pos_features.shape[1] == neg_features.shape[1]
            assert len(neg_features) == len(neg_weights)
            cls = self.train_single_model_impl(pos_features, neg_features, neg_weights)
            models.append(cls)
        times.append(time.time())
        models = np.hstack(models)
        times.append(time.time())
        scores = self.classify_many_rel(models)
        times.append(time.time())
        show_times(times)
        return (scores, models)

    def train_and_classify_many_rel(
        self,
        pos_keys: List[str],
        neg_keys: List[str] = [],
    ) -> Tuple[Counter, np.ndarray]:
        """Trains several models and classifies all items in this dataset.

        This does a "relative" style of "training", actually it's not even training, it's just
        getting the direction.

        Returns `(scores, models)`. `scores` is a counter from key to score, and `models` is a numpy
        array of the models trained.
        """
        logging.info("Direction based training rel exemplar classifiers")
        f_by_key = self.features_by_key
        dir = f_by_key[neg_keys[0]] - f_by_key[pos_keys[0]]
        dir = np.append(dir, [0.0])
        logging.debug('got items, %s - %s = %s', f_by_key[neg_keys[0]], f_by_key[pos_keys[0]], dir)
        models = np.hstack([dir])
        scores = self.classify_many_rel(models)
        return (scores, models)

    def train_and_classify_many(
        self,
        type: str,
        pos_keys: List[str],
        neg_keys: List[str] = [],
        save_classifier: bool = False,
    ) -> Tuple[Counter, Optional[str]]:
        """Trains several models and classifies all items in this dataset.

        Returns `(scores, cls_id)`
        """
        if type == "plus-minus":
            scores, models = self.train_and_classify_many_plus_minus(pos_keys, neg_keys)
        elif type == "rel":
            scores, models = self.train_and_classify_many_rel(pos_keys, neg_keys)
        if save_classifier:
            subdir = "images"
            cls_id = save_raw_classifiers(models, subdir=subdir)
        else:
            cls_id = None
        return scores, cls_id


class MyStaticHandler(StaticFileHandler):
    """A simple subclass to allow for some additional functionality"""

    def validate_absolute_path(self, root: str, absolute_path: str) -> Optional[str]:
        """This is the same as in the base implementation, but without the check for being in our dir"""
        # The trailing slash also needs to be temporarily added back
        # the requested path so a request to root/ will match.
        if os.path.isdir(absolute_path) and self.default_filename is not None:
            # need to look at the request.path here for when path is empty
            # but there is some prefix to the path that was already
            # trimmed by the routing
            if not self.request.path.endswith("/"):
                self.redirect(self.request.path + "/", permanent=True)
                return None
            absolute_path = os.path.join(absolute_path, self.default_filename)
        if not os.path.exists(absolute_path):
            raise HTTPError(404)
        if not os.path.isfile(absolute_path):
            raise HTTPError(403, "%s is not a file", self.path)
        thumb = self.get_argument("thumb", "")
        if thumb == "1":
            absolute_path += "?thumb"
        return absolute_path

    def _stat(self) -> os.stat_result:
        """We override this to do the thumbnailing as necessary"""
        if self.absolute_path.endswith("?thumb"):
            # convert to thumb path
            dirname, basename = os.path.split(self.absolute_path)
            thumb_path = os.path.join(dirname, "." + basename[:-6])
            orig_path = self.absolute_path[:-6]
            if not exists(thumb_path):
                logging.info("trying to thumbnail %s", orig_path)
                im = Image.open(orig_path)
                im.thumbnail((200, 200))
                im.save(thumb_path)
            self.absolute_path = thumb_path
        return super()._stat()

    @classmethod
    def get_content(
        cls, abspath: str, start: Optional[int] = None, end: Optional[int] = None
    ) -> Generator[bytes, None, None]:
        """We check for thumb in params and return accordingly"""
        logging.info("in get_content, with %s, %s, %s", abspath, start, end)
        return super().get_content(abspath, start, end)


class BaseHandler(RequestHandler):
    """Convenience functions for tornado requests"""

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with, content-type")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def options(self):
        self.set_default_headers()

    def return_jsonp(self, **kw):
        """Returns a json or jsonp response with the given return `kw`.
        It checks for the query parameter "callback" and if present, wraps the result with a function
        call to that name. If not, it returns the json object as-is.
        This function also sets the content-type explicitly to 'application/json'
        """
        ret = json.dumps(kw, sort_keys=True)
        callback = self.get_argument("callback", "")
        content_type = "application/json"
        if callback:
            ret = "{}({});".format(callback, ret)
            content_type = "application/javascript"
        self.set_header("Content-Type", content_type)
        self.write(ret + "\n")

    def get_json_arg(self, name, default="null"):
        """Returns the given arg `name` and decodes it using json"""
        return json.loads(self.get_argument(name, default))


class MyJSONEncoder(json.JSONEncoder):
    """Does on-the-fly translation of numpy types, etc"""

    def default(self, o):
        if "int" in o.__class__.__name__:
            return int(o)
        if "float" in o.__class__.__name__:
            return float(o)
        if "double" in o.__class__.__name__:
            return float(o)
        if "bool" in o.__class__.__name__:
            return bool(o)
        return json.JSONEncoder.default(self, o)


class MainHandler(BaseHandler):
    def get(self):
        cfg = self.application.cfg
        data = dict(cfg=cfg)
        kwargs = dict(
            data=json.dumps(data),
            static_base_name=cfg["static_base_name"],
            css_section="",
            js_section="",
        )
        relopen = lambda filename: open(os.path.join(os.path.dirname(__file__), filename))
        if cfg["use_default_js"]:
            with relopen("fastcls.js") as f:
                kwargs["js_section"] = "<script>%s</script>" % (f.read())
        if cfg["use_default_css"]:
            with relopen("fastcls.css") as f:
                kwargs["css_section"] = "<style>%s</style>" % (f.read())
        with relopen("fastcls_index.html") as f:
            self.write(f.read() % kwargs)


class ItemsHandler(BaseHandler):
    def get(self):
        itemstr_by_key = self.application.itemstr_by_key
        items_str = ",".join(f"{json.dumps(key)}:{value}" for key, value in itemstr_by_key.items())
        self.write(f"{{ {items_str} }}")


class ClassifyHandler(BaseHandler):
    def post(self):
        args = json.loads(self.request.body)
        logging.info("got args %s", args)
        ret = args
        try:
            scores, _ = self.application.fcls.train_and_classify_many(
                args["type"], args["pos"], args["neg"]
            )
            ret.update(status="ok", cls=scores.most_common())
        except Exception as e:
            raise
            ret.update(status=f"error: {type(e)}: {e}")
        return self.return_jsonp(**ret)


class Application(tornado.web.Application):
    """Custom application, so we can define our own settings"""

    def __init__(self, config_path, **kw):
        handlers = [
            (r"/", MainHandler),
            (r"/items", ItemsHandler),
            (r"/classify", ClassifyHandler),
            (r"/static/(.*)", MyStaticHandler, {"path": "static"}),
        ]
        settings = dict(
            xsrf_cookies=False,
            debug=True,
        )
        with open(config_path) as f:
            self.cfg = json.load(f)
        t0 = time.time()
        self.fcls = FastClassifier(**self.cfg["classifier_config"], **kw)
        t1 = time.time()
        # make field funcs
        self.item_fields = self.cfg.get("item_fields", [])
        self.field_funcs = {field: self.make_func(func_str) for field, func_str in self.item_fields}
        t2 = time.time()
        # load items
        self.itemstr_by_key = {}
        items = self.fcls.key_to_item.items()
        # TODO note that right now, the multiprocessing version is way slower
        if 0:
            with mp.Pool(os.cpu_count()) as pool:
                for key, itemstr in pool.map(self.load_item, items, chunksize=10000):
                    self.itemstr_by_key[key] = itemstr
        else:
            self.itemstr_by_key = dict(self.load_item((key, item)) for key, item in items)
        t3 = time.time()
        logging.info("main time diffs %s+%s+%s=%s", t1 - t0, t2 - t1, t3 - t2, t3 - t0)
        tornado.web.Application.__init__(self, handlers, **settings)

    @staticmethod
    def make_func(s):
        def ret(key, paths, **kwargs):
            try:
                return eval(s)
            except Exception as e:
                logging.error(
                    f"Got error of type {type(e)} with key {key}, paths {paths}, kw {kwargs}: {e}"
                )

        return ret

    def load_item(self, item_with_key):
        key, item = item_with_key
        # add our custom fields
        for field, _ in self.item_fields:
            item[field] = self.field_funcs[field](key=key, **item)
        # cast items to the right type
        for field, value in item.items():
            name = value.__class__.__name__
            if "int" in name:
                item[field] = int(value)
            if "float" in name or "double" in name:
                item[field] = float(value)
            if "bool" in name:
                item[field] = bool(value)
        item_str = json.dumps(item, indent=2)
        return (key, item_str)


if __name__ == "__main__":
    LOG_FORMAT = "%(asctime)s.%(msecs)03d\t%(filename)s:%(lineno)s\t%(funcName)s\t%(message)s"
    logging.basicConfig(format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    parser = ArgumentParser(description="Server for fast interactive classification")
    parser.add_argument("config_path", help="Path to config file (json)")
    parser.add_argument(
        "-p", "--port", type=int, default=8000, help="What port to run the server on [8000]"
    )
    parser.add_argument(
        "-f",
        "--max_features",
        type=int,
        default=-1,
        help="If >0, limit to that many input features",
    )
    args = parser.parse_args()
    kw = vars(args)
    port = kw.pop("port")
    Application(**kw).listen(port)
    logging.info("Ready to start serving fast classification on port %d", port)
    tornado.ioloop.IOLoop.instance().start()
