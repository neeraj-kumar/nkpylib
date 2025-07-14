"""Full search parser"""

from __future__ import annotations

import json
import logging
import sys
import time

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pprint import pprint
from typing import Sequence, Any, Callable, Generator

from lark import Lark, Transformer

import numpy as np

logger = logging.getLogger(__name__)


