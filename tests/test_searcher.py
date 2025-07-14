"""Tests out nkpylib.filegroup"""

from __future__ import annotations

import pytest
import os

from os.path import basename, dirname, join, abspath, isdir, split, exists

from nkpylib.search.searcher import SearchCond, Op, OpCond, JoinType, JoinCond, SearchResult, SearchImpl, Searcher
