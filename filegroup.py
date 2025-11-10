"""A FileGroup is a group of files that are strongly related to each other.

This module provides a class `FileGroup` that you can use to manage such groups of files. It assumes
a few things:
- The files all live in the same directory
- The filenames are of the form {prefix}{base}{suffix}, where base is identical for files in a
  group, the prefix (commonly . to make them hidden) and suffix (like extensions) are both optional.
- There's a "original" file type that you want to use as the main referent for the group. Frequently
  this is literally the original file, and the others are derivatives of it.
- The base class provided here assumes that one of the others is a "json" file. But you can override
  that in your subclass if needed.

Given a `FileGroup`, you can do things like:
- Given an "original" file (or any of the other types), find the other files in that group.
- Performing file operations on the group, like renaming or moving files together.
- "Translate" filenames between the different file types in the group.
- Construct a file group from a list of file paths, optionally reading from a file list or stdin.

A `FileGroup` subclass should override the class variable TYPES with a dict mapping from file types
(strings) to a (prefix, suffix) tuple for the type of group it supports. (By default, this contains
'orig' and 'data'.) There are some primitive class methods:
- `split_path(path)` -> (dir, prefix, base, suffix)
- `identify_type(path)` -> type (as str)
- `is_type(path, type)` -> bool
- `translate_path(path, type)` -> converts a path to the given type

These primitives are then used to add some convenience properties:
- to_{type_name}_path: converts this instance's path to the specified type
"""

#TODO allow the orig prefix/suffix to be regexps

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys

from abc import ABC, abstractmethod
from os.path import basename, dirname, join, abspath, isdir, split, exists
from typing import Any, Generator, Sequence, TypeVar, Iterator

logger = logging.getLogger(__name__)

# type alias for JSONable objects
JSONable = Any


class FileOp:
    """Container class for file operation flags."""

    def __init__(
        self,
        dry_run: bool = False,
        copy: bool = False,
        symlink: bool = False,
        hardlink: bool = False,
        rename: bool = False,
        suffix: bool = False,
    ):
        self.dry_run = dry_run
        self.copy = copy
        self.symlink = symlink
        self.hardlink = hardlink
        self.rename = rename
        self.suffix = suffix
        # assert only one of copy, symlink, hardlink, or rename is set, suffix can be combined
        values = [copy, symlink, hardlink, rename]
        assert (
            sum(values) == 1
        ), "Exactly one of copy, symlink, hardlink, or rename must be set, currently {values}"

    def __repr__(self) -> str:
        """Return a string representation of the file operation."""
        parts = []
        if self.copy:
            parts.append("copy")
        if self.symlink:
            parts.append("symlink")
        if self.hardlink:
            parts.append("hardlink")
        if self.rename:
            parts.append("rename")
        if self.suffix:
            parts.append("suffix")
        if self.dry_run:
            parts.append("dry_run")
        return f"FileOp<{', '.join(parts)}>"

    @property
    def name(self) -> str:
        """Return the name of the chosen file operation."""
        if self.copy:
            return "copy"
        elif self.symlink:
            return "symlink"
        elif self.hardlink:
            return "hardlink"
        elif self.rename:
            return "rename"
        raise ValueError("No file operation set")

    def execute(self, src: str, dst: str, quiet: bool=False):
        """Perform the file operation based on the flags."""
        if src == dst:
            return
        msg = f"{self.name}: {src} -> {dst}"
        if self.dry_run:
            msg += " [dry run]"
        if quiet:
            logger.debug(msg)
        else:
            logger.info(msg)
        if not self.dry_run:
            try:
                os.makedirs(dirname(dst), exist_ok=True)
            except Exception:
                pass
            if self.copy:
                shutil.copy2(src, dst)
            elif self.symlink:
                os.symlink(abspath(src), abspath(dst))
            elif self.hardlink:
                os.link(abspath(src), abspath(dst))
            elif self.rename:
                os.rename(src, dst)

    def add_num_suffix(self, path: str, existing_paths: set, delim: str = " ") -> str:
        """Add a numerical suffix to the path if it collides with existing paths."""
        base, ext = os.path.splitext(path)
        counter = 1
        new_path = path
        while new_path in existing_paths:
            new_path = f"{base}{delim}{counter}{ext}"
            counter += 1
        return new_path


class FileGroupMeta(type):
    """Metaclass needed to generate properties automatically"""
    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)

        types = namespace.get('TYPES', {})
        for base in bases:
            if hasattr(base, 'TYPES'):
                types = {**base.TYPES} # overwrite with the current class's TYPES

        for type_key in types:
            prop_name = f"{type_key}_path"

            # Avoid overwriting existing properties
            if not hasattr(cls, prop_name):
                def make_property(key):
                    @property
                    def _prop(self):
                        return self.paths[key]
                    return _prop

                setattr(cls, prop_name, make_property(type_key))

        return cls

FG = TypeVar('FG', bound='FileGroup')

class FileGroup(metaclass=FileGroupMeta):
    """A base class for file groups."""
    TYPES = {'orig': ('', ''),
             'json': ('.', '.json')}
    def __init__(self, path: str, assert_exist: bool = True, assert_data: bool = True):
        """Initialize the FileGroup with the given path.

        The path can be any of our TYPES.
        If `assert_exist` is True, it will raise an error if any of our TYPES do not exist on disk.
        If `assert_data` is True, it will raise an error if the JSON data cannot be loaded.
        """
        logger.debug(f'Initializing FileGroup with path: {path}')
        self.paths = {}
        t = self.identify_type(path)
        self.paths[t] = path
        logger.debug(f'Identified type: {t} for path: {self.split_path(path)}')
        for type_name, (prefix, suffix) in self.TYPES.items():
            if type_name != t:
                self.paths[type_name] = self.translate_path(path, type_name)
                if assert_exist and not exists(self.paths[type_name]):
                    raise FileNotFoundError(f"Path {self.paths[type_name]} does not exist for type {type_name}")
        logger.debug(f'All paths: {self.paths}')
        self._data: Any = None
        try:
            self._load_data()
        except Exception as e:
            if assert_data:
                raise

    @classmethod
    def ordered_types(cls) -> list[str]:
        """Return the type names ordered by most constrained first."""
        ret = []
        # first add those with both prefix and suffix
        for type_name, (prefix, suffix) in cls.TYPES.items():
            if prefix and suffix:
                ret.append(type_name)
        # now add those with just prefix
        for type_name, (prefix, suffix) in cls.TYPES.items():
            if prefix and not suffix:
                ret.append(type_name)
        # now add those with just suffix
        for type_name, (prefix, suffix) in cls.TYPES.items():
            if not prefix and suffix:
                ret.append(type_name)
        # now add those with neither
        for type_name, (prefix, suffix) in cls.TYPES.items():
            if not prefix and not suffix:
                ret.append(type_name)
        return ret

    @classmethod
    def split_path(cls, path: str) -> tuple[str, str, str, str]:
        """Splits a file path into its dir, prefix, base, and suffix components.

        If it's a directory, then it will return `(path (with trailing /), '', '', '')`

        This checks our TYPES dict to look for prefixes and suffixes on the filename (without dirs).
        If it doesn't seem to match any of the types, it will return the directory and assume the
        entire filename portion is the base.

        Override this function if that approach won't work for your subclass.
        """
        if isdir(path):
            if not path.endswith('/'):
                path += '/'
            return path, '', '', ''
        dir, rest = split(path)
        for type_name in cls.ordered_types():
            prefix, suffix = cls.TYPES[type_name]
            if rest.startswith(prefix) and rest.endswith(suffix):
                end_idx = -len(suffix) if suffix else None
                base = rest[len(prefix):end_idx]
                return dir, prefix, base, suffix
        else:
            return (dir, '', rest, '')

    @classmethod
    def identify_type(cls, path: str) -> str:
        """Identify the type of file based on its extension.

        Can raise ValueError if there are issues, such as empty base or unknown type.
        """
        parse = dir, prefix, base, suffix = cls.split_path(path)
        if not base:
            raise ValueError(f"Path must have a base. Current parse: {parse}")
        for type_name in cls.ordered_types():
            p, s = cls.TYPES[type_name]
            if prefix == p and suffix == s:
                return type_name
        raise ValueError(f"Path {path} does not match any known file type: {cls.TYPES.keys()}")

    @classmethod
    def is_type(cls, path: str, type_name: str) -> bool:
        """Check if the file path matches the given type."""
        assert type_name in cls.TYPES, f"Unknown type: {type_name}. Known types: {cls.TYPES.keys()}"
        return cls.identify_type(path) == type_name

    @classmethod
    def translate_path(cls, path: str, type_name: str) -> str:
        """Convert a file path to the specified type (could be the same as given)."""
        assert type_name in cls.TYPES, f"Unknown type: {type_name}. Known types: {cls.TYPES.keys()}"
        cur_type = cls.identify_type(path)
        if cur_type == type_name:
            return path
        dir, prefix, base, suffix = cls.split_path(path)
        new_prefix, new_suffix = cls.TYPES[type_name]
        return join(dir, f"{new_prefix}{base}{new_suffix}")

    def __iter__(self) -> Iterator[str]:
        """Iterate over the file paths in the group, in order of our TYPES."""
        for type_name in self.TYPES:
            yield self.paths[type_name]

    @property
    def first(self) -> str:
        """Returns the first file path in this group."""
        for s in self:
            break
        return s

    def iteritems(self) -> Iterator[tuple[str, str]]:
        """Iterate over the file paths in the group, yielding (type_name, path) tuples."""
        for type_name in self.TYPES:
            yield type_name, self.paths[type_name]

    def exists(self, t: str) -> bool:
        """Check if the file of the given type exists on disk."""
        return exists(self.paths[t])

    def __repr__(self) -> str:
        """Return the path without the prefix or suffix"""
        dir, prefix, base, suffix = self.split_path(self.first)
        logger.debug(f'FileGroup __repr__ called with dir={dir}, prefix={prefix}, base={base}, suffix={suffix}')
        return f'<{dir}/{base}>'

    def _load_data(self):
        """Load data data from our JSON file."""
        self._data = None
        logger.debug(f'for {self.orig_path} -> opening {self.json_path}')
        with open(self.json_path) as f:
            self._data = json.load(f)

    @property
    def data(self) -> JSONable:
        """Return our data."""
        return self._data

    @property
    def json_str(self) -> str:
        """Return our data as a string (keys sorted for consistency)."""
        return json.dumps(self._data, sort_keys=True)

    @property
    def pretty(self) -> str:
        """Return the JSON data as a string in pretty-printed format."""
        return json.dumps(self._data, indent=2)

    @classmethod
    def from_paths(cls: type[FG],
                   paths: list[str],
                   file_list_path: str='',
                   raise_errors: bool=True,
                   **kwargs) -> list[FG]:
        """Create a list of `FileGroup` objects (or subclasses) from a list of file paths.

        If the first one is -, then it reads paths from stdin, skipping empty lines. It will
        continue to read any other paths in the list.

        You can optionally provide a `file_list_path` to read from a file in addition. In that case, we
        skip lines that start with '#' or are empty.

        If `raise_errors` is True, then it will raise an error if any of the paths cannot be loaded.

        All other kwargs are passed to the `cls` constructor.
        """
        all_paths = []
        if paths and paths[0] == "-":
            all_paths.extend(list(sys.stdin))
            paths = paths[1:]
        all_paths.extend(paths)
        if file_list_path:
            with open(file_list_path, 'r') as f:
                all_paths.extend([line for line in f if not line.startswith('#')])
        ret = []
        seen_first_paths = set()
        for p in all_paths:
            p = p.strip()
            if not p:
                continue
            try:
                fg = cls(p, **kwargs)
                if fg.first in seen_first_paths:
                    continue
                seen_first_paths.add(fg.first)
                ret.append(fg)
            except Exception as e:
                logger.error(f"Error loading {p}: {e}")
                if raise_errors:
                    raise
        return ret
