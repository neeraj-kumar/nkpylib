"""Generic file watcher utility.

Okay. So I want to think about a generic moving utility or like file tracker. That's basically the main impetus is of course, the biker gang stuff. And so here's the thing we have a list of files, they have file names, their file paths, their sizes of hashes. And they can move. They can, we can add new ones they can be deleted. And then these things, have some correspondence.

What is the

relationship of

this secondary representation like air table? I think even my choice that the word suggests that I'm thinking of it as a dependent relationship and so the filesystem is a source of truth.

And I think the other key thing is we want to do this, whatever we do on a state comparison basis as opposed to an ongoing diff this

basis meaning not something like I modified

so

So here what we'd have is

two states before an A and B.

And so I guess

what I want to be able to do this I want to

take those dates and generate a diff out of them.

And I guess the diff I want to generate is

deletes.

Actually, let me spell it out. So a delete is a file that previously existed in a but has no
counterpart in B which are files that exist in B but have no counterpart in a moves which includes
renames which are files that have exactly our part in birth but the name or path is different and
then copies were.

Certain copies we there's a file and that now has two counterparts so I think that's the space of
possibilities. Some simple edge cases so think we'll treat siblings special files. For the moment.
We won't consider folders to be files. So we're only talking about files that actually files rather,
we're talking about non folders

Okay, so now I should say what counterpart means. So counterpart means that the file hashes the same
right so I think that's the ground truth that we consider which source of truth that we consider
and. Okay and then I think, the output, so the input is to, like, two snapshots at different states
and the output is a diff, and a diff is an ordered list of changes, that if you apply them in that
order to state A, you get to state B. Now, obviously there are infinite possible deaths that can
represent two states. And so we just want kind of something simple and we could take cues from
actual file creation times or whatever you want to other we are, by the way, by necessarily missing
information. For example, when you delete a file, in state B, you have no idea whether whether the
file actually existed or not. And, or rather, when it was really good.

Okay, and so I think with this framing, it's pretty straightforward. We just have to implement
something that satisfies this satisfies this contract and so I think think we can assume yes, I
think we also have to assume something else, that we have various metadata at stake. Well, in
general, I think we have to assume that we have access to relevant metadata, both that state A and
state B. And this metadata includes both ironically Pat since that's almost part of data, but it
does include the file size, content hash, maybe modification time but maybe it's better not to
assume we have that.

So given that I think the simplest thing is to kind of the naive, exhaustive algorithm and then make
it more efficient. So I think, nicely. Another edge case to handle is what if we have multiple files
with the same hash in a given state? Okay, we'll put that put that to the side for now. So I think
the exhaust process is pretty simple. We just we just match hashes right. So again, assuming that
there's no duplicates in the initial state. For every hash, we look for its hash, and B. And if the
match is unique, then we look at the path and if the path is different, than that's a move or
rename. And if the. If the past and if it doesn't exist. Then we assume that it was deleted. And if
it was and then if it's gone from one to multiple then it was copied. And I guess if none of the
copies are where the original was, then that's actually removed or renamed first, and then it's
copied to the other ones.

And so here in the future, we can have some optimizations around preferring the fewest number of
changes to be the one that was moved. And then the other ones to be copies. But again, it's a minor
point. Yeah, so I think that's the easiest. Answer then there's a question of optimizations. So.

Think we can first assume copies sorry that if the sizes are different than it's definitely the
sizes are the same. It's probably the same but not for sure. We can check with the hash. And then we
can make a much more tentative assumption which is probably true in practice, at least for the
cooking example that. If the path is the same, the file is the same. So I think then the process
becomes simpler I guess. Which is we first just get five listings for both A and B. Anything that's
the same we assume is the same and we remove it. For everything left the first computers I said is
if those match then we're done. To compute the hashes I guess if we're doing that, we might as well
just compute. Just compute hashes to begin with
"""

from __future__ import annotations

import functools
import json
import os

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from hashlib import sha256, md5
from typing import Optional, Iterable

class DiffType(str, Enum):
    ADD = 'ADD'
    DELETE = 'DELETE'
    MOVE = 'MOVE'
    COPY = 'COPY'


@dataclass
class Diff:
    """Represents a single diff between two tree listings.

    Has a `type` and two keys, `a` and `b`. If `a` is None, then this is an ADD. If `b` is None,
    then this is a DELETE. For MOVE or COPY and `a` is the original key and `b` is the new key.
    """
    type: DiffType
    a: Optional[str] # None for ADD
    b: Optional[str] # None for DELETE

    @classmethod
    def group_diffs(cls, diffs: Iterable[Diff]) -> list[list[Diff]]:
        """Groups sequential diffs of the same type together.

        Returns a list of lists of diffs.
        """
        ret = []
        cur = []
        for d in diffs:
            if not cur:
                cur.append(d)
            else:
                if d.type == cur[-1].type:
                    cur.append(d)
                else:
                    ret.append(cur)
                    cur = []
                    cur.append(d)
        if cur:
            ret.append(cur)
        return ret

@functools.cache
def hash_path(path: str, hash_constructor=sha256) -> str:
    """Returns the sha256 hash of a file."""
    with open(path, 'rb') as f:
        # read it in chunks to avoid memory issues
        h = hash_constructor()
        while chunk := f.read(4096):
            h.update(chunk)
    return h.hexdigest()


class Tree(ABC):
    """Represents a file listing as a tree and allows for some common operations on them.

    At base, this contains:
    - a list of `keys` (which are just strings).
    - a dict of `hash_by_key` (computed lazily as needed).

    """
    def __init__(self, keys: list[str]):
        self.keys = keys[:]
        self.hash_by_key = {}

    def __len__(self):
        return len(self.keys)

    @abstractmethod
    def hash_function(self, keys: Iterable[str]) -> Iterable[str]:
        """Hashes the given `keys`."""
        pass

    def hash(self, keys: Iterable[str]) -> Iterable[str]:
        """Returns the hashes of the given `keys`, computing them if necessary.

        This just calls `hash_function` and caches the result.
        """
        todo = [key for key in keys if key not in self.hash_by_key]
        if todo:
            hashes = self.hash_function(todo)
            self.hash_by_key.update(zip(todo, hashes))
        return [self.hash_by_key[key] for key in keys]


    def compare(self, other: Tree, assume_keys_constant=False) -> list[Diff]:
        """Compares this tree (before) with `other` (after) and returns an ordered list of diffs.

        If `assume_keys_constant` is True, then files with the same keys are assumed to be the same
        (without checking hashes).
        """
        diffs = []
        self_keys = set(self.keys)
        other_keys = set(other.keys)
        if assume_keys_constant:
            # first filter out the keys that are the same
            common = self_keys & other_keys
            self_keys -= common
            other_keys -= common
        # hash remaining keys
        self_keys = sorted(self_keys)
        other_keys = sorted(other_keys)
        self_hashes = self.hash(self_keys)
        # we can't handle duplicates in the original for now
        assert len(set(self_hashes)) == len(self_hashes), "Duplicate hashes in original"
        other_hashes = other.hash(other_keys)
        before_by_hash = {h: k for h, k in zip(self_hashes, self_keys)}
        after_by_hash = {h: defaultdict(list) for h in other_hashes}
        for key, hash in zip(other_keys, other_hashes):
            after_by_hash[hash][key].append(key)
        # now we can do the comparison
        for hash, keys in after_by_hash.items():
            if hash in before_by_hash:
                # moves and copies
                before_key = before_by_hash[hash]
                # if we have the same key, remove it
                if before_key in keys:
                    keys.remove(before_key)
                # now if we have keys left, we have a move or copy
                for idx, after_key in enumerate(keys):
                    # mark the first one as a move, the rest as copies
                    diff_type = DiffType.MOVE if idx == 0 else DiffType.COPY
                    diffs.append(Diff(diff_type, before_key, after_key))
            else:
                # new file(s)
                for key in keys:
                    diffs.append(Diff(DiffType.ADD, None, key))
        # now we need to find the deletes
        for hash, key in before_by_hash.items():
            if hash not in after_by_hash:
                diffs.append(Diff(DiffType.DELETE, key, None))
        return diffs


    def execute_diffs(self, diffs, other):
        """Executes a list of a diffs.

        Assuming you have `diffs` from comparing `self` (before) to `other` (after), this method
        "executes" the diffs on `self`. This first groups sequential list of diffs of the same type
        together using `group_diffs()`. It then calls the various `execute_XXX` functions (which
        must be subclassed).
        """
        grouped = Diff.group_diffs(diffs)
        function_by_type = {
                DiffType.ADD: self.execute_add,
                DiffType.DELETE: self.execute_delete,
                DiffType.MOVE: self.execute_move,
                DiffType.COPY: self.execute_copy,
        }
        for group in grouped:
            func = function_by_type[group[0].type]
            func(group, other)

    def execute_add(self, diffs: list[Diff], other: Tree) -> None:
        """Executes ADD operations from given `diffs` going from `self` to `other`"""
        raise NotImplementedError("execute ADD not implemented")

    def execute_delete(self, diffs: list[Diff], other: Tree) -> None:
        """Executes DELETE operations from given `diffs` going from `self` to `other`"""
        raise NotImplementedError("execute DELETE not implemented")

    def execute_move(self, diffs: list[Diff], other: Tree) -> None:
        """Executes MOVE operations from given `diffs` going from `self` to `other`"""
        raise NotImplementedError("execute MOVE not implemented")

    def execute_copy(self, diffs: list[Diff], other: Tree) -> None:
        """Executes COPY operations from given `diffs` going from `self` to `other`"""
        raise NotImplementedError("execute COPY not implemented")


class FileTree(Tree):
    """A tree that represents a file listing.

    This sets a `root` directory that the keys are relative to.
    """
    def __init__(self,
                 root: str,
                 keys: Optional[list[str]],
                 hash_constructor: callable = sha256,
                 read_dir: bool = False,
                 filter_func: Optional[callable] = None):
        """Initializes this file tree at given `root`.

        If you give a list of keys, then initializes with those keys.
        If you set `read_dir` to True, then gets the file listing from the directory, recursively.
        If you set `filter_func`, then only includes keys for which `filter_func(key)` is True.
        """
        if keys is None:
            keys = []
        super().__init__(keys)
        self.root = root
        self.filter_func = filter_func
        self.hash_constructor = hash_constructor
        if read_dir:
            self.keys = self.get_file_listing(root)

    def hash_function(self, keys: Iterable[str]) -> Iterable[str]:
        """Hashes given paths using our hash function"""
        return [hash_path(os.path.join(self.root, key), hash_constructor=self.hash_constructor) for key in keys]

    def get_file_listing(self, dir: str) -> list[str]:
        """Returns a list of all files (no dirs) in a directory, recursively."""
        ret = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                path = os.path.join(root, file)
                if self.filter_func is None or self.filter_func(path):
                    ret.append(os.path.relpath(path, self.root))
        ret.sort()
        return ret



if __name__ == '__main__':
    parser = ArgumentParser(description="Compare two states of a directory and generate a diff.")
    parser.add_argument("dir_a", help="'Before' directory")
    parser.add_argument("dir_b", help="'After' directory")
    args = parser.parse_args()
    files1 = get_file_listing(args.dir_a)
    files2 = get_file_listing(args.dir_b)
    print(json.dumps(files1, indent=2))

