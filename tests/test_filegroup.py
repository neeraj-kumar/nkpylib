"""Tests out nkpylib.filegroup"""

from __future__ import annotations

import pytest
import os

from os.path import basename, dirname, join, abspath, isdir, split, exists

from nkpylib.filegroup import FileGroup, FileOp

DATA_DIR = join(dirname(__file__), 'data')
DOC1_PATH = join(DATA_DIR, 'slow magic tickets.pdf')

def test_filegroup_initialization():
    """Test initialization of FileGroup with valid and invalid paths."""
    valid_path = DOC1_PATH
    invalid_path = join(DATA_DIR, 'non_existent_file')

    # Test valid initialization
    fg = FileGroup(valid_path, assert_exist=False)
    assert fg.paths['orig'] == valid_path

    # Test invalid initialization
    with pytest.raises(FileNotFoundError):
        FileGroup(invalid_path)


def test_filegroup_translate_path():
    """Test the translate_path method."""
    path = DOC1_PATH
    fg = FileGroup(path, assert_exist=False)
    json_path = fg.translate_path(path, 'json')
    assert json_path == join(DATA_DIR, '.slow magic tickets.pdf.json')

def test_filegroup_iter():
    """Test the iteration over file paths in the group."""
    path = DOC1_PATH
    fg = FileGroup(path, assert_exist=False)
    paths = list(fg)
    assert paths == [DOC1_PATH, join(DATA_DIR, '.slow magic tickets.pdf.json')]

def test_fileop_execute():
    """Test the execute method of FileOp."""
    src = join(DATA_DIR, 'test_source.pdf')
    dst = join(DATA_DIR, 'test_copy.pdf')
    file_op = FileOp(copy=True)

    # Create a dummy source file
    with open(src, 'w') as f:
        f.write('dummy content')

    # Execute file operation
    file_op.execute(src, dst)

    # Check if the destination file exists
    assert os.path.exists(dst)

    # Clean up
    os.remove(src)
    os.remove(dst)
