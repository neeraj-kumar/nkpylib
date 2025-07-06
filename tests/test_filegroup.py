"""Tests out nkpylib.filegroup"""

from __future__ import annotations

import pytest
import os

from os.path import basename, dirname, join, abspath, isdir, split, exists

from nkpylib.filegroup import FileGroup, FileOp

DATA_DIR = join(dirname(__file__), 'data')

@pytest.fixture
def path1():
    """Fixture for the path to the test document."""
    return join(dirname(__file__), 'data', 'slow magic tickets.pdf')

def test_filegroup_initialization(path1):
    """Test initialization of FileGroup with valid and invalid paths."""
    invalid_path = join(DATA_DIR, 'non_existent_file')

    # Test valid initialization
    fg = FileGroup(path1, assert_exist=False)
    assert fg.paths['orig'] == path1

    # Test invalid initialization
    with pytest.raises(FileNotFoundError):
        FileGroup(invalid_path)


def test_filegroup_translate_path(path1):
    """Test the translate_path method."""
    fg = FileGroup(path1, assert_exist=False)
    json_path = fg.translate_path(path1, 'json')
    assert json_path == join(DATA_DIR, '.slow magic tickets.pdf.json')

def test_filegroup_iter(path1):
    """Test the iteration over file paths in the group."""
    fg = FileGroup(path1, assert_exist=False)
    paths = list(fg)
    assert paths == [path1, join(DATA_DIR, '.slow magic tickets.pdf.json')]

def test_filegroup_exists(path1):
    """Test the exists method of FileGroup."""
    fg = FileGroup(path1, assert_exist=False)
    assert fg.exists('orig')
    assert fg.exists('json')

def test_filegroup_iteritems(path1):
    """Test the iteritems method of FileGroup."""
    path = path1
    fg = FileGroup(path, assert_exist=False)
    items = list(fg.iteritems())
    assert items == [('orig', path1), ('json', join(DATA_DIR, '.slow magic tickets.pdf.json'))]

def notest_filegroup_apply_fileop(path1):
    """Test the apply_fileop method of FileGroup."""
    path = path1
    fg = FileGroup(path, assert_exist=False)
    new_name = 'test_renamed'
    file_op = FileOp(rename=True)

    # Create a dummy original file
    with open(fg.orig_path, 'w') as f:
        f.write('dummy content')

    # Apply file operation
    fg.apply_fileop(new_name, file_op)

    # Check if the new file exists
    new_path = join(DATA_DIR, f'{new_name}.pdf')
    assert os.path.exists(new_path)

    # Clean up
    os.remove(new_path)
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
