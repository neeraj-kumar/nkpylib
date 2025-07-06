"""Tests out nkpylib.filegroup"""

from __future__ import annotations

import pytest
import os

from os.path import basename, dirname, join, abspath, isdir, split, exists

from nkpylib.filegroup import FileGroup, FileOp

DATA_DIR = join(dirname(__file__), 'data')

@pytest.fixture
def path1():
    """Fixture for the path to a full test document."""
    return join(dirname(__file__), 'data', 'slow magic tickets.pdf')

@pytest.fixture
def path2():
    """Fixture for the path to a small test document."""
    return join(dirname(__file__), 'data', 'driver.pdf')

@pytest.fixture(params=["slow magic tickets.pdf", "driver.pdf"])
def path(request):
    """Parameterized fixture for paths to test documents."""
    return join(dirname(__file__), 'data', request.param)

def test_filegroup_initialization(path):
    """Test initialization of FileGroup with valid and invalid paths."""

    # Test valid initialization
    fg = FileGroup(path, assert_exist=False)
    assert fg.paths['orig'] == path

    # test initialization with json version of path
    json_path = FileGroup.translate_path(path, 'json')
    fg = FileGroup(json_path, assert_exist=False)
    assert fg.paths['json'] == json_path

    # Test invalid initialization
    with pytest.raises(FileNotFoundError):
        FileGroup(join(DATA_DIR, 'non_existent_file'))

    # test not checking for either existence or data
    FileGroup(__file__, assert_exist=False, assert_data=False)

    with pytest.raises(FileNotFoundError):
        FileGroup(__file__, assert_exist=False)
        FileGroup(__file__, assert_data=False)


def test_filegroup_translate_path(path1):
    """Test the translate_path method."""
    fg = FileGroup(path1, assert_exist=False)
    json_path = fg.translate_path(path1, 'json')
    assert json_path == join(DATA_DIR, '.slow magic tickets.pdf.json')
    orig_path = fg.translate_path(path1, 'orig')
    assert orig_path == path1
    orig_path = fg.translate_path(json_path, 'orig')
    assert orig_path == join(path1)

def test_filegroup_iter(path1):
    """Test the iteration over file paths in the group."""
    fg = FileGroup(path1, assert_exist=False)
    paths = list(fg)
    assert paths == [path1, join(DATA_DIR, '.slow magic tickets.pdf.json')]

def test_filegroup_iteritems(path1):
    """Test the iteritems method of FileGroup."""
    path = path1
    fg = FileGroup(path, assert_exist=False)
    items = list(fg.iteritems())
    assert items == [('orig', path1), ('json', join(DATA_DIR, '.slow magic tickets.pdf.json'))]

def test_filegroup_first(path):
    """Test the first method of FileGroup."""
    fg = FileGroup(path)
    assert fg.first == path

def test_filegroup_exists(path):
    """Test the exists method of FileGroup."""
    fg = FileGroup(path, assert_exist=False)
    assert fg.exists('orig')
    assert fg.exists('json')
    with pytest.raises(KeyError):
        fg.exists('non_existent_type')

def test_filegroup_data(path2):
    """Test the data property of FileGroup."""
    fg = FileGroup(path2)
    assert fg.data == {'baz': [1, 3, 5], 'foo': 'bar'}

def test_filegroup_json_str(path2):
    """Test the json_str property of FileGroup."""
    fg = FileGroup(path2)
    assert fg.json_str == '{"baz": [1, 3, 5], "foo": "bar"}'

def test_filegroup_pretty(path2):
    """Test the pretty property of FileGroup."""
    fg = FileGroup(path2)
    assert fg.pretty == '''{
  "foo": "bar",
  "baz": [
    1,
    3,
    5
  ]
}'''

def test_filegroup_repr(path):
    """Test the __repr__ method of FileGroup."""
    fg = FileGroup(path, assert_exist=False)
    expected_repr = f"<{dirname(path)}{basename(path)}>"
    assert repr(fg) == expected_repr

def test_filegroup_is_type(path):
    """Test the is_type class method of FileGroup."""
    assert FileGroup.is_type(path, 'orig')
    json_path = FileGroup.translate_path(path, 'json')
    assert FileGroup.is_type(json_path, 'json')

def test_filegroup_from_paths(path):
    """Test the from_paths class method of FileGroup."""
    file_groups = FileGroup.from_paths([path])
    assert len(file_groups) == 1
    fg = file_groups[0]
    print(fg, fg.paths)
    assert fg.paths['orig'] == path

def notest_filegroup_apply_fileop():
    """Test the apply_fileop method of FileGroup."""
    path = join(DATA_DIR, 'test_file.pdf')
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
