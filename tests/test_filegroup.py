"""Tests out nkpylib.filegroup"""

from __future__ import annotations

import pytest
import os
from filegroup import FileGroup, FileOp

def test_filegroup_initialization():
    """Test initialization of FileGroup with valid and invalid paths."""
    # Assuming test files are in a directory named 'test_files'
    valid_path = 'test_files/original_file'
    invalid_path = 'test_files/non_existent_file'

    # Test valid initialization
    fg = FileGroup(valid_path, assert_exist=False)
    assert fg.paths['orig'] == valid_path

    # Test invalid initialization
    with pytest.raises(FileNotFoundError):
        FileGroup(invalid_path)

def test_filegroup_translate_path():
    """Test the translate_path method."""
    path = 'test_files/original_file'
    fg = FileGroup(path, assert_exist=False)
    json_path = fg.translate_path(path, 'json')
    assert json_path == 'test_files/.original_file.json'

def test_filegroup_iter():
    """Test the iteration over file paths in the group."""
    path = 'test_files/original_file'
    fg = FileGroup(path, assert_exist=False)
    paths = list(fg)
    assert paths == ['test_files/original_file', 'test_files/.original_file.json']

def test_fileop_execute():
    """Test the execute method of FileOp."""
    src = 'test_files/source_file'
    dst = 'test_files/destination_file'
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
