import os
import tempfile
from pathlib import Path

import pytest

from nkpylib.cacher.file_utils import _write_atomic, _read_file

def test_write_atomic_creates_file():
    """Test that _write_atomic creates a file with the correct contents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.txt"
        data = b"test data"
        
        _write_atomic(path, data)
        
        # Check file exists and has correct contents
        assert path.exists()
        assert path.read_bytes() == data
        
        # Check no temporary files were left behind
        temp_files = [f for f in os.listdir(tmpdir) if f.startswith("test.txt.")]
        assert not temp_files

def test_write_atomic_overwrites_existing():
    """Test that _write_atomic safely overwrites an existing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.txt"
        
        # Create initial file
        path.write_bytes(b"initial data")
        
        # Overwrite with new data
        new_data = b"new data"
        _write_atomic(path, new_data)
        
        # Check file has new contents
        assert path.read_bytes() == new_data

def test_read_file_success():
    """Test that _read_file successfully reads existing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.txt"
        data = b"test data"
        path.write_bytes(data)
        
        assert _read_file(path) == data

def test_read_file_missing():
    """Test that _read_file returns None for missing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nonexistent.txt"
        
        assert _read_file(path) is None

def test_read_file_permission_error():
    """Test that _read_file returns None for unreadable file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "unreadable.txt"
        path.write_bytes(b"secret data")
        path.chmod(0)  # Remove all permissions
        
        assert _read_file(path) is None
