import hashlib
import json
import os
import tempfile
from pathlib import Path

import pytest

from nkpylib.cacher.file_utils import _write_atomic, _read_file
from nkpylib.cacher.keyers import (
    TupleKeyer, StringKeyer, HashStringKeyer, HashBytesKeyer
)
from nkpylib.cacher.formatters import CacheFormatter, JsonFormatter

def test_write_atomic_creates_file():
    """Test that _write_atomic creates a file with the correct contents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "new_dir" / "test.txt"
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

# Keyer Tests
def test_tuple_keyer_basic():
    """Test TupleKeyer with basic types."""
    keyer = TupleKeyer()
    key = keyer.make_key((1, "test"), {"x": 2})
    assert isinstance(key, tuple)
    assert key[0] == 1
    assert key[1] == "test"
    assert isinstance(key[2], frozenset)

def test_tuple_keyer_nested():
    """Test TupleKeyer with nested structures."""
    keyer = TupleKeyer()
    args = ([1, 2], {"a": [3, 4]}, {5, 6})
    kwargs = {"x": [7, 8], "y": {"b": 9}}
    key = keyer.make_key(args, kwargs)

    # Check types are converted
    assert isinstance(key[0], tuple)  # list -> tuple
    assert isinstance(key[1], frozenset)  # dict -> frozenset
    assert isinstance(key[2], frozenset)  # set -> frozenset
    assert isinstance(key[3], frozenset)  # kwargs -> frozenset

def test_tuple_keyer_unhashable():
    """Test TupleKeyer with unhashable objects."""
    class UnhashableObject:
        def __hash__(self):
            raise TypeError("unhashable")

    keyer = TupleKeyer()
    obj = UnhashableObject()
    with pytest.raises(TypeError):
        key = keyer.make_key((obj,), {})

def test_string_keyer():
    """Test StringKeyer produces consistent string keys."""
    keyer = StringKeyer()
    key1 = keyer.make_key((1, "test"), {1: 'a', "x": 2})
    key2 = keyer.make_key((1, "test"), {"x": 2, 1: 'a'})

    assert isinstance(key1, str)
    assert key1 == key2

def test_hash_string_keyer_builtin():
    """Test HashStringKeyer with builtin hash function."""
    keyer = HashStringKeyer('sha256')
    key = keyer.make_key((1, "test"), {"x": 2})

    assert isinstance(key, str)
    assert len(key) == 64  # sha256 hex digest length

def test_hash_string_keyer_custom():
    """Test HashStringKeyer with custom hash function."""
    def custom_hash(s: str) -> str:
        return 'hash_' + s

    keyer = HashStringKeyer(custom_hash)
    key = keyer.make_key((1,), {})
    assert key.startswith('hash_')

def test_hash_bytes_keyer_builtin():
    """Test HashBytesKeyer with builtin hash function."""
    keyer = HashBytesKeyer('sha256')
    key = keyer.make_key((1, "test"), {"x": 2})

    assert isinstance(key, bytes)
    assert len(key) == 32  # sha256 raw digest length

def test_hash_bytes_keyer_custom():
    """Test HashBytesKeyer with custom hash function."""
    def custom_hash(s: str) -> bytes:
        return b'hash_' + s.encode()

    keyer = HashBytesKeyer(custom_hash)
    key = keyer.make_key((1,), {})
    assert key.startswith(b'hash_')

def test_hash_keyer_invalid_algorithm():
    """Test HashKeyers raise error for invalid hash algorithm."""
    with pytest.raises(ValueError):
        HashStringKeyer('invalid_algorithm')

    with pytest.raises(ValueError):
        HashBytesKeyer('invalid_algorithm')

# Formatter Tests
def test_json_formatter_basic():
    """Test basic JSON serialization/deserialization."""
    formatter = JsonFormatter()
    obj = {'a': 1, 'b': [2, 3], 'c': {'d': 4}}

    data = formatter.dumps(obj)
    assert isinstance(data, bytes)

    decoded = formatter.loads(data)
    assert decoded == obj

def test_json_formatter_custom_encoder():
    """Test JSON formatter with custom encoder."""
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)
            return super().default(obj)

    formatter = JsonFormatter(EncoderCls=CustomEncoder)
    obj = {'a': {1, 2, 3}}  # Sets aren't normally JSON serializable

    data = formatter.dumps(obj)
    decoded = formatter.loads(data)
    assert decoded == {'a': [1, 2, 3]}

def test_json_formatter_custom_decoder():
    """Test JSON formatter with custom decoder."""
    class CustomDecoder(json.JSONDecoder):
        def decode(self, s):
            obj = super().decode(s)
            # Convert all lists to tuples
            if isinstance(obj, list):
                return tuple(obj)
            if isinstance(obj, dict):
                return {k: tuple(v) if isinstance(v, list) else v for k, v in obj.items()}
            return obj

    formatter = JsonFormatter(DecoderCls=CustomDecoder)
    obj = {'a': [1, 2, 3]}

    data = formatter.dumps(obj)
    decoded = formatter.loads(data)
    assert decoded == {'a': (1, 2, 3)}

def test_json_formatter_invalid_input():
    """Test JSON formatter with invalid input."""
    formatter = JsonFormatter()

    # Test invalid JSON bytes
    with pytest.raises(json.JSONDecodeError):
        formatter.loads(b'invalid json')

    # Test non-serializable object
    class UnserializableObject:
        pass

    with pytest.raises(TypeError):
        formatter.dumps(UnserializableObject())
