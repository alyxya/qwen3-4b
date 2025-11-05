"""Tests for weight loading functionality"""

import pytest
from load_weights import list_weight_files


def test_list_weight_files():
    """Test that weight files can be listed"""
    weight_files = list_weight_files()

    assert isinstance(weight_files, list)
    assert len(weight_files) > 0
    assert all(isinstance(f, str) for f in weight_files)
    assert all(f.endswith(".safetensors") for f in weight_files)


def test_weight_files_sorted():
    """Test that weight files are returned in sorted order"""
    weight_files = list_weight_files()
    assert weight_files == sorted(weight_files)


# Note: We don't test load_weights() directly because it would download
# multiple GB of data. This should be tested manually or in integration tests.
