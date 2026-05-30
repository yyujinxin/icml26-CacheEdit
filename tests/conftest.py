"""Pytest configuration and shared fixtures for CacheEdit tests."""

import sys
from pathlib import Path

# Ensure repo root is on sys.path so 'cache_edit' is importable when running
# pytest from any directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import torch


@pytest.fixture
def cpu_device():
    """A CPU device fixture for tests that should not require CUDA."""
    return torch.device("cpu")


@pytest.fixture
def small_tensor():
    """A small deterministic tensor for shape/dtype assertions."""
    return torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
