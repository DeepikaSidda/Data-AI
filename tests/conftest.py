"""Shared pytest fixtures for the ranking test suite.

STUB: fixtures are fleshed out alongside the tests in later tasks (sample
candidate records, deterministic fake embeddings, temp artifact paths, etc.).
Ensures the repository root is importable so ``import ranking`` works when
pytest is invoked from any working directory.
"""

from __future__ import annotations

import os
import sys

import pytest

# Make the repository root importable regardless of pytest's rootdir.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@pytest.fixture
def repo_root() -> str:
    """Absolute path to the repository root."""
    return _REPO_ROOT
