"""Pytest fixtures for Recaller tests."""

import tempfile
from pathlib import Path

import pytest

from recaller.database.repository import Repository


@pytest.fixture
def temp_db_path():
    """Provide a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def repository(temp_db_path):
    """Provide a repository with a temporary database."""
    return Repository(f"sqlite:///{temp_db_path}")
