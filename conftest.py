"""Global pytest configuration and fixtures."""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "unsafe: marks tests as unsafe (requiring real credentials/API calls)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip unsafe tests unless RUN_UNSAFE_TESTS environment variable is set."""
    run_unsafe = os.getenv("RUN_UNSAFE_TESTS") == "1"

    if not run_unsafe:
        skip_unsafe = pytest.mark.skip(reason="Unsafe test - set RUN_UNSAFE_TESTS=1 to run")
        for item in items:
            if "unsafe" in item.keywords:
                item.add_marker(skip_unsafe)
