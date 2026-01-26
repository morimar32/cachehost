"""Shared fixtures and configuration for pytest."""

import os
import platform
import shutil
import tempfile
from typing import List
from unittest.mock import MagicMock

import pytest

from cachehost.config import Config
from cachehost.schemas.common import ChatMessage
from cachehost.backends.protocol import GenerationParams


def pytest_collection_modifyitems(config, items):
    """Skip macOS-only tests on non-macOS platforms."""
    if platform.system() != "Darwin":
        skip_macos = pytest.mark.skip(reason="macOS only test")
        for item in items:
            if "macos_only" in item.keywords:
                item.add_marker(skip_macos)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def temp_model_file(temp_dir):
    """Create a fake model file for testing."""
    model_path = os.path.join(temp_dir, "test_model.gguf")
    with open(model_path, "wb") as f:
        f.write(b"fake model data")
    return model_path


@pytest.fixture
def sample_config(temp_model_file, temp_dir):
    """Create a test Config with minimal settings."""
    return Config(
        model_path=temp_model_file,
        cache_dir=os.path.join(temp_dir, "cache"),
        n_ctx=2048,
        n_threads=2,
    )


@pytest.fixture
def sample_messages() -> List[ChatMessage]:
    """Sample chat messages for testing."""
    return [
        ChatMessage(role="user", content="Hello!"),
        ChatMessage(role="assistant", content="Hi there! How can I help you?"),
        ChatMessage(role="user", content="Tell me a joke."),
    ]


@pytest.fixture
def single_message() -> List[ChatMessage]:
    """Single message for simple tests."""
    return [ChatMessage(role="user", content="Hello!")]


@pytest.fixture
def system_message_conversation() -> List[ChatMessage]:
    """Conversation with a system message."""
    return [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Hello!"),
    ]


@pytest.fixture
def sample_generation_params() -> GenerationParams:
    """Sample generation parameters."""
    return GenerationParams(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_tokens=100,
    )


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    from tests.fixtures.mock_backend import MockBackend
    return MockBackend()
