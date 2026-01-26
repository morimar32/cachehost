"""Tests for CacheManager."""

import hashlib
import os
import pickle
import tempfile
import shutil

import pytest

from cachehost.cache.manager import CacheManager
from cachehost.schemas.common import ChatMessage


class TestCacheManagerInit:
    """Tests for CacheManager initialization."""

    def test_init_sets_attributes(self, temp_dir):
        """Test that __init__ sets all expected attributes."""
        manager = CacheManager(
            cache_dir=temp_dir,
            model_name="test_model",
            backend_name="test_backend",
        )
        assert manager.cache_dir == temp_dir
        assert manager.model_name == "test_model"
        assert manager.backend_name == "test_backend"
        assert manager.model_cache_dir == os.path.join(
            temp_dir, "test_model", "test_backend"
        )

    def test_init_creates_cache_path(self, temp_dir):
        """Test that init() creates the cache directory structure."""
        cache_dir = os.path.join(temp_dir, "cache")
        manager = CacheManager(
            cache_dir=cache_dir,
            model_name="my_model",
            backend_name="llama_cpp",
        )
        manager.init()

        expected_path = os.path.join(cache_dir, "my_model", "llama_cpp")
        assert os.path.exists(expected_path)

    def test_init_clears_existing_cache(self, temp_dir):
        """Test that init() clears any existing cache."""
        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir)

        # Create a file in the cache directory
        existing_file = os.path.join(cache_dir, "existing.txt")
        with open(existing_file, "w") as f:
            f.write("existing content")

        manager = CacheManager(
            cache_dir=cache_dir,
            model_name="model",
            backend_name="backend",
        )
        manager.init()

        # The old file should be gone
        assert not os.path.exists(existing_file)
        # But the new structure should exist
        assert os.path.exists(manager.model_cache_dir)


class TestComputeCacheKey:
    """Tests for _compute_cache_key method."""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create a CacheManager for testing."""
        return CacheManager(
            cache_dir=temp_dir,
            model_name="test_model",
            backend_name="test",
        )

    def test_empty_history_returns_special_key(self, manager):
        """Test that empty history returns a deterministic special key."""
        key1 = manager._compute_cache_key([])
        key2 = manager._compute_cache_key([])

        # Should be consistent
        assert key1 == key2

        # Should be a valid SHA256 hash
        assert len(key1) == 64
        assert all(c in "0123456789abcdef" for c in key1)

        # Verify the expected format
        expected_content = "test_model__initial_state__"
        expected_hash = hashlib.sha256(expected_content.encode("utf-8")).hexdigest()
        assert key1 == expected_hash

    def test_single_message_key(self, manager):
        """Test key generation for a single message."""
        messages = [ChatMessage(role="user", content="Hello")]
        key = manager._compute_cache_key(messages)

        # Compute expected hash
        expected_content = "test_model" + "user:Hello"
        expected_hash = hashlib.sha256(expected_content.encode("utf-8")).hexdigest()
        assert key == expected_hash

    def test_multiple_messages_key(self, manager):
        """Test key generation for multiple messages."""
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
        ]
        key = manager._compute_cache_key(messages)

        expected_content = "test_model" + "user:Hello" + "assistant:Hi there"
        expected_hash = hashlib.sha256(expected_content.encode("utf-8")).hexdigest()
        assert key == expected_hash

    def test_role_affects_key(self, manager):
        """Test that different roles produce different keys."""
        msg1 = [ChatMessage(role="user", content="Hello")]
        msg2 = [ChatMessage(role="assistant", content="Hello")]

        key1 = manager._compute_cache_key(msg1)
        key2 = manager._compute_cache_key(msg2)

        assert key1 != key2

    def test_content_affects_key(self, manager):
        """Test that different content produces different keys."""
        msg1 = [ChatMessage(role="user", content="Hello")]
        msg2 = [ChatMessage(role="user", content="Goodbye")]

        key1 = manager._compute_cache_key(msg1)
        key2 = manager._compute_cache_key(msg2)

        assert key1 != key2

    def test_order_affects_key(self, manager):
        """Test that message order affects the key."""
        msg1 = [
            ChatMessage(role="user", content="A"),
            ChatMessage(role="assistant", content="B"),
        ]
        msg2 = [
            ChatMessage(role="assistant", content="B"),
            ChatMessage(role="user", content="A"),
        ]

        key1 = manager._compute_cache_key(msg1)
        key2 = manager._compute_cache_key(msg2)

        assert key1 != key2


class TestGetCacheFilepath:
    """Tests for get_cache_filepath method."""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create a CacheManager for testing."""
        mgr = CacheManager(
            cache_dir=temp_dir,
            model_name="model",
            backend_name="backend",
        )
        mgr.init()
        return mgr

    def test_returns_pkl_path(self, manager):
        """Test that the returned path has .pkl extension."""
        messages = [ChatMessage(role="user", content="test")]
        filepath = manager.get_cache_filepath(messages)

        assert filepath.endswith(".pkl")

    def test_path_includes_model_cache_dir(self, manager):
        """Test that the path is under model_cache_dir."""
        messages = [ChatMessage(role="user", content="test")]
        filepath = manager.get_cache_filepath(messages)

        assert filepath.startswith(manager.model_cache_dir)

    def test_path_uses_correct_hash(self, manager):
        """Test that the filename is the computed hash."""
        messages = [ChatMessage(role="user", content="test")]
        filepath = manager.get_cache_filepath(messages)

        expected_key = manager._compute_cache_key(messages)
        expected_filename = f"{expected_key}.pkl"

        assert os.path.basename(filepath) == expected_filename


class TestSaveAndLoadState:
    """Tests for save_state and load_state methods."""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create a CacheManager for testing."""
        mgr = CacheManager(
            cache_dir=temp_dir,
            model_name="model",
            backend_name="backend",
        )
        mgr.init()
        return mgr

    def test_save_state_creates_file(self, manager):
        """Test that save_state creates a cache file."""
        messages = [ChatMessage(role="user", content="test")]
        state = {"key": "value", "data": [1, 2, 3]}

        result = manager.save_state(messages, state)

        assert result is True
        filepath = manager.get_cache_filepath(messages)
        assert os.path.exists(filepath)

    def test_save_state_uses_pickle(self, manager):
        """Test that saved state can be loaded with pickle."""
        messages = [ChatMessage(role="user", content="test")]
        state = {"key": "value"}

        manager.save_state(messages, state)
        filepath = manager.get_cache_filepath(messages)

        with open(filepath, "rb") as f:
            loaded = pickle.load(f)

        assert loaded == state

    def test_load_state_returns_saved_state(self, manager):
        """Test that load_state returns the saved state."""
        messages = [ChatMessage(role="user", content="test")]
        state = {"data": [1, 2, 3], "nested": {"a": "b"}}

        manager.save_state(messages, state)
        loaded = manager.load_state(messages)

        assert loaded == state

    def test_load_state_returns_none_for_missing(self, manager):
        """Test that load_state returns None for non-existent cache."""
        messages = [ChatMessage(role="user", content="nonexistent")]
        loaded = manager.load_state(messages)

        assert loaded is None

    def test_save_state_returns_false_on_error(self, manager, temp_dir):
        """Test that save_state returns False when it can't write."""
        messages = [ChatMessage(role="user", content="test")]

        # Remove write permission from cache dir
        os.chmod(manager.model_cache_dir, 0o444)

        try:
            result = manager.save_state(messages, {"data": "test"})
            assert result is False
        finally:
            os.chmod(manager.model_cache_dir, 0o755)

    def test_load_state_handles_corrupt_file(self, manager):
        """Test that load_state handles corrupt pickle files gracefully."""
        messages = [ChatMessage(role="user", content="test")]
        filepath = manager.get_cache_filepath(messages)

        # Write corrupt data
        with open(filepath, "wb") as f:
            f.write(b"not a valid pickle")

        loaded = manager.load_state(messages)
        assert loaded is None


class TestFindLongestPrefixMatch:
    """Tests for find_longest_prefix_match method."""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create a CacheManager for testing."""
        mgr = CacheManager(
            cache_dir=temp_dir,
            model_name="model",
            backend_name="backend",
        )
        mgr.init()
        return mgr

    def test_no_cache_returns_none_and_zero(self, manager):
        """Test that no cached state returns (None, 0)."""
        messages = [ChatMessage(role="user", content="test")]
        state, length = manager.find_longest_prefix_match(messages)

        assert state is None
        assert length == 0

    def test_empty_messages_returns_none_and_zero(self, manager):
        """Test with empty message list."""
        state, length = manager.find_longest_prefix_match([])

        assert state is None
        assert length == 0

    def test_finds_exact_match(self, manager):
        """Test that exact prefix match is found."""
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi"),
        ]
        saved_state = {"cached": True}

        manager.save_state(messages, saved_state)
        state, length = manager.find_longest_prefix_match(messages)

        assert state == saved_state
        assert length == 2

    def test_finds_partial_prefix_match(self, manager):
        """Test finding a cached prefix of the full history."""
        msg1 = ChatMessage(role="user", content="Hello")
        msg2 = ChatMessage(role="assistant", content="Hi")
        msg3 = ChatMessage(role="user", content="How are you?")

        # Cache only the first message
        manager.save_state([msg1], {"state_1": True})

        # Search with more messages
        full_messages = [msg1, msg2, msg3]
        state, length = manager.find_longest_prefix_match(full_messages)

        assert state == {"state_1": True}
        assert length == 1

    def test_finds_longest_match(self, manager):
        """Test that the longest matching prefix is returned."""
        msg1 = ChatMessage(role="user", content="A")
        msg2 = ChatMessage(role="assistant", content="B")
        msg3 = ChatMessage(role="user", content="C")

        # Cache multiple prefixes
        manager.save_state([msg1], {"length": 1})
        manager.save_state([msg1, msg2], {"length": 2})

        full_messages = [msg1, msg2, msg3]
        state, length = manager.find_longest_prefix_match(full_messages)

        # Should find the longest match (length 2)
        assert state == {"length": 2}
        assert length == 2

    def test_returns_full_match_when_available(self, manager):
        """Test that full message list match is preferred."""
        msg1 = ChatMessage(role="user", content="A")
        msg2 = ChatMessage(role="assistant", content="B")

        # Cache all prefixes
        manager.save_state([msg1], {"length": 1})
        manager.save_state([msg1, msg2], {"length": 2})

        full_messages = [msg1, msg2]
        state, length = manager.find_longest_prefix_match(full_messages)

        assert state == {"length": 2}
        assert length == 2


class TestCacheCleanup:
    """Tests for cache cleanup functionality."""

    def test_cleanup_removes_cache_dir(self, temp_dir):
        """Test that cleanup removes the cache directory."""
        cache_dir = os.path.join(temp_dir, "cache")
        manager = CacheManager(
            cache_dir=cache_dir,
            model_name="model",
            backend_name="backend",
        )
        manager.init()

        # Verify cache exists
        assert os.path.exists(cache_dir)

        # Cleanup
        manager.cleanup()

        # Cache should be gone
        assert not os.path.exists(cache_dir)

    def test_cleanup_handles_missing_dir(self, temp_dir):
        """Test that cleanup handles already-deleted cache gracefully."""
        cache_dir = os.path.join(temp_dir, "cache")
        manager = CacheManager(
            cache_dir=cache_dir,
            model_name="model",
            backend_name="backend",
        )
        # Don't call init(), so no dir exists

        # Should not raise
        manager.cleanup()
