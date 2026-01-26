"""Tests for MLX backend (macOS only)."""

import platform
from unittest.mock import MagicMock, patch

import pytest

# Mark entire module as macOS-only
pytestmark = pytest.mark.macos_only


@pytest.fixture
def mock_mlx_modules():
    """Mock MLX modules for testing."""
    mock_mlx = MagicMock()
    mock_mlx_core = MagicMock()
    mock_mlx_lm = MagicMock()

    # Setup mock load function
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

    # Setup mock generate function
    mock_mlx_lm.generate.return_value = "Mock generated response"

    # Setup mock stream_generate
    mock_mlx_lm.stream_generate.return_value = iter(["Hello", " ", "World"])

    with patch.dict(
        "sys.modules",
        {
            "mlx": mock_mlx,
            "mlx.core": mock_mlx_core,
            "mlx_lm": mock_mlx_lm,
        },
    ):
        yield {
            "mlx": mock_mlx,
            "mlx_core": mock_mlx_core,
            "mlx_lm": mock_mlx_lm,
            "model": mock_model,
            "tokenizer": mock_tokenizer,
        }


@pytest.fixture
def mlx_backend(sample_config, mock_mlx_modules):
    """Create MLX backend with mocked dependencies."""
    from cachehost.backends.mlx_backend import MLXBackend

    backend = MLXBackend(sample_config)
    return backend


class TestMLXBackendProperties:
    """Tests for MLXBackend properties."""

    def test_model_name(self, mlx_backend, sample_config):
        """Test model_name property."""
        assert mlx_backend.model_name == sample_config.model_name

    def test_backend_name(self, mlx_backend):
        """Test backend_name property."""
        assert mlx_backend.backend_name == "mlx"


class TestMLXBackendLoad:
    """Tests for MLXBackend load method."""

    def test_load_calls_mlx_lm_load(self, mlx_backend, mock_mlx_modules, sample_config):
        """Test that load() calls mlx_lm.load."""
        mlx_backend.load()

        mock_mlx_modules["mlx_lm"].load.assert_called_once_with(
            sample_config.model_path
        )

    def test_load_sets_model_and_tokenizer(self, mlx_backend, mock_mlx_modules):
        """Test that load() sets internal model and tokenizer."""
        mlx_backend.load()

        assert mlx_backend._model is not None
        assert mlx_backend._tokenizer is not None


class TestMLXBackendReset:
    """Tests for MLXBackend reset method."""

    def test_reset_clears_state(self, mlx_backend):
        """Test that reset clears internal state."""
        mlx_backend._state = {"some": "state"}
        mlx_backend.reset()

        assert mlx_backend._state is None


class TestMLXBackendStateMethods:
    """Tests for MLXBackend save_state and load_state methods."""

    def test_save_state_returns_state(self, mlx_backend):
        """Test that save_state returns current state."""
        mlx_backend._state = {"test": "data"}
        state = mlx_backend.save_state()

        # MLX backend currently returns the internal state
        assert state == {"test": "data"}

    def test_save_state_logs_warning(self, mlx_backend, caplog):
        """Test that save_state logs a warning about limited implementation."""
        import logging

        with caplog.at_level(logging.WARNING):
            mlx_backend.save_state()

        assert "not fully implemented" in caplog.text

    def test_load_state_sets_state(self, mlx_backend):
        """Test that load_state sets internal state."""
        state = {"loaded": "state"}
        mlx_backend.load_state(state)

        assert mlx_backend._state == state

    def test_load_state_logs_warning(self, mlx_backend, caplog):
        """Test that load_state logs a warning about limited implementation."""
        import logging

        with caplog.at_level(logging.WARNING):
            mlx_backend.load_state({"test": "state"})

        assert "not fully implemented" in caplog.text


class TestMLXBackendGenerate:
    """Tests for MLXBackend generate method."""

    def test_generate_raises_if_not_loaded(self, mlx_backend):
        """Test that generate raises if model not loaded."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            from cachehost.backends.protocol import GenerationParams
            from cachehost.schemas.common import ChatMessage

            mlx_backend.generate(
                [ChatMessage(role="user", content="Hello")],
                GenerationParams(),
            )

    def test_generate_calls_mlx_lm_generate(self, mlx_backend, mock_mlx_modules):
        """Test that generate calls mlx_lm.generate."""
        from cachehost.backends.protocol import GenerationParams
        from cachehost.schemas.common import ChatMessage

        mlx_backend.load()
        result = mlx_backend.generate(
            [ChatMessage(role="user", content="Hello")],
            GenerationParams(temperature=0.5, top_p=0.9, max_tokens=50),
        )

        mock_mlx_modules["mlx_lm"].generate.assert_called_once()
        assert result.content == "Mock generated response"
        assert result.finish_reason == "stop"

    def test_generate_returns_generation_result(self, mlx_backend, mock_mlx_modules):
        """Test that generate returns GenerationResult."""
        from cachehost.backends.protocol import GenerationParams, GenerationResult
        from cachehost.schemas.common import ChatMessage

        mlx_backend.load()
        result = mlx_backend.generate(
            [ChatMessage(role="user", content="Hello")],
            GenerationParams(),
        )

        assert isinstance(result, GenerationResult)
        assert result.content == "Mock generated response"


class TestMLXBackendGenerateStream:
    """Tests for MLXBackend generate_stream method."""

    def test_generate_stream_raises_if_not_loaded(self, mlx_backend):
        """Test that generate_stream raises if model not loaded."""
        from cachehost.backends.protocol import GenerationParams
        from cachehost.schemas.common import ChatMessage

        with pytest.raises(RuntimeError, match="Model not loaded"):
            list(
                mlx_backend.generate_stream(
                    [ChatMessage(role="user", content="Hello")],
                    GenerationParams(),
                )
            )

    def test_generate_stream_yields_chunks(self, mlx_backend, mock_mlx_modules):
        """Test that generate_stream yields GenerationChunks."""
        from cachehost.backends.protocol import GenerationChunk, GenerationParams
        from cachehost.schemas.common import ChatMessage

        mlx_backend.load()
        chunks = list(
            mlx_backend.generate_stream(
                [ChatMessage(role="user", content="Hello")],
                GenerationParams(),
            )
        )

        # First chunk has role
        assert chunks[0].role == "assistant"

        # Middle chunks have content
        assert any(c.content for c in chunks)

        # Last chunk is final
        assert chunks[-1].is_final is True
        assert chunks[-1].finish_reason == "stop"

    def test_generate_stream_first_chunk_has_role(self, mlx_backend, mock_mlx_modules):
        """Test that first chunk has assistant role."""
        from cachehost.backends.protocol import GenerationParams
        from cachehost.schemas.common import ChatMessage

        mlx_backend.load()
        chunks = list(
            mlx_backend.generate_stream(
                [ChatMessage(role="user", content="Hello")],
                GenerationParams(),
            )
        )

        assert chunks[0].role == "assistant"
        assert chunks[0].content == ""


class TestMLXBackendShutdown:
    """Tests for MLXBackend shutdown method."""

    def test_shutdown_clears_model(self, mlx_backend, mock_mlx_modules):
        """Test that shutdown clears model and tokenizer."""
        mlx_backend.load()
        assert mlx_backend._model is not None

        mlx_backend.shutdown()

        assert mlx_backend._model is None
        assert mlx_backend._tokenizer is None


class TestMLXBackendFormatMessages:
    """Tests for MLXBackend _format_messages method."""

    def test_format_messages_with_chat_template(self, mlx_backend, mock_mlx_modules):
        """Test formatting when tokenizer has chat template."""
        from cachehost.schemas.common import ChatMessage

        mlx_backend.load()

        # Setup tokenizer with chat template
        mock_tokenizer = mock_mlx_modules["tokenizer"]
        mock_tokenizer.apply_chat_template.return_value = "Formatted prompt"

        messages = [
            ChatMessage(role="system", content="Be helpful"),
            ChatMessage(role="user", content="Hello"),
        ]

        result = mlx_backend._format_messages(messages)

        assert result == "Formatted prompt"
        mock_tokenizer.apply_chat_template.assert_called_once()

    def test_format_messages_fallback(self, mlx_backend, mock_mlx_modules):
        """Test fallback formatting when no chat template."""
        from cachehost.schemas.common import ChatMessage

        mlx_backend.load()

        # Remove chat template method
        mock_tokenizer = mock_mlx_modules["tokenizer"]
        del mock_tokenizer.apply_chat_template

        messages = [
            ChatMessage(role="system", content="Be helpful"),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
        ]

        result = mlx_backend._format_messages(messages)

        assert "System: Be helpful" in result
        assert "User: Hello" in result
        assert "Assistant: Hi there" in result
        assert result.endswith("Assistant:")
