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
    mock_mlx_lm_cache = MagicMock()

    # Setup mock load function
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

    # Setup mock generate function
    mock_mlx_lm.generate.return_value = "Mock generated response"

    # Setup mock stream_generate
    mock_mlx_lm.stream_generate.return_value = iter(["Hello", " ", "World"])

    # Setup mock prompt cache functions
    mock_prompt_cache = MagicMock()
    mock_mlx_lm_cache.make_prompt_cache.return_value = mock_prompt_cache
    mock_mlx_lm_cache.save_prompt_cache = MagicMock()
    mock_mlx_lm_cache.load_prompt_cache.return_value = (mock_model, mock_prompt_cache)

    # Setup mock sample_utils
    mock_sample_utils = MagicMock()
    mock_sampler = MagicMock()
    mock_sample_utils.make_sampler.return_value = mock_sampler
    mock_sample_utils.make_logits_processors.return_value = [MagicMock()]

    # Setup mock tokenizer encode
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens

    with patch.dict(
        "sys.modules",
        {
            "mlx": mock_mlx,
            "mlx.core": mock_mlx_core,
            "mlx_lm": mock_mlx_lm,
            "mlx_lm.models": MagicMock(),
            "mlx_lm.models.cache": mock_mlx_lm_cache,
            "mlx_lm.sample_utils": mock_sample_utils,
        },
    ):
        yield {
            "mlx": mock_mlx,
            "mlx_core": mock_mlx_core,
            "mlx_lm": mock_mlx_lm,
            "mlx_lm_cache": mock_mlx_lm_cache,
            "sample_utils": mock_sample_utils,
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "prompt_cache": mock_prompt_cache,
            "sampler": mock_sampler,
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

    def test_load_initializes_prompt_cache(self, mlx_backend, mock_mlx_modules):
        """Test that load() initializes the prompt cache."""
        mlx_backend.load()

        mock_mlx_modules["mlx_lm_cache"].make_prompt_cache.assert_called_once_with(
            mock_mlx_modules["model"]
        )
        assert mlx_backend._prompt_cache is not None


class TestMLXBackendReset:
    """Tests for MLXBackend reset method."""

    def test_reset_creates_fresh_prompt_cache(self, mlx_backend, mock_mlx_modules):
        """Test that reset creates a fresh prompt cache."""
        mlx_backend.load()
        initial_cache = mlx_backend._prompt_cache

        # Reset should create a new cache
        mlx_backend.reset()

        # make_prompt_cache should be called twice (once in load, once in reset)
        assert mock_mlx_modules["mlx_lm_cache"].make_prompt_cache.call_count == 2

    def test_reset_clears_cache_when_model_not_loaded(self, mlx_backend):
        """Test that reset clears cache when model is not loaded."""
        mlx_backend._prompt_cache = MagicMock()
        mlx_backend.reset()

        assert mlx_backend._prompt_cache is None


class TestMLXBackendStateMethods:
    """Tests for MLXBackend save_state and load_state methods."""

    def test_save_state_returns_mlx_cache_dict(self, mlx_backend, mock_mlx_modules):
        """Test that save_state returns dict with MLX marker and cache."""
        mlx_backend.load()
        state = mlx_backend.save_state()

        assert isinstance(state, dict)
        assert state.get("_mlx_prompt_cache") is True
        assert "cache" in state
        assert state["cache"] == mock_mlx_modules["prompt_cache"]

    def test_save_state_returns_none_when_no_cache(self, mlx_backend, caplog):
        """Test that save_state returns None when no cache exists."""
        import logging

        with caplog.at_level(logging.WARNING):
            state = mlx_backend.save_state()

        assert state is None
        assert "No prompt cache to save" in caplog.text

    def test_load_state_sets_prompt_cache(self, mlx_backend, mock_mlx_modules):
        """Test that load_state sets the prompt cache."""
        mlx_backend.load()

        new_cache = MagicMock()
        state = {
            "_mlx_prompt_cache": True,
            "cache": new_cache,
        }

        mlx_backend.load_state(state)

        assert mlx_backend._prompt_cache == new_cache

    def test_load_state_handles_none(self, mlx_backend, mock_mlx_modules):
        """Test that load_state handles None state."""
        mlx_backend.load()
        mlx_backend.load_state(None)

        assert mlx_backend._prompt_cache is None

    def test_load_state_warns_on_invalid_format(self, mlx_backend, caplog):
        """Test that load_state warns on invalid state format."""
        import logging

        mlx_backend.load()
        with caplog.at_level(logging.WARNING):
            mlx_backend.load_state({"invalid": "state"})

        assert "Invalid MLX state format" in caplog.text


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

    def test_generate_passes_prompt_cache(self, mlx_backend, mock_mlx_modules):
        """Test that generate passes prompt_cache to mlx_lm.generate."""
        from cachehost.backends.protocol import GenerationParams
        from cachehost.schemas.common import ChatMessage

        mlx_backend.load()
        mlx_backend.generate(
            [ChatMessage(role="user", content="Hello")],
            GenerationParams(),
        )

        # Check that prompt_cache was passed in kwargs
        call_kwargs = mock_mlx_modules["mlx_lm"].generate.call_args.kwargs
        assert "prompt_cache" in call_kwargs
        assert call_kwargs["prompt_cache"] == mock_mlx_modules["prompt_cache"]

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

    def test_generate_counts_tokens(self, mlx_backend, mock_mlx_modules):
        """Test that generate counts tokens."""
        from cachehost.backends.protocol import GenerationParams
        from cachehost.schemas.common import ChatMessage

        mlx_backend.load()
        result = mlx_backend.generate(
            [ChatMessage(role="user", content="Hello")],
            GenerationParams(),
        )

        # Tokenizer returns [1,2,3,4,5] = 5 tokens for any input
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 5
        assert result.total_tokens == 10


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

    def test_generate_stream_passes_prompt_cache(self, mlx_backend, mock_mlx_modules):
        """Test that generate_stream passes prompt_cache."""
        from cachehost.backends.protocol import GenerationParams
        from cachehost.schemas.common import ChatMessage

        mlx_backend.load()
        list(
            mlx_backend.generate_stream(
                [ChatMessage(role="user", content="Hello")],
                GenerationParams(),
            )
        )

        # Check that prompt_cache was passed in kwargs
        call_kwargs = mock_mlx_modules["mlx_lm"].stream_generate.call_args.kwargs
        assert "prompt_cache" in call_kwargs
        assert call_kwargs["prompt_cache"] == mock_mlx_modules["prompt_cache"]

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

    def test_generate_stream_final_chunk_has_token_counts(self, mlx_backend, mock_mlx_modules):
        """Test that final chunk has token counts."""
        from cachehost.backends.protocol import GenerationParams
        from cachehost.schemas.common import ChatMessage

        mlx_backend.load()
        chunks = list(
            mlx_backend.generate_stream(
                [ChatMessage(role="user", content="Hello")],
                GenerationParams(),
            )
        )

        final_chunk = chunks[-1]
        assert final_chunk.prompt_tokens == 5  # From mock tokenizer
        assert final_chunk.completion_tokens == 3  # "Hello", " ", "World"
        assert final_chunk.total_tokens == 8


class TestMLXBackendShutdown:
    """Tests for MLXBackend shutdown method."""

    def test_shutdown_clears_model(self, mlx_backend, mock_mlx_modules):
        """Test that shutdown clears model and tokenizer."""
        mlx_backend.load()
        assert mlx_backend._model is not None

        mlx_backend.shutdown()

        assert mlx_backend._model is None
        assert mlx_backend._tokenizer is None
        assert mlx_backend._prompt_cache is None


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


class TestMLXBackendBuildGenerationKwargs:
    """Tests for MLXBackend _build_generation_kwargs method."""

    def test_returns_sampler(self, mlx_backend, mock_mlx_modules):
        """Test that kwargs include a sampler callable."""
        from cachehost.backends.protocol import GenerationParams

        mlx_backend.load()
        params = GenerationParams(temperature=0.5, top_p=0.8)
        kwargs = mlx_backend._build_generation_kwargs(params)

        assert "sampler" in kwargs
        assert kwargs["sampler"] == mock_mlx_modules["sampler"]

    def test_make_sampler_called_with_basic_params(self, mlx_backend, mock_mlx_modules):
        """Test that make_sampler is called with temp and top_p."""
        from cachehost.backends.protocol import GenerationParams

        mlx_backend.load()
        params = GenerationParams(temperature=0.5, top_p=0.8, top_k=0, min_p=0)
        mlx_backend._build_generation_kwargs(params)

        mock_mlx_modules["sample_utils"].make_sampler.assert_called_once_with(
            temp=0.5, top_p=0.8,
        )

    def test_make_sampler_includes_top_k_when_positive(self, mlx_backend, mock_mlx_modules):
        """Test that top_k is passed to make_sampler when > 0."""
        from cachehost.backends.protocol import GenerationParams

        mlx_backend.load()
        params = GenerationParams(top_k=50, min_p=0)
        mlx_backend._build_generation_kwargs(params)

        call_kwargs = mock_mlx_modules["sample_utils"].make_sampler.call_args.kwargs
        assert call_kwargs["top_k"] == 50

    def test_make_sampler_excludes_top_k_when_zero(self, mlx_backend, mock_mlx_modules):
        """Test that top_k is not passed to make_sampler when 0."""
        from cachehost.backends.protocol import GenerationParams

        mlx_backend.load()
        params = GenerationParams(top_k=0, min_p=0)
        mlx_backend._build_generation_kwargs(params)

        call_kwargs = mock_mlx_modules["sample_utils"].make_sampler.call_args.kwargs
        assert "top_k" not in call_kwargs

    def test_make_sampler_includes_min_p_when_positive(self, mlx_backend, mock_mlx_modules):
        """Test that min_p is passed to make_sampler when > 0."""
        from cachehost.backends.protocol import GenerationParams

        mlx_backend.load()
        params = GenerationParams(min_p=0.1, top_k=0)
        mlx_backend._build_generation_kwargs(params)

        call_kwargs = mock_mlx_modules["sample_utils"].make_sampler.call_args.kwargs
        assert call_kwargs["min_p"] == 0.1

    def test_includes_logits_processors_for_repeat_penalty(self, mlx_backend, mock_mlx_modules):
        """Test that logits_processors is included when repeat_penalty != 1.0."""
        from cachehost.backends.protocol import GenerationParams

        mlx_backend.load()
        params = GenerationParams(repeat_penalty=1.2)
        kwargs = mlx_backend._build_generation_kwargs(params)

        assert "logits_processors" in kwargs
        mock_mlx_modules["sample_utils"].make_logits_processors.assert_called_once_with(
            repetition_penalty=1.2,
        )

    def test_excludes_logits_processors_when_no_repeat_penalty(self, mlx_backend, mock_mlx_modules):
        """Test that logits_processors is excluded when repeat_penalty is 1.0."""
        from cachehost.backends.protocol import GenerationParams

        mlx_backend.load()
        params = GenerationParams(repeat_penalty=1.0)
        kwargs = mlx_backend._build_generation_kwargs(params)

        assert "logits_processors" not in kwargs

    def test_includes_max_tokens(self, mlx_backend, mock_mlx_modules):
        """Test that max_tokens is included."""
        from cachehost.backends.protocol import GenerationParams

        mlx_backend.load()
        params = GenerationParams(max_tokens=100)
        kwargs = mlx_backend._build_generation_kwargs(params)

        assert kwargs["max_tokens"] == 100
