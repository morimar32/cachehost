"""Tests for backend protocol dataclasses."""

import pytest

from cachehost.backends.protocol import (
    GenerationChunk,
    GenerationParams,
    GenerationResult,
)


class TestGenerationParams:
    """Tests for GenerationParams dataclass."""

    def test_default_values(self):
        """Test all default values."""
        params = GenerationParams()

        assert params.temperature == 0.7
        assert params.top_p == 0.95
        assert params.top_k == 40
        assert params.min_p == 0.05
        assert params.typical_p == 1.0
        assert params.stop is None
        assert params.max_tokens is None
        assert params.presence_penalty == 0.0
        assert params.frequency_penalty == 0.0
        assert params.repeat_penalty == 1.1
        assert params.tfs_z == 1.0
        assert params.mirostat_mode == 0
        assert params.mirostat_tau == 5.0
        assert params.mirostat_eta == 0.1
        assert params.grammar is None

    def test_custom_values(self):
        """Test setting custom values."""
        params = GenerationParams(
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            min_p=0.1,
            typical_p=0.95,
            stop=["END", "STOP"],
            max_tokens=100,
            presence_penalty=0.5,
            frequency_penalty=0.3,
            repeat_penalty=1.2,
            tfs_z=0.9,
            mirostat_mode=2,
            mirostat_tau=4.0,
            mirostat_eta=0.2,
            grammar="root ::= 'hello'",
        )

        assert params.temperature == 0.5
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.min_p == 0.1
        assert params.typical_p == 0.95
        assert params.stop == ["END", "STOP"]
        assert params.max_tokens == 100
        assert params.presence_penalty == 0.5
        assert params.frequency_penalty == 0.3
        assert params.repeat_penalty == 1.2
        assert params.tfs_z == 0.9
        assert params.mirostat_mode == 2
        assert params.mirostat_tau == 4.0
        assert params.mirostat_eta == 0.2
        assert params.grammar == "root ::= 'hello'"

    def test_stop_sequences_list(self):
        """Test that stop can be a list of strings."""
        params = GenerationParams(stop=["END", "STOP", "EXIT"])
        assert len(params.stop) == 3
        assert "END" in params.stop

    def test_stop_sequences_none(self):
        """Test that stop defaults to None."""
        params = GenerationParams()
        assert params.stop is None

    def test_grammar_string(self):
        """Test grammar as a string."""
        grammar = 'root ::= "yes" | "no"'
        params = GenerationParams(grammar=grammar)
        assert params.grammar == grammar


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_required_content(self):
        """Test that content is required."""
        result = GenerationResult(content="Hello, world!")
        assert result.content == "Hello, world!"

    def test_default_values(self):
        """Test default values for optional fields."""
        result = GenerationResult(content="test")

        assert result.finish_reason is None
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0
        assert result.raw_response is None

    def test_all_fields(self):
        """Test setting all fields."""
        result = GenerationResult(
            content="Generated text",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            raw_response={"model": "test", "usage": {}},
        )

        assert result.content == "Generated text"
        assert result.finish_reason == "stop"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.total_tokens == 15
        assert result.raw_response["model"] == "test"

    def test_finish_reasons(self):
        """Test various finish reasons."""
        for reason in ["stop", "length", "content_filter", "tool_calls"]:
            result = GenerationResult(content="", finish_reason=reason)
            assert result.finish_reason == reason

    def test_empty_content(self):
        """Test empty content is allowed."""
        result = GenerationResult(content="")
        assert result.content == ""

    def test_multiline_content(self):
        """Test multiline content."""
        content = "Line 1\nLine 2\nLine 3"
        result = GenerationResult(content=content)
        assert result.content == content


class TestGenerationChunk:
    """Tests for GenerationChunk dataclass."""

    def test_all_defaults(self):
        """Test all default values."""
        chunk = GenerationChunk()

        assert chunk.content == ""
        assert chunk.role is None
        assert chunk.finish_reason is None
        assert chunk.prompt_tokens == 0
        assert chunk.completion_tokens == 0
        assert chunk.total_tokens == 0
        assert chunk.is_final is False

    def test_content_chunk(self):
        """Test a content chunk."""
        chunk = GenerationChunk(content="Hello")
        assert chunk.content == "Hello"
        assert chunk.is_final is False

    def test_role_chunk(self):
        """Test a chunk with role (first chunk)."""
        chunk = GenerationChunk(content="", role="assistant")
        assert chunk.role == "assistant"
        assert chunk.content == ""

    def test_final_chunk(self):
        """Test a final chunk with finish reason."""
        chunk = GenerationChunk(
            content="",
            finish_reason="stop",
            is_final=True,
        )

        assert chunk.finish_reason == "stop"
        assert chunk.is_final is True

    def test_final_chunk_with_usage(self):
        """Test a final chunk with usage information."""
        chunk = GenerationChunk(
            content="",
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            is_final=True,
        )

        assert chunk.prompt_tokens == 50
        assert chunk.completion_tokens == 20
        assert chunk.total_tokens == 70
        assert chunk.is_final is True

    def test_is_final_flag(self):
        """Test is_final flag behavior."""
        # Non-final chunks
        chunk1 = GenerationChunk(content="Hello")
        assert chunk1.is_final is False

        # Final chunk
        chunk2 = GenerationChunk(content="", is_final=True)
        assert chunk2.is_final is True

    def test_streaming_sequence(self):
        """Test typical streaming chunk sequence."""
        chunks = [
            # First chunk with role
            GenerationChunk(content="", role="assistant"),
            # Content chunks
            GenerationChunk(content="Hello"),
            GenerationChunk(content=" "),
            GenerationChunk(content="world"),
            GenerationChunk(content="!"),
            # Final chunk
            GenerationChunk(
                content="",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                is_final=True,
            ),
        ]

        # First chunk has role
        assert chunks[0].role == "assistant"
        assert chunks[0].content == ""

        # Middle chunks have content
        assert chunks[1].content == "Hello"
        assert chunks[2].content == " "

        # Last chunk is final with usage
        assert chunks[-1].is_final is True
        assert chunks[-1].finish_reason == "stop"
        assert chunks[-1].total_tokens == 15
