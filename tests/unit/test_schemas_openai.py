"""Tests for OpenAI schema definitions."""

import time
import uuid

import pytest
from pydantic import ValidationError

from cachehost.schemas.common import ChatMessage
from cachehost.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
    UsageInfo,
)


class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest model."""

    def test_messages_required(self):
        """Test that messages field is required."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest()

    def test_minimal_request(self):
        """Test creating request with only required fields."""
        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatCompletionRequest(messages=messages)

        assert request.messages == messages
        assert request.model is None

    def test_default_values(self):
        """Test default values for optional fields."""
        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatCompletionRequest(messages=messages)

        assert request.temperature == 0.7
        assert request.top_p == 0.95
        assert request.top_k == 40
        assert request.min_p == 0.05
        assert request.typical_p == 1.0
        assert request.stream is False
        assert request.stop is None
        assert request.max_tokens is None
        assert request.presence_penalty == 0.0
        assert request.frequency_penalty == 0.0
        assert request.repeat_penalty == 1.1
        assert request.tfs_z == 1.0
        assert request.mirostat_mode == 0
        assert request.mirostat_tau == 5.0
        assert request.mirostat_eta == 0.1
        assert request.grammar is None

    def test_all_fields(self):
        """Test setting all fields."""
        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=messages,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            min_p=0.1,
            typical_p=0.95,
            stream=True,
            stop=["END", "STOP"],
            max_tokens=100,
            presence_penalty=0.5,
            frequency_penalty=0.5,
            repeat_penalty=1.2,
            tfs_z=0.9,
            mirostat_mode=2,
            mirostat_tau=4.0,
            mirostat_eta=0.2,
            grammar="root ::= 'hello'",
        )

        assert request.model == "gpt-4"
        assert request.temperature == 0.5
        assert request.stream is True
        assert request.stop == ["END", "STOP"]
        assert request.max_tokens == 100
        assert request.grammar == "root ::= 'hello'"

    def test_multiple_messages(self):
        """Test request with conversation history."""
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
            ChatMessage(role="user", content="How are you?"),
        ]
        request = ChatCompletionRequest(messages=messages)

        assert len(request.messages) == 4
        assert request.messages[0].role == "system"
        assert request.messages[-1].role == "user"


class TestUsageInfo:
    """Tests for UsageInfo model."""

    def test_required_fields(self):
        """Test that all fields are required."""
        with pytest.raises(ValidationError):
            UsageInfo()

        with pytest.raises(ValidationError):
            UsageInfo(prompt_tokens=10)

    def test_valid_usage(self):
        """Test creating valid usage info."""
        usage = UsageInfo(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30


class TestChatCompletionResponseChoice:
    """Tests for ChatCompletionResponseChoice model."""

    def test_required_fields(self):
        """Test required fields."""
        message = ChatMessage(role="assistant", content="Hello")
        choice = ChatCompletionResponseChoice(index=0, message=message)

        assert choice.index == 0
        assert choice.message == message
        assert choice.finish_reason is None

    def test_with_finish_reason(self):
        """Test with finish reason."""
        message = ChatMessage(role="assistant", content="Hello")
        choice = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason="stop",
        )

        assert choice.finish_reason == "stop"


class TestChatCompletionResponse:
    """Tests for ChatCompletionResponse model."""

    def test_auto_generated_id(self):
        """Test that ID is auto-generated."""
        message = ChatMessage(role="assistant", content="Hello")
        choice = ChatCompletionResponseChoice(index=0, message=message)
        usage = UsageInfo(prompt_tokens=5, completion_tokens=1, total_tokens=6)

        response = ChatCompletionResponse(
            model="test",
            choices=[choice],
            usage=usage,
        )

        assert response.id.startswith("chatcmpl-")
        assert len(response.id) > 10

    def test_auto_generated_created(self):
        """Test that created timestamp is auto-generated."""
        message = ChatMessage(role="assistant", content="Hello")
        choice = ChatCompletionResponseChoice(index=0, message=message)
        usage = UsageInfo(prompt_tokens=5, completion_tokens=1, total_tokens=6)

        before = int(time.time())
        response = ChatCompletionResponse(
            model="test",
            choices=[choice],
            usage=usage,
        )
        after = int(time.time())

        assert before <= response.created <= after

    def test_default_object_type(self):
        """Test that object type defaults correctly."""
        message = ChatMessage(role="assistant", content="Hello")
        choice = ChatCompletionResponseChoice(index=0, message=message)
        usage = UsageInfo(prompt_tokens=5, completion_tokens=1, total_tokens=6)

        response = ChatCompletionResponse(
            model="test",
            choices=[choice],
            usage=usage,
        )

        assert response.object == "chat.completion"

    def test_unique_ids(self):
        """Test that each response gets a unique ID."""
        message = ChatMessage(role="assistant", content="Hello")
        choice = ChatCompletionResponseChoice(index=0, message=message)
        usage = UsageInfo(prompt_tokens=5, completion_tokens=1, total_tokens=6)

        responses = [
            ChatCompletionResponse(model="test", choices=[choice], usage=usage)
            for _ in range(10)
        ]

        ids = [r.id for r in responses]
        assert len(ids) == len(set(ids))  # All unique


class TestDeltaMessage:
    """Tests for DeltaMessage model."""

    def test_all_fields_optional(self):
        """Test that all fields are optional."""
        delta = DeltaMessage()
        assert delta.role is None
        assert delta.content is None

    def test_with_role(self):
        """Test delta with role only."""
        delta = DeltaMessage(role="assistant")
        assert delta.role == "assistant"
        assert delta.content is None

    def test_with_content(self):
        """Test delta with content only."""
        delta = DeltaMessage(content="Hello")
        assert delta.role is None
        assert delta.content == "Hello"

    def test_with_both(self):
        """Test delta with both fields."""
        delta = DeltaMessage(role="assistant", content="Hi")
        assert delta.role == "assistant"
        assert delta.content == "Hi"


class TestChatCompletionResponseStreamChoice:
    """Tests for ChatCompletionResponseStreamChoice model."""

    def test_required_fields(self):
        """Test required fields."""
        delta = DeltaMessage(content="Hello")
        choice = ChatCompletionResponseStreamChoice(index=0, delta=delta)

        assert choice.index == 0
        assert choice.delta == delta
        assert choice.finish_reason is None

    def test_with_finish_reason(self):
        """Test with finish reason."""
        delta = DeltaMessage()
        choice = ChatCompletionResponseStreamChoice(
            index=0,
            delta=delta,
            finish_reason="stop",
        )

        assert choice.finish_reason == "stop"


class TestChatCompletionStreamResponse:
    """Tests for ChatCompletionStreamResponse model."""

    def test_auto_generated_fields(self):
        """Test auto-generated id and created."""
        delta = DeltaMessage(content="Hi")
        choice = ChatCompletionResponseStreamChoice(index=0, delta=delta)

        response = ChatCompletionStreamResponse(
            model="test",
            choices=[choice],
        )

        assert response.id.startswith("chatcmpl-")
        assert response.created > 0

    def test_stream_object_type(self):
        """Test that object type is chat.completion.chunk."""
        delta = DeltaMessage(content="Hi")
        choice = ChatCompletionResponseStreamChoice(index=0, delta=delta)

        response = ChatCompletionStreamResponse(
            model="test",
            choices=[choice],
        )

        assert response.object == "chat.completion.chunk"

    def test_optional_usage(self):
        """Test that usage is optional."""
        delta = DeltaMessage(content="Hi")
        choice = ChatCompletionResponseStreamChoice(index=0, delta=delta)

        response = ChatCompletionStreamResponse(
            model="test",
            choices=[choice],
        )

        assert response.usage is None

    def test_with_usage(self):
        """Test stream response with usage info."""
        delta = DeltaMessage()
        choice = ChatCompletionResponseStreamChoice(
            index=0, delta=delta, finish_reason="stop"
        )
        usage = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        response = ChatCompletionStreamResponse(
            model="test",
            choices=[choice],
            usage=usage,
        )

        assert response.usage.total_tokens == 15
