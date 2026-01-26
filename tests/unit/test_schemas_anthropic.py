"""Tests for Anthropic schema definitions."""

import pytest
from pydantic import ValidationError

from cachehost.schemas.anthropic import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicResponse,
    AnthropicUsage,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    ErrorEvent,
    ImageBlock,
    ImageSource,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    PingEvent,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)


class TestTextBlock:
    """Tests for TextBlock model."""

    def test_type_literal(self):
        """Test that type is literally 'text'."""
        block = TextBlock(text="Hello")
        assert block.type == "text"

    def test_text_required(self):
        """Test that text field is required."""
        with pytest.raises(ValidationError):
            TextBlock()

    def test_empty_text_allowed(self):
        """Test that empty text is allowed."""
        block = TextBlock(text="")
        assert block.text == ""


class TestImageSource:
    """Tests for ImageSource model."""

    def test_type_literal(self):
        """Test that type is literally 'base64'."""
        source = ImageSource(media_type="image/png", data="abc123")
        assert source.type == "base64"

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            ImageSource(media_type="image/png")

        with pytest.raises(ValidationError):
            ImageSource(data="abc123")

    def test_valid_source(self):
        """Test valid image source."""
        source = ImageSource(media_type="image/jpeg", data="base64data...")
        assert source.media_type == "image/jpeg"
        assert source.data == "base64data..."


class TestImageBlock:
    """Tests for ImageBlock model."""

    def test_type_literal(self):
        """Test that type is literally 'image'."""
        source = ImageSource(media_type="image/png", data="...")
        block = ImageBlock(source=source)
        assert block.type == "image"

    def test_source_required(self):
        """Test that source is required."""
        with pytest.raises(ValidationError):
            ImageBlock()


class TestToolUseBlock:
    """Tests for ToolUseBlock model."""

    def test_type_literal(self):
        """Test that type is literally 'tool_use'."""
        block = ToolUseBlock(id="tool_1", name="get_weather", input={"city": "NYC"})
        assert block.type == "tool_use"

    def test_required_fields(self):
        """Test all required fields."""
        with pytest.raises(ValidationError):
            ToolUseBlock(id="tool_1", name="test")  # missing input

        with pytest.raises(ValidationError):
            ToolUseBlock(id="tool_1", input={})  # missing name

    def test_valid_tool_use(self):
        """Test valid tool use block."""
        block = ToolUseBlock(
            id="toolu_123",
            name="calculate",
            input={"expression": "2+2"},
        )
        assert block.id == "toolu_123"
        assert block.name == "calculate"
        assert block.input == {"expression": "2+2"}


class TestToolResultBlock:
    """Tests for ToolResultBlock model."""

    def test_type_literal(self):
        """Test that type is literally 'tool_result'."""
        block = ToolResultBlock(tool_use_id="tool_1", content="Result")
        assert block.type == "tool_result"

    def test_string_content(self):
        """Test with string content."""
        block = ToolResultBlock(tool_use_id="tool_1", content="The result is 4")
        assert block.content == "The result is 4"

    def test_list_content(self):
        """Test with list content."""
        block = ToolResultBlock(
            tool_use_id="tool_1",
            content=[TextBlock(text="Part 1"), TextBlock(text="Part 2")],
        )
        assert len(block.content) == 2

    def test_is_error_default(self):
        """Test that is_error defaults to False."""
        block = ToolResultBlock(tool_use_id="tool_1", content="OK")
        assert block.is_error is False

    def test_is_error_true(self):
        """Test setting is_error to True."""
        block = ToolResultBlock(
            tool_use_id="tool_1",
            content="Error occurred",
            is_error=True,
        )
        assert block.is_error is True


class TestAnthropicMessage:
    """Tests for AnthropicMessage model."""

    def test_role_literal_user(self):
        """Test that role accepts 'user'."""
        msg = AnthropicMessage(role="user", content="Hello")
        assert msg.role == "user"

    def test_role_literal_assistant(self):
        """Test that role accepts 'assistant'."""
        msg = AnthropicMessage(role="assistant", content="Hi")
        assert msg.role == "assistant"

    def test_invalid_role_rejected(self):
        """Test that invalid roles are rejected."""
        with pytest.raises(ValidationError):
            AnthropicMessage(role="system", content="Hello")

        with pytest.raises(ValidationError):
            AnthropicMessage(role="function", content="Hello")

    def test_string_content(self):
        """Test with string content."""
        msg = AnthropicMessage(role="user", content="Hello")
        assert msg.content == "Hello"

    def test_list_content(self):
        """Test with list content blocks."""
        msg = AnthropicMessage(
            role="user",
            content=[TextBlock(text="Hello")],
        )
        assert len(msg.content) == 1


class TestAnthropicRequest:
    """Tests for AnthropicRequest model."""

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            AnthropicRequest(model="claude-3", messages=[])  # missing max_tokens

    def test_minimal_request(self):
        """Test minimal valid request."""
        messages = [AnthropicMessage(role="user", content="Hello")]
        request = AnthropicRequest(
            model="claude-3-sonnet",
            messages=messages,
            max_tokens=100,
        )

        assert request.model == "claude-3-sonnet"
        assert request.max_tokens == 100
        assert len(request.messages) == 1

    def test_default_values(self):
        """Test default values for optional fields."""
        messages = [AnthropicMessage(role="user", content="Hello")]
        request = AnthropicRequest(
            model="claude-3-sonnet",
            messages=messages,
            max_tokens=100,
        )

        assert request.system is None
        assert request.temperature is None
        assert request.top_p is None
        assert request.top_k is None
        assert request.stop_sequences is None
        assert request.stream is False
        assert request.tools is None
        assert request.tool_choice is None
        assert request.metadata is None

    def test_all_fields(self):
        """Test setting all fields."""
        messages = [AnthropicMessage(role="user", content="Hello")]
        tools = [
            ToolDefinition(
                name="get_time",
                description="Get current time",
                input_schema={"type": "object", "properties": {}},
            )
        ]

        request = AnthropicRequest(
            model="claude-3-opus",
            messages=messages,
            max_tokens=500,
            system="You are helpful.",
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            stop_sequences=["END"],
            stream=True,
            tools=tools,
            tool_choice={"type": "auto"},
            metadata={"user_id": "123"},
        )

        assert request.system == "You are helpful."
        assert request.temperature == 0.5
        assert request.stream is True
        assert len(request.tools) == 1


class TestAnthropicUsage:
    """Tests for AnthropicUsage model."""

    def test_required_fields(self):
        """Test that both fields are required."""
        with pytest.raises(ValidationError):
            AnthropicUsage(input_tokens=10)

    def test_valid_usage(self):
        """Test valid usage."""
        usage = AnthropicUsage(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50


class TestAnthropicResponse:
    """Tests for AnthropicResponse model."""

    def test_auto_generated_id(self):
        """Test that ID is auto-generated."""
        response = AnthropicResponse(
            content=[TextBlock(text="Hello")],
            model="claude-3",
            usage=AnthropicUsage(input_tokens=10, output_tokens=5),
        )

        assert response.id.startswith("msg_")

    def test_default_type(self):
        """Test that type defaults to 'message'."""
        response = AnthropicResponse(
            content=[TextBlock(text="Hello")],
            model="claude-3",
            usage=AnthropicUsage(input_tokens=10, output_tokens=5),
        )

        assert response.type == "message"

    def test_default_role(self):
        """Test that role defaults to 'assistant'."""
        response = AnthropicResponse(
            content=[TextBlock(text="Hello")],
            model="claude-3",
            usage=AnthropicUsage(input_tokens=10, output_tokens=5),
        )

        assert response.role == "assistant"

    def test_stop_reason_literal(self):
        """Test that stop_reason accepts valid literals."""
        for reason in ["end_turn", "max_tokens", "stop_sequence", "tool_use"]:
            response = AnthropicResponse(
                content=[TextBlock(text="Hi")],
                model="claude-3",
                stop_reason=reason,
                usage=AnthropicUsage(input_tokens=10, output_tokens=5),
            )
            assert response.stop_reason == reason


class TestStreamingEvents:
    """Tests for streaming event types."""

    def test_message_start_event(self):
        """Test MessageStartEvent."""
        message = AnthropicResponse(
            content=[],
            model="claude-3",
            usage=AnthropicUsage(input_tokens=10, output_tokens=0),
        )
        event = MessageStartEvent(message=message)

        assert event.type == "message_start"
        assert event.message.model == "claude-3"

    def test_content_block_start_event(self):
        """Test ContentBlockStartEvent."""
        event = ContentBlockStartEvent(
            index=0,
            content_block=TextBlock(text=""),
        )

        assert event.type == "content_block_start"
        assert event.index == 0

    def test_content_block_delta_event(self):
        """Test ContentBlockDeltaEvent."""
        event = ContentBlockDeltaEvent(
            index=0,
            delta={"type": "text_delta", "text": "Hello"},
        )

        assert event.type == "content_block_delta"
        assert event.delta["text"] == "Hello"

    def test_content_block_stop_event(self):
        """Test ContentBlockStopEvent."""
        event = ContentBlockStopEvent(index=0)

        assert event.type == "content_block_stop"
        assert event.index == 0

    def test_message_delta_event(self):
        """Test MessageDeltaEvent."""
        event = MessageDeltaEvent(
            delta={"stop_reason": "end_turn"},
            usage={"output_tokens": 50},
        )

        assert event.type == "message_delta"
        assert event.delta["stop_reason"] == "end_turn"

    def test_message_stop_event(self):
        """Test MessageStopEvent."""
        event = MessageStopEvent()
        assert event.type == "message_stop"

    def test_ping_event(self):
        """Test PingEvent."""
        event = PingEvent()
        assert event.type == "ping"

    def test_error_event(self):
        """Test ErrorEvent."""
        event = ErrorEvent(
            error={"type": "server_error", "message": "Something went wrong"}
        )

        assert event.type == "error"
        assert event.error["type"] == "server_error"


class TestToolDefinition:
    """Tests for ToolDefinition model."""

    def test_required_fields(self):
        """Test all fields are required."""
        with pytest.raises(ValidationError):
            ToolDefinition(name="test", description="A test tool")

    def test_valid_definition(self):
        """Test valid tool definition."""
        tool = ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        )

        assert tool.name == "get_weather"
        assert "location" in tool.input_schema["properties"]
