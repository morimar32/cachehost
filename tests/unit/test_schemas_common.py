"""Tests for common schema definitions."""

import pytest
from pydantic import ValidationError

from cachehost.schemas.common import (
    ChatMessage,
    ImageContent,
    MultiModalChatMessage,
    TextContent,
    normalize_message,
)


class TestChatMessage:
    """Tests for ChatMessage model."""

    def test_required_fields(self):
        """Test that role and content are required."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_missing_role_raises(self):
        """Test that missing role raises validation error."""
        with pytest.raises(ValidationError):
            ChatMessage(content="Hello")

    def test_missing_content_raises(self):
        """Test that missing content raises validation error."""
        with pytest.raises(ValidationError):
            ChatMessage(role="user")

    def test_any_role_value_accepted(self):
        """Test that any string role is accepted."""
        for role in ["user", "assistant", "system", "function", "tool"]:
            msg = ChatMessage(role=role, content="test")
            assert msg.role == role

    def test_empty_content_allowed(self):
        """Test that empty content is allowed."""
        msg = ChatMessage(role="user", content="")
        assert msg.content == ""

    def test_multiline_content(self):
        """Test that multiline content is preserved."""
        content = "Line 1\nLine 2\nLine 3"
        msg = ChatMessage(role="user", content=content)
        assert msg.content == content

    def test_unicode_content(self):
        """Test that unicode content is handled correctly."""
        content = "Hello! 你好! مرحبا! 🎉"
        msg = ChatMessage(role="user", content=content)
        assert msg.content == content


class TestTextContent:
    """Tests for TextContent model."""

    def test_default_type(self):
        """Test that type defaults to 'text'."""
        tc = TextContent(text="Hello")
        assert tc.type == "text"

    def test_text_required(self):
        """Test that text field is required."""
        with pytest.raises(ValidationError):
            TextContent()

    def test_explicit_type(self):
        """Test that type can be set explicitly."""
        tc = TextContent(type="text", text="Hello")
        assert tc.type == "text"


class TestImageContent:
    """Tests for ImageContent model."""

    def test_default_type(self):
        """Test that type defaults to 'image'."""
        ic = ImageContent(source={"type": "base64", "data": "..."})
        assert ic.type == "image"

    def test_source_required(self):
        """Test that source is required."""
        with pytest.raises(ValidationError):
            ImageContent()

    def test_accepts_source_dict(self):
        """Test that source accepts a dictionary."""
        source = {
            "type": "base64",
            "media_type": "image/png",
            "data": "iVBORw0KGgo...",
        }
        ic = ImageContent(source=source)
        assert ic.source == source


class TestMultiModalChatMessage:
    """Tests for MultiModalChatMessage model."""

    def test_string_content(self):
        """Test with string content."""
        msg = MultiModalChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_list_content_with_text(self):
        """Test with list content containing text blocks."""
        content = [TextContent(text="Hello"), TextContent(text="World")]
        msg = MultiModalChatMessage(role="user", content=content)
        assert len(msg.content) == 2

    def test_list_content_with_image(self):
        """Test with list content containing image blocks."""
        content = [
            TextContent(text="Look at this:"),
            ImageContent(source={"type": "base64", "data": "..."}),
        ]
        msg = MultiModalChatMessage(role="user", content=content)
        assert len(msg.content) == 2


class TestNormalizeMessage:
    """Tests for normalize_message function."""

    def test_passthrough_chat_message(self):
        """Test that ChatMessage is returned as-is."""
        original = ChatMessage(role="user", content="Hello")
        result = normalize_message(original)
        assert result is original

    def test_dict_to_chat_message(self):
        """Test converting dict to ChatMessage."""
        msg_dict = {"role": "user", "content": "Hello"}
        result = normalize_message(msg_dict)

        assert isinstance(result, ChatMessage)
        assert result.role == "user"
        assert result.content == "Hello"

    def test_dict_default_role(self):
        """Test that missing role defaults to 'user'."""
        msg_dict = {"content": "Hello"}
        result = normalize_message(msg_dict)
        assert result.role == "user"

    def test_dict_default_content(self):
        """Test that missing content defaults to empty string."""
        msg_dict = {"role": "user"}
        result = normalize_message(msg_dict)
        assert result.content == ""

    def test_multimodal_string_content(self):
        """Test MultiModalChatMessage with string content."""
        msg = MultiModalChatMessage(role="user", content="Hello")
        result = normalize_message(msg)

        assert isinstance(result, ChatMessage)
        assert result.role == "user"
        assert result.content == "Hello"

    def test_multimodal_extracts_text_blocks(self):
        """Test that text is extracted from multimodal content."""
        msg = MultiModalChatMessage(
            role="user",
            content=[
                TextContent(text="Part 1"),
                TextContent(text="Part 2"),
            ],
        )
        result = normalize_message(msg)

        assert isinstance(result, ChatMessage)
        assert result.content == "Part 1\nPart 2"

    def test_multimodal_ignores_images(self):
        """Test that images are ignored in normalization."""
        msg = MultiModalChatMessage(
            role="user",
            content=[
                TextContent(text="Check this:"),
                ImageContent(source={"type": "base64", "data": "..."}),
                TextContent(text="What do you think?"),
            ],
        )
        result = normalize_message(msg)
        assert result.content == "Check this:\nWhat do you think?"

    def test_dict_with_list_content(self):
        """Test dict with multimodal list content."""
        msg_dict = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
        }
        result = normalize_message(msg_dict)

        assert result.content == "Hello\nWorld"

    def test_dict_multimodal_ignores_images(self):
        """Test that dict multimodal content ignores images."""
        msg_dict = {
            "role": "user",
            "content": [
                {"type": "text", "text": "See this:"},
                {"type": "image", "source": {"data": "..."}},
            ],
        }
        result = normalize_message(msg_dict)
        assert result.content == "See this:"

    def test_unknown_type_raises(self):
        """Test that unknown message types raise ValueError."""
        with pytest.raises(ValueError, match="Unknown message type"):
            normalize_message(12345)

    def test_preserves_role(self):
        """Test that role is preserved through normalization."""
        for role in ["user", "assistant", "system"]:
            msg = MultiModalChatMessage(role=role, content="test")
            result = normalize_message(msg)
            assert result.role == role
