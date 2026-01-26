"""Common schema definitions shared across API formats."""

from typing import List, Optional, Union
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    role: str
    content: str


class ImageContent(BaseModel):
    """Image content for vision models."""
    type: str = "image"
    source: dict  # Contains type, media_type, data for base64 images


class TextContent(BaseModel):
    """Text content block."""
    type: str = "text"
    text: str


class MultiModalChatMessage(BaseModel):
    """A message that can contain text and/or images."""
    role: str
    content: Union[str, List[Union[TextContent, ImageContent]]]


def normalize_message(msg: Union[ChatMessage, MultiModalChatMessage, dict]) -> ChatMessage:
    """
    Normalize various message formats to a simple ChatMessage.

    For multimodal messages with images, extracts only the text content.
    """
    if isinstance(msg, dict):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Handle multimodal content
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                elif hasattr(block, "type") and block.type == "text":
                    text_parts.append(block.text)
            content = "\n".join(text_parts)

        return ChatMessage(role=role, content=content)

    if isinstance(msg, ChatMessage):
        return msg

    if isinstance(msg, MultiModalChatMessage):
        if isinstance(msg.content, str):
            return ChatMessage(role=msg.role, content=msg.content)
        else:
            text_parts = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
            return ChatMessage(role=msg.role, content="\n".join(text_parts))

    raise ValueError(f"Unknown message type: {type(msg)}")
