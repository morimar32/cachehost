"""Anthropic API schema definitions."""

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# Content block types
class TextBlock(BaseModel):
    """Text content block."""
    type: Literal["text"] = "text"
    text: str


class ImageSource(BaseModel):
    """Source data for an image."""
    type: Literal["base64"] = "base64"
    media_type: str  # e.g., "image/jpeg", "image/png"
    data: str  # Base64-encoded image data


class ImageBlock(BaseModel):
    """Image content block."""
    type: Literal["image"] = "image"
    source: ImageSource


class ToolUseBlock(BaseModel):
    """Tool use content block (assistant response)."""
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class ToolResultBlock(BaseModel):
    """Tool result content block (user message)."""
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[Union[TextBlock, ImageBlock]]]
    is_error: bool = False


ContentBlock = Union[TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock]


class AnthropicMessage(BaseModel):
    """A message in the Anthropic format."""
    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]


class ToolDefinition(BaseModel):
    """Definition of a tool available to the model."""
    name: str
    description: str
    input_schema: Dict[str, Any]


class AnthropicRequest(BaseModel):
    """Anthropic messages API request."""
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int
    system: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Dict[str, Any]] = None  # {"type": "auto"}, {"type": "any"}, {"type": "tool", "name": "..."}
    metadata: Optional[Dict[str, Any]] = None


class AnthropicUsage(BaseModel):
    """Token usage information for Anthropic API."""
    input_tokens: int
    output_tokens: int


class AnthropicResponse(BaseModel):
    """Anthropic messages API response."""
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[ContentBlock]
    model: str
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage


# Streaming event types
class MessageStartEvent(BaseModel):
    """Event sent at the start of a message stream."""
    type: Literal["message_start"] = "message_start"
    message: AnthropicResponse


class ContentBlockStartEvent(BaseModel):
    """Event sent at the start of a content block."""
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: ContentBlock


class ContentBlockDeltaEvent(BaseModel):
    """Event sent for content block updates."""
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: Dict[str, Any]  # {"type": "text_delta", "text": "..."} or {"type": "input_json_delta", "partial_json": "..."}


class ContentBlockStopEvent(BaseModel):
    """Event sent at the end of a content block."""
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaEvent(BaseModel):
    """Event sent for message-level updates."""
    type: Literal["message_delta"] = "message_delta"
    delta: Dict[str, Any]  # {"stop_reason": "...", "stop_sequence": ...}
    usage: Dict[str, int]  # {"output_tokens": ...}


class MessageStopEvent(BaseModel):
    """Event sent at the end of a message stream."""
    type: Literal["message_stop"] = "message_stop"


class PingEvent(BaseModel):
    """Keepalive ping event."""
    type: Literal["ping"] = "ping"


class ErrorEvent(BaseModel):
    """Error event in stream."""
    type: Literal["error"] = "error"
    error: Dict[str, Any]  # {"type": "...", "message": "..."}


StreamEvent = Union[
    MessageStartEvent,
    ContentBlockStartEvent,
    ContentBlockDeltaEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStopEvent,
    PingEvent,
    ErrorEvent,
]
