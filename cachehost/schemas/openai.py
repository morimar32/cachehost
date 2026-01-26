"""OpenAI-compatible API schema definitions."""

import time
import uuid
from typing import List, Optional

from pydantic import BaseModel, Field

from .common import ChatMessage


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    typical_p: float = 1.0
    stream: bool = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repeat_penalty: float = 1.1
    tfs_z: float = 1.0
    mirostat_mode: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    grammar: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    """A single choice in a chat completion response."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    """Delta message for streaming responses."""
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    """A single choice in a streaming chat completion response."""
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming chat completion response chunk."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None
