"""Protocol definition for LLM backends."""

from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, List, Optional, Protocol, TypeVar

from ..schemas.common import ChatMessage

StateT = TypeVar("StateT")


@dataclass
class GenerationParams:
    """Parameters for text generation."""
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    typical_p: float = 1.0
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


@dataclass
class GenerationResult:
    """Result of non-streaming generation."""
    content: str
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw_response: Optional[dict] = None


@dataclass
class GenerationChunk:
    """A single chunk in streaming generation."""
    content: str = ""
    role: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    is_final: bool = False


class LLMBackend(Protocol[StateT]):
    """
    Protocol defining the interface for LLM backends.

    This uses structural typing (Protocol) rather than ABC for more Pythonic
    duck typing and easier testing with mocks.
    """

    @property
    def model_name(self) -> str:
        """Return the name of the loaded model."""
        ...

    @property
    def backend_name(self) -> str:
        """Return the name of this backend (e.g., 'llama_cpp', 'mlx')."""
        ...

    def load(self) -> None:
        """Load the model. Called during server startup."""
        ...

    def reset(self) -> None:
        """Reset the backend state to initial state."""
        ...

    def save_state(self) -> StateT:
        """Save and return the current KV-cache state."""
        ...

    def load_state(self, state: StateT) -> None:
        """Load a previously saved KV-cache state."""
        ...

    def generate(
        self, messages: List[ChatMessage], params: GenerationParams
    ) -> GenerationResult:
        """
        Generate a complete response (non-streaming).

        Args:
            messages: List of chat messages
            params: Generation parameters

        Returns:
            GenerationResult with the complete response
        """
        ...

    def generate_stream(
        self, messages: List[ChatMessage], params: GenerationParams
    ) -> Iterator[GenerationChunk]:
        """
        Generate a streaming response.

        Args:
            messages: List of chat messages
            params: Generation parameters

        Yields:
            GenerationChunk objects for each token/chunk
        """
        ...

    def shutdown(self) -> None:
        """Clean up resources. Called during server shutdown."""
        ...
