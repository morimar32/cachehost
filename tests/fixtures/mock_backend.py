"""Mock backend for testing without real LLM models."""

from typing import Any, Iterator, List, Optional

from cachehost.schemas.common import ChatMessage
from cachehost.backends.protocol import (
    GenerationChunk,
    GenerationParams,
    GenerationResult,
)


class MockBackend:
    """
    Mock LLM backend implementing the LLMBackend protocol.

    This backend returns predictable responses for testing without
    requiring a real model.
    """

    def __init__(
        self,
        model_name: str = "mock_model",
        backend_name: str = "mock",
        default_response: str = "This is a mock response.",
    ):
        """
        Initialize the mock backend.

        Args:
            model_name: Name to report as model name
            backend_name: Name to report as backend name
            default_response: Default response content
        """
        self._model_name = model_name
        self._backend_name = backend_name
        self._default_response = default_response
        self._state: Optional[Any] = None
        self._loaded = False
        self._reset_count = 0
        self._generate_count = 0
        self._stream_generate_count = 0
        self._load_state_count = 0
        self._save_state_count = 0

        # Custom response handler for testing specific scenarios
        self._custom_response_handler = None
        self._custom_stream_handler = None

        # For simulating errors
        self._should_fail_generate = False
        self._should_fail_load_state = False
        self._generate_error_message = "Simulated generation error"

    @property
    def model_name(self) -> str:
        """Return the name of the loaded model."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend identifier."""
        return self._backend_name

    def load(self) -> None:
        """Load the model (no-op for mock)."""
        self._loaded = True

    def reset(self) -> None:
        """Reset the backend state."""
        self._state = None
        self._reset_count += 1

    def save_state(self) -> Any:
        """Save and return the current state."""
        self._save_state_count += 1
        return {"mock_state": True, "data": self._state}

    def load_state(self, state: Any) -> None:
        """Load a previously saved state."""
        self._load_state_count += 1
        if self._should_fail_load_state:
            raise RuntimeError("Simulated load_state error")
        self._state = state.get("data") if isinstance(state, dict) else state

    def generate(
        self, messages: List[ChatMessage], params: GenerationParams
    ) -> GenerationResult:
        """Generate a complete response."""
        self._generate_count += 1

        if self._should_fail_generate:
            raise RuntimeError(self._generate_error_message)

        if self._custom_response_handler:
            return self._custom_response_handler(messages, params)

        # Generate predictable response based on last message
        last_message = messages[-1] if messages else None
        content = self._default_response

        if last_message:
            if "joke" in last_message.content.lower():
                content = "Why don't scientists trust atoms? Because they make up everything!"
            elif "hello" in last_message.content.lower():
                content = "Hello! How can I help you today?"

        # Simulate token counts
        prompt_tokens = sum(len(m.content.split()) for m in messages)
        completion_tokens = len(content.split())

        return GenerationResult(
            content=content,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            raw_response={"mock": True},
        )

    def generate_stream(
        self, messages: List[ChatMessage], params: GenerationParams
    ) -> Iterator[GenerationChunk]:
        """Generate a streaming response."""
        self._stream_generate_count += 1

        if self._should_fail_generate:
            raise RuntimeError(self._generate_error_message)

        if self._custom_stream_handler:
            yield from self._custom_stream_handler(messages, params)
            return

        # First chunk with role
        yield GenerationChunk(content="", role="assistant")

        # Generate content
        result = self.generate(messages, params)
        words = result.content.split()

        # Stream word by word
        for i, word in enumerate(words):
            chunk_content = word if i == 0 else f" {word}"
            yield GenerationChunk(content=chunk_content)

        # Final chunk with usage info
        yield GenerationChunk(
            content="",
            finish_reason="stop",
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            is_final=True,
        )

    def shutdown(self) -> None:
        """Clean up resources."""
        self._loaded = False
        self._state = None

    # Test helper methods

    def set_custom_response(self, response: str) -> None:
        """Set a custom response for generate calls."""
        self._default_response = response

    def set_custom_response_handler(self, handler) -> None:
        """Set a custom handler for generate calls."""
        self._custom_response_handler = handler

    def set_custom_stream_handler(self, handler) -> None:
        """Set a custom handler for streaming calls."""
        self._custom_stream_handler = handler

    def set_should_fail(self, should_fail: bool, message: str = None) -> None:
        """Configure the backend to fail on generate calls."""
        self._should_fail_generate = should_fail
        if message:
            self._generate_error_message = message

    def set_should_fail_load_state(self, should_fail: bool) -> None:
        """Configure the backend to fail on load_state calls."""
        self._should_fail_load_state = should_fail

    def get_call_counts(self) -> dict:
        """Get counts of various method calls for assertions."""
        return {
            "reset": self._reset_count,
            "generate": self._generate_count,
            "stream_generate": self._stream_generate_count,
            "load_state": self._load_state_count,
            "save_state": self._save_state_count,
        }
