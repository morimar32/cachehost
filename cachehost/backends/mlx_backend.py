"""MLX backend implementation for macOS."""

from typing import Any, Iterator, List, Optional

from ..config import Config
from ..logging_config import get_logger
from ..schemas.common import ChatMessage
from .protocol import GenerationChunk, GenerationParams, GenerationResult

logger = get_logger("backend.mlx")


class MLXBackend:
    """
    LLM backend using MLX for Apple Silicon Macs.

    Note: MLX state management is currently limited. This implementation
    provides the basic generation interface but may not support full
    KV-cache state save/restore like llama-cpp-python.
    """

    def __init__(self, config: Config):
        """
        Initialize the backend with configuration.

        Args:
            config: Server configuration
        """
        self.config = config
        self._model = None
        self._tokenizer = None
        self._model_name = config.model_name
        self._state: Optional[Any] = None

    @property
    def model_name(self) -> str:
        """Return the name of the loaded model."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend identifier."""
        return "mlx"

    def load(self) -> None:
        """Load the model using MLX."""
        try:
            from mlx_lm import load
        except ImportError:
            raise RuntimeError(
                "mlx-lm is not installed. Install with: pip install mlx-lm"
            )

        logger.debug(f"Loading MLX model from: {self.config.model_path}")

        # MLX expects a directory path with the model files
        self._model, self._tokenizer = load(self.config.model_path)

        logger.debug("MLX model loaded successfully")

    def reset(self) -> None:
        """Reset the model state."""
        # MLX doesn't have explicit state reset like llama-cpp
        # The KV cache is typically managed per-generation
        self._state = None
        logger.debug("MLX state reset (no-op for stateless generation)")

    def save_state(self) -> Any:
        """
        Save and return the current KV-cache state.

        Note: MLX state management is limited. This returns a placeholder
        that may not fully capture the KV cache state.
        """
        # TODO: Research MLX state management capabilities
        # For now, return None as MLX doesn't expose KV cache state
        logger.warning("MLX backend: save_state is not fully implemented")
        return self._state

    def load_state(self, state: Any) -> None:
        """
        Load a previously saved KV-cache state.

        Note: MLX state management is limited.
        """
        # TODO: Implement proper state loading when MLX supports it
        logger.warning("MLX backend: load_state is not fully implemented")
        self._state = state

    def _format_messages(self, messages: List[ChatMessage]) -> str:
        """Format chat messages into a prompt string."""
        # Use tokenizer's chat template if available
        if hasattr(self._tokenizer, "apply_chat_template"):
            message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
            return self._tokenizer.apply_chat_template(
                message_dicts, tokenize=False, add_generation_prompt=True
            )

        # Fallback: simple formatting
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def generate(
        self, messages: List[ChatMessage], params: GenerationParams
    ) -> GenerationResult:
        """Generate a complete response (non-streaming)."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")

        try:
            from mlx_lm import generate
        except ImportError:
            raise RuntimeError("mlx-lm is not installed")

        prompt = self._format_messages(messages)

        # MLX generate options
        gen_kwargs = {
            "temp": params.temperature,
            "top_p": params.top_p,
        }
        if params.max_tokens:
            gen_kwargs["max_tokens"] = params.max_tokens

        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            **gen_kwargs,
        )

        # MLX doesn't provide detailed token counts
        return GenerationResult(
            content=response,
            finish_reason="stop",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            raw_response={"text": response},
        )

    def generate_stream(
        self, messages: List[ChatMessage], params: GenerationParams
    ) -> Iterator[GenerationChunk]:
        """Generate a streaming response."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")

        try:
            from mlx_lm import stream_generate
        except ImportError:
            raise RuntimeError("mlx-lm is not installed")

        prompt = self._format_messages(messages)

        gen_kwargs = {
            "temp": params.temperature,
            "top_p": params.top_p,
        }
        if params.max_tokens:
            gen_kwargs["max_tokens"] = params.max_tokens

        # First chunk with role
        yield GenerationChunk(content="", role="assistant")

        for token_text in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            **gen_kwargs,
        ):
            yield GenerationChunk(content=token_text)

        # Final chunk
        yield GenerationChunk(content="", finish_reason="stop", is_final=True)

    def shutdown(self) -> None:
        """Clean up resources."""
        logger.debug("Shutting down MLX backend")
        self._model = None
        self._tokenizer = None
