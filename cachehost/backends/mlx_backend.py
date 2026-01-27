"""MLX backend implementation for macOS."""

from typing import Any, Dict, Iterator, List, Optional

from ..config import Config
from ..logging_config import get_logger
from ..schemas.common import ChatMessage
from .protocol import GenerationChunk, GenerationParams, GenerationResult

logger = get_logger("backend.mlx")

# Marker key for MLX prompt cache states (matches cache/manager.py)
MLX_CACHE_MARKER = "_mlx_prompt_cache"


class MLXBackend:
    """
    LLM backend using MLX for Apple Silicon Macs.

    Supports full KV-cache state management via mlx_lm's prompt caching API.
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
        self._prompt_cache = None

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
            from mlx_lm.models.cache import make_prompt_cache
        except ImportError:
            raise RuntimeError(
                "mlx-lm is not installed. Install with: pip install mlx-lm"
            )

        logger.debug(f"Loading MLX model from: {self.config.model_path}")

        # MLX expects a directory path with the model files
        self._model, self._tokenizer = load(self.config.model_path)

        # Initialize prompt cache for KV-cache management
        self._prompt_cache = make_prompt_cache(self._model)

        logger.debug("MLX model loaded successfully with prompt cache initialized")

    def reset(self) -> None:
        """Reset the model state by creating a fresh prompt cache."""
        if self._model is None:
            self._prompt_cache = None
            return

        try:
            from mlx_lm.models.cache import make_prompt_cache

            self._prompt_cache = make_prompt_cache(self._model)
            logger.debug("MLX prompt cache reset")
        except ImportError:
            logger.warning("mlx-lm not available for cache reset")
            self._prompt_cache = None

    def save_state(self) -> Any:
        """
        Save and return the current KV-cache state.

        Returns a dict with MLX marker and prompt cache.
        The CacheManager will handle saving to safetensors format.
        """
        if self._prompt_cache is None:
            logger.warning("No prompt cache to save")
            return None

        return {
            MLX_CACHE_MARKER: True,
            "cache": self._prompt_cache,
        }

    def load_state(self, state: Any) -> None:
        """
        Load a previously saved KV-cache state.

        Args:
            state: Dict containing MLX marker and cache
        """
        if state is None:
            self._prompt_cache = None
            return

        if not isinstance(state, dict) or not state.get(MLX_CACHE_MARKER):
            logger.warning("Invalid MLX state format")
            return

        # Extract prompt cache from loaded state
        loaded_cache = state.get("cache")
        if loaded_cache is not None:
            self._prompt_cache = loaded_cache
            logger.debug("Loaded MLX prompt cache from state")
        else:
            logger.warning("No cache found in loaded state")

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

    def _build_generation_kwargs(self, params: GenerationParams) -> Dict[str, Any]:
        """
        Build generation kwargs from GenerationParams.

        MLX-lm's generate_step() accepts a `sampler` callable and
        `logits_processors` list rather than raw sampling parameter values.
        This method constructs those using mlx_lm.sample_utils.

        Args:
            params: Generation parameters

        Returns:
            Dict of kwargs for mlx_lm.generate/stream_generate
        """
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        # Build sampler from params
        sampler_kwargs = {
            "temp": params.temperature,
            "top_p": params.top_p,
        }
        if params.top_k > 0:
            sampler_kwargs["top_k"] = params.top_k
        if params.min_p > 0:
            sampler_kwargs["min_p"] = params.min_p

        kwargs: Dict[str, Any] = {
            "sampler": make_sampler(**sampler_kwargs),
        }

        # Build logits processors for repetition penalty
        if params.repeat_penalty != 1.0:
            kwargs["logits_processors"] = make_logits_processors(
                repetition_penalty=params.repeat_penalty,
            )

        if params.max_tokens:
            kwargs["max_tokens"] = params.max_tokens

        return kwargs

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.

        Args:
            text: The text to tokenize

        Returns:
            Number of tokens
        """
        if self._tokenizer is None:
            return 0

        try:
            tokens = self._tokenizer.encode(text)
            return len(tokens)
        except Exception:
            return 0

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
        gen_kwargs = self._build_generation_kwargs(params)

        # Add prompt cache if available
        if self._prompt_cache is not None:
            gen_kwargs["prompt_cache"] = self._prompt_cache

        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            **gen_kwargs,
        )

        # Count tokens
        prompt_tokens = self._count_tokens(prompt)
        completion_tokens = self._count_tokens(response)
        total_tokens = prompt_tokens + completion_tokens

        return GenerationResult(
            content=response,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
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
        gen_kwargs = self._build_generation_kwargs(params)

        # Add prompt cache if available
        if self._prompt_cache is not None:
            gen_kwargs["prompt_cache"] = self._prompt_cache

        # Count prompt tokens upfront
        prompt_tokens = self._count_tokens(prompt)

        # First chunk with role
        yield GenerationChunk(content="", role="assistant", prompt_tokens=prompt_tokens)

        completion_tokens = 0
        accumulated_text = ""

        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            **gen_kwargs,
        ):
            # stream_generate can yield strings or GenerationResponse objects
            if isinstance(response, str):
                token_text = response
            else:
                # GenerationResponse object has text attribute
                token_text = getattr(response, "text", str(response))

            accumulated_text += token_text
            completion_tokens += 1  # Approximate: 1 token per yield

            yield GenerationChunk(content=token_text)

        # Final chunk with token counts
        total_tokens = prompt_tokens + completion_tokens
        yield GenerationChunk(
            content="",
            finish_reason="stop",
            is_final=True,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def shutdown(self) -> None:
        """Clean up resources."""
        logger.debug("Shutting down MLX backend")
        self._model = None
        self._tokenizer = None
        self._prompt_cache = None
