"""LLM backend implementations."""

import platform
from typing import TYPE_CHECKING

from .protocol import LLMBackend, GenerationParams, GenerationResult, GenerationChunk

if TYPE_CHECKING:
    from ..config import Config

__all__ = [
    "LLMBackend",
    "GenerationParams",
    "GenerationResult",
    "GenerationChunk",
    "get_backend",
    "AVAILABLE_BACKENDS",
]

AVAILABLE_BACKENDS: list[str] = []

try:
    import llama_cpp  # noqa: F401
    AVAILABLE_BACKENDS.append("llama_cpp")
except ImportError:
    pass

if platform.system() == "Darwin":
    try:
        import mlx.core  # noqa: F401
        AVAILABLE_BACKENDS.append("mlx")
    except ImportError:
        pass


def get_backend(config: "Config") -> LLMBackend:
    """Factory function to get the appropriate backend based on config."""
    backend_name = config.backend

    if backend_name == "auto":
        # Prefer MLX on Mac if available, otherwise llama_cpp
        if "mlx" in AVAILABLE_BACKENDS:
            backend_name = "mlx"
        elif "llama_cpp" in AVAILABLE_BACKENDS:
            backend_name = "llama_cpp"
        else:
            raise RuntimeError(
                "No LLM backend is installed. Install one with:\n"
                "  pip install cachehost[llama-cpu]    # CPU\n"
                "  pip install cachehost[mlx]          # Apple Silicon"
            )

    if backend_name == "llama_cpp":
        if "llama_cpp" not in AVAILABLE_BACKENDS:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it with:\n"
                "  pip install cachehost[llama-cpu]"
            )
        from .llama_cpp import LlamaCppBackend
        return LlamaCppBackend(config)
    elif backend_name == "mlx":
        if "mlx" not in AVAILABLE_BACKENDS:
            raise RuntimeError("MLX backend requested but mlx is not installed or not on macOS")
        from .mlx_backend import MLXBackend
        return MLXBackend(config)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")
