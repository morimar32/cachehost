"""Configuration management for CacheHost."""

import argparse
import os
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class Config:
    """Configuration for CacheHost server."""

    # Model settings
    model_path: str
    n_ctx: int = 4096
    n_threads: int = field(default_factory=lambda: os.cpu_count() or 4)

    # Sampling defaults
    top_k: int = 20
    repeat_penalty: float = 1.0
    temperature: float = 0.7
    min_p: float = 0.05
    seed: int = 3407

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    timeout: int = 300

    # Backend and API format
    backend: Literal["auto", "llama_cpp", "mlx"] = "auto"
    api_format: Literal["openai", "anthropic"] = "openai"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Cache settings
    cache_dir: str = "./cache"

    @property
    def model_name(self) -> str:
        """Extract model name from path for caching."""
        return os.path.splitext(os.path.basename(self.model_path))[0].replace(".", "_")


def parse_args(args: Optional[list] = None) -> Config:
    """
    Parse command line arguments and return Config.

    Args:
        args: Optional list of arguments (for testing). If None, uses sys.argv.

    Returns:
        Config instance with parsed values
    """
    parser = argparse.ArgumentParser(
        description="CacheHost - Local LLM server with intelligent KV-cache state management",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "-m", "--model-path",
        required=True,
        help="Path to the LLM model file (GGUF format for llama_cpp backend)"
    )

    # Model settings
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=4096,
        help="Context window size in tokens"
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of CPU threads to use"
    )

    # Sampling parameters
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K sampling parameter"
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.0,
        help="Repeat penalty parameter"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature sampling parameter"
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.05,
        help="Min-P sampling parameter"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for reproducibility"
    )

    # Server settings
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds"
    )

    # Backend and API format
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "llama_cpp", "mlx"],
        default="auto",
        help="LLM backend to use (auto selects best available)"
    )
    parser.add_argument(
        "--api-format",
        type=str,
        choices=["openai", "anthropic"],
        default="openai",
        help="API format to serve"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path"
    )

    # Cache settings
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Directory for cache storage"
    )

    parsed = parser.parse_args(args)

    # Validate model path exists
    if not os.path.exists(parsed.model_path):
        parser.error(f"Model file not found at: {parsed.model_path}")

    return Config(
        model_path=parsed.model_path,
        n_ctx=parsed.n_ctx,
        n_threads=parsed.n_threads,
        top_k=parsed.top_k,
        repeat_penalty=parsed.repeat_penalty,
        temperature=parsed.temperature,
        min_p=parsed.min_p,
        seed=parsed.seed,
        host=parsed.host,
        port=parsed.port,
        timeout=parsed.timeout,
        backend=parsed.backend,
        api_format=parsed.api_format,
        log_level=parsed.log_level,
        log_file=parsed.log_file,
        cache_dir=parsed.cache_dir,
    )
