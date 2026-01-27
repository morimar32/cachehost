"""FastAPI application factory."""

import queue
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from ..backends import get_backend
from ..cache.manager import CacheManager
from ..config import Config
from ..logging_config import get_logger, setup_logging
from ..worker.worker import LLMWorker
from .anthropic_routes import create_anthropic_router
from .common_routes import create_common_router
from .openai_routes import create_openai_router

logger = get_logger("app")


def create_app(config: Config) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        config: Server configuration

    Returns:
        Configured FastAPI application
    """
    # Setup logging
    setup_logging(level=config.log_level, log_file=config.log_file)

    # Application state - will be populated during lifespan
    app_state: dict = {
        "config": config,
        "backend": None,
        "cache_manager": None,
        "worker": None,
        "request_queue": None,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Manage application lifecycle."""
        logger.debug("Server startup...")

        # Initialize backend
        logger.debug(f"Initializing {config.backend} backend...")
        backend = get_backend(config)
        backend.load()
        app_state["backend"] = backend

        # Initialize cache manager
        cache_manager = CacheManager(
            cache_dir=config.cache_dir,
            model_name=config.model_name,
            backend_name=backend.backend_name,
        )
        cache_manager.init()
        app_state["cache_manager"] = cache_manager

        # Initialize request queue and worker
        request_queue: queue.Queue = queue.Queue()
        app_state["request_queue"] = request_queue

        worker = LLMWorker(
            backend=backend,
            cache_manager=cache_manager,
            request_queue=request_queue,
        )
        worker.start()
        app_state["worker"] = worker

        logger.debug(f"Server startup complete. API format: {config.api_format}")
        logger.debug(f"Model: {backend.model_name}, Backend: {backend.backend_name}")

        yield

        # Shutdown
        logger.debug("Server shutdown initiated...")

        if worker:
            worker.stop()

        if backend:
            backend.shutdown()

        # Cache cleanup is handled by atexit in CacheManager
        logger.debug("Server shutdown complete.")

    # Create FastAPI app with lifespan
    app = FastAPI(
        title="CacheHost - LLM Server with KV Cache",
        description="Local LLM server with intelligent state caching",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Include common routes
    app.include_router(create_common_router(app_state))

    # Include API-format specific routes
    if config.api_format == "openai":
        app.include_router(create_openai_router(app_state))
        logger.debug("Registered OpenAI-compatible routes")
    elif config.api_format == "anthropic":
        app.include_router(create_anthropic_router(app_state))
        logger.debug("Registered Anthropic-compatible routes")

    return app
