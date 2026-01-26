#!/usr/bin/env python3
"""CacheHost - Local LLM server with intelligent KV-cache state management.

This is the entry point for the CacheHost server. It parses command-line
arguments, creates the FastAPI application, and starts the uvicorn server.
"""

import uvicorn

from cachehost.config import parse_args
from cachehost.api import create_app


def main() -> None:
    """Main entry point."""
    # Parse command line arguments
    config = parse_args()

    # Create FastAPI application
    app = create_app(config)

    # Run server
    print(f"Starting CacheHost server...")
    print(f"  Model: {config.model_name}")
    print(f"  Backend: {config.backend}")
    print(f"  API format: {config.api_format}")
    print(f"  Address: http://{config.host}:{config.port}")

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
