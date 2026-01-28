"""CacheHost - Local LLM server with intelligent KV-cache state management.

Enables running via `python -m cachehost` and the `cachehost` console script.
"""

import uvicorn

from cachehost.config import parse_args
from cachehost.api import create_app


def main() -> None:
    """Main entry point."""
    config = parse_args()

    app = create_app(config)

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
