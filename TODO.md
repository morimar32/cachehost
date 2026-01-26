# TODO

## Completed (v0.1.0 Refactoring)

- [x] Modularize codebase (separate files for models, cache, worker, API)
- [x] Add LLM backend abstraction (Protocol-based)
- [x] Add MLX backend support (Mac only) - stub implementation
- [x] Add Anthropic API compatibility (--api-format flag)
- [x] Replace print() with Python logging
- [x] Fix cache key to include message roles
- [x] Add configurable timeout (--timeout flag)
- [x] Update CLAUDE.md documentation

## Testing

- [ ] Add unit tests for cache key generation (CacheManager._compute_cache_key)
- [ ] Add unit tests for Pydantic models (request/response validation)
- [ ] Add integration tests for the API endpoints
- [ ] Set up pytest and test configuration
- [ ] Test MLX backend on Mac hardware (state save/load limitations)
- [ ] End-to-end testing with real model

## Future Features

- [ ] SQLite audit logging (timestamp, message hash, raw input/output)
- [ ] Investigate MLX KV-cache state management for proper caching support

*(To be discussed)*
