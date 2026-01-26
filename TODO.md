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

## Completed (Test Suite)

- [x] Set up pytest and test configuration (pytest.ini, conftest.py)
- [x] Create MockBackend for testing without real models
- [x] Add unit tests for CacheManager (20 tests)
  - Cache key generation (_compute_cache_key)
  - State save/load with pickle serialization
  - Longest prefix matching
  - Cache cleanup
- [x] Add unit tests for Config and CLI parsing (22 tests)
- [x] Add unit tests for Pydantic schemas (91 tests total)
  - Common schemas (ChatMessage, normalize_message)
  - OpenAI schemas (request/response validation)
  - Anthropic schemas (message types, streaming events)
- [x] Add unit tests for backend protocol dataclasses (18 tests)
- [x] Add unit tests for MLX backend (18 tests, macOS-only)
- [x] Add integration tests for API endpoints (27 tests)
  - Health endpoint
  - OpenAI chat completions (streaming and non-streaming)
  - Anthropic messages (streaming and non-streaming)

## Testing - Remaining

- [ ] Test MLX backend on Mac hardware (state save/load limitations)
- [ ] End-to-end testing with real model

## Future Features

- [ ] SQLite audit logging (timestamp, message hash, raw input/output)
- [ ] Investigate MLX KV-cache state management for proper caching support

*(To be discussed)*
