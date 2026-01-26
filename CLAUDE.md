# CLAUDE.md

## Project Overview

CacheHost is a local LLM server that provides an OpenAI-compatible API with intelligent state caching. It automatically saves and restores LLM KV-cache states to avoid reprocessing context when making iterative requests, dramatically speeding up workflows with coding agents like Roo Code, Open Code, and similar tools.

## Quick Start

```bash
# Install dependencies
pip3 install -r requirements.txt

# Set required environment variable
export LLM_MODEL_PATH=/path/to/your/model.gguf

# Optional: set context size (default 4096)
export LLM_N_CTX=8192

# Run the server
python3 main.py
```

The server starts on `http://0.0.0.0:8000`.

## Architecture

### Core Concept: Prefix-Based State Caching

The key innovation is caching LLM states based on conversation history prefixes:

1. When a request arrives, the worker looks for cached states matching the longest prefix of the message history
2. If found, it loads that state (skipping recomputation of that context)
3. After generation, it saves the new state (including the assistant response) to cache
4. Cache keys are SHA256 hashes of `MODEL_NAME + message contents`

This means if you send messages A, B, C and then later send A, B, C, D, the server can load the cached state from after C and only process D.

### Components

```
main.py (single file, ~470 lines)
├── Configuration (lines 21-33)
├── Pydantic Models - OpenAI-compatible schemas (lines 40-98)
├── Cache Management (lines 102-125)
├── LLM Worker Thread (lines 128-268)
├── FastAPI Application (lines 271-458)
└── Entry Point (lines 460-468)
```

### Request Flow

```
HTTP Request → FastAPI Endpoint → REQUEST_QUEUE → LLM Worker Thread
                                                        ↓
                                              1. Cache lookup (longest prefix match)
                                              2. Load state or reset
                                              3. Generate response
                                              4. Save new state to cache
                                                        ↓
                                              Response (streaming SSE or JSON)
```

### Threading Model

- Single dedicated worker thread processes all LLM requests sequentially
- `REQUEST_QUEUE` (thread-safe) passes requests from FastAPI to worker
- Streaming uses per-request queues for chunk delivery
- Non-streaming uses `threading.Event` for completion signaling

## API Endpoints

### POST /v1/chat/completions
OpenAI-compatible chat completion endpoint. Supports:
- Streaming (`stream: true`) via Server-Sent Events
- All standard sampling parameters (temperature, top_p, top_k, etc.)
- Grammar constraints (GBNF format)
- Stop sequences

### GET /health
Returns server status and model info.

## File Structure

```
cachehost/
├── main.py          # All application code
├── README.md        # Brief project description
├── CLAUDE.md        # This file
├── TODO.md          # Task tracking
├── .gitignore       # Python project ignores
└── cache/           # Runtime cache directory (auto-created, auto-cleaned)
    └── {model_name}/
        └── {hash}.pkl   # Pickled LLM states
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_MODEL_PATH` | Yes | - | Path to GGUF model file |
| `LLM_N_CTX` | No | 4096 | Context window size in tokens |

Thread count is auto-detected from CPU count.

## Code Style

- Type hints are preferred on all function signatures
- No formatter currently configured (open to adding one)
- Single-file architecture (modularization planned)

## Key Implementation Details

### Cache Key Generation (line 102)
```python
combined_content = MODEL_NAME + "".join(msg.content for msg in history_prefix)
hash = SHA256(combined_content)
```
Note: Only message content is hashed, not roles. This means cache keys don't distinguish between user/assistant for the same text.

### State Serialization
Uses Python's `pickle` module with `llama-cpp-python`'s `save_state()`/`load_state()` methods.

### Cache Lifecycle
- Cleared on startup (`init_cache_directory`)
- Cleared on shutdown (registered with `atexit`)
- States persist only during a single server session

### Error Handling
- Cache load failures fall back to fresh state (graceful degradation)
- Grammar parse failures proceed without grammar
- Worker exceptions are returned to client as HTTP 500

## Testing

Example request using curl:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "stream": false
  }'
```

Example model for testing: `glm4.7-flash`

## Dependencies

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `llama-cpp-python` - LLM inference engine (wraps llama.cpp)
