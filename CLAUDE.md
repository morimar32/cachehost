# CLAUDE.md

## Project Overview

CacheHost is a local LLM server that provides OpenAI and Anthropic-compatible APIs with intelligent state caching. It automatically saves and restores LLM KV-cache states to avoid reprocessing context when making iterative requests, dramatically speeding up workflows with coding agents like Roo Code, Open Code, and similar tools.

## Quick Start

```bash
# Install dependencies
pip3 install -r requirements.txt

# Run the server with a model
python3 main.py -m /path/to/your/model.gguf

# With custom options
python3 main.py -m /path/to/model.gguf --n-ctx 8192 --api-format openai --backend llama_cpp
```

The server starts on `http://0.0.0.0:8000`.

## Architecture

### Core Concept: Prefix-Based State Caching

The key innovation is caching LLM states based on conversation history prefixes:

1. When a request arrives, the worker looks for cached states matching the longest prefix of the message history
2. If found, it loads that state (skipping recomputation of that context)
3. After generation, it saves the new state (including the assistant response) to cache
4. Cache keys are SHA256 hashes of `MODEL_NAME + role:content` for each message

This means if you send messages A, B, C and then later send A, B, C, D, the server can load the cached state from after C and only process D.

### Directory Structure

```
cachehost/
├── main.py                      # Entry point - CLI parsing, uvicorn launch
├── requirements.txt             # Dependencies (optional MLX)
├── README.md                    # Brief project description
├── CLAUDE.md                    # This file
├── TODO.md                      # Task tracking
├── .gitignore                   # Python project ignores
│
├── cachehost/                   # Main package
│   ├── __init__.py
│   ├── config.py                # Configuration dataclass, CLI args
│   ├── logging_config.py        # Python logging setup
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── common.py            # Shared ChatMessage base
│   │   ├── openai.py            # OpenAI request/response models
│   │   └── anthropic.py         # Anthropic request/response models
│   │
│   ├── cache/
│   │   ├── __init__.py
│   │   └── manager.py           # CacheManager class
│   │
│   ├── backends/
│   │   ├── __init__.py          # Backend registry, factory
│   │   ├── protocol.py          # LLMBackend Protocol definition
│   │   ├── llama_cpp.py         # llama-cpp-python implementation
│   │   └── mlx_backend.py       # MLX implementation (Mac only)
│   │
│   ├── worker/
│   │   ├── __init__.py
│   │   └── worker.py            # Worker thread implementation
│   │
│   └── api/
│       ├── __init__.py
│       ├── app.py               # FastAPI app factory + lifespan
│       ├── openai_routes.py     # /v1/chat/completions
│       ├── anthropic_routes.py  # /v1/messages
│       └── common_routes.py     # /health
│
└── cache/                       # Runtime cache directory (auto-created, auto-cleaned)
    └── {model_name}/{backend_name}/
        └── {hash}.pkl           # Pickled LLM states
```

### Request Flow

```
HTTP Request → FastAPI Endpoint → REQUEST_QUEUE → LLM Worker Thread
                                                        ↓
                                              1. Cache lookup (longest prefix match)
                                              2. Load state or reset
                                              3. Generate response via Backend
                                              4. Save new state to cache
                                                        ↓
                                              Response (streaming SSE or JSON)
```

### Threading Model

- Single dedicated worker thread processes all LLM requests sequentially
- `REQUEST_QUEUE` (thread-safe) passes requests from FastAPI to worker
- Streaming uses per-request queues for chunk delivery
- Non-streaming uses `threading.Event` for completion signaling

### Backend Abstraction

The `LLMBackend` Protocol defines the interface for LLM backends:

```python
class LLMBackend(Protocol[StateT]):
    @property
    def model_name(self) -> str: ...
    @property
    def backend_name(self) -> str: ...

    def load(self) -> None: ...
    def reset(self) -> None: ...
    def save_state(self) -> StateT: ...
    def load_state(self, state: StateT) -> None: ...
    def generate(self, messages, params) -> GenerationResult: ...
    def generate_stream(self, messages, params) -> Iterator[GenerationChunk]: ...
    def shutdown(self) -> None: ...
```

Available backends:
- `llama_cpp`: Uses llama-cpp-python (default, works on all platforms)
- `mlx`: Uses MLX for Apple Silicon Macs (optional, requires mlx-lm)

## CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `-m, --model-path` | Yes | - | Path to LLM model file |
| `--n-ctx` | No | 4096 | Context window size in tokens |
| `--n-threads` | No | CPU count | Number of CPU threads |
| `--top-k` | No | 20 | Top-K sampling parameter |
| `--repeat-penalty` | No | 1.0 | Repeat penalty parameter |
| `--temperature` | No | 0.7 | Temperature sampling parameter |
| `--min-p` | No | 0.05 | Min-P sampling parameter |
| `--seed` | No | 3407 | Random seed |
| `--host` | No | 0.0.0.0 | Host address to bind to |
| `--port` | No | 8000 | Port to listen on |
| `--timeout` | No | 300 | Request timeout in seconds |
| `--backend` | No | auto | Backend: auto, llama_cpp, mlx |
| `--api-format` | No | openai | API format: openai, anthropic |
| `--log-level` | No | INFO | Log level |
| `--log-file` | No | None | Optional log file path |
| `--cache-dir` | No | ./cache | Cache directory |

## API Endpoints

### OpenAI Format (default)

#### POST /v1/chat/completions
OpenAI-compatible chat completion endpoint. Supports:
- Streaming (`stream: true`) via Server-Sent Events
- All standard sampling parameters (temperature, top_p, top_k, etc.)
- Grammar constraints (GBNF format)
- Stop sequences

### Anthropic Format (--api-format anthropic)

#### POST /v1/messages
Anthropic-compatible messages endpoint. Supports:
- Streaming via Server-Sent Events
- System messages
- Text content blocks

### Common

#### GET /health
Returns server status and model info.

## Code Style

- Type hints on all function signatures
- Python Protocol for backend abstraction (structural typing)
- Dataclasses for configuration
- Logging via Python's logging module

## Key Implementation Details

### Cache Key Generation
```python
combined_content = MODEL_NAME + "".join(f"{msg.role}:{msg.content}" for msg in history_prefix)
hash = SHA256(combined_content)
```
Note: Both role and content are included in the hash to properly distinguish different conversation structures.

### State Serialization
Uses Python's `pickle` module with backend-specific `save_state()`/`load_state()` methods.

### Cache Lifecycle
- Cleared on startup
- Cleared on shutdown (registered with `atexit`)
- States persist only during a single server session
- Cache path: `{cache_dir}/{model_name}/{backend_name}/{hash}.pkl`

### Error Handling
- Cache load failures fall back to fresh state (graceful degradation)
- Grammar parse failures proceed without grammar
- Worker exceptions are returned to client as HTTP 500

## Testing

Example request using curl (OpenAI format):
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "stream": false
  }'
```

Example request (Anthropic format):
```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "local",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Verify cache hit on follow-up:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"},
      {"role": "assistant", "content": "Hi there!"},
      {"role": "user", "content": "How are you?"}
    ],
    "stream": false
  }'
```

## Dependencies

Core:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `llama-cpp-python` - LLM inference engine (default backend)

Optional (Mac only):
- `mlx` - Apple's ML framework
- `mlx-lm` - MLX language model utilities
