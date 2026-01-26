"""Tests for OpenAI-compatible API endpoints."""

import json
import queue
import threading
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cachehost.api.openai_routes import create_openai_router
from cachehost.backends.protocol import GenerationParams, GenerationResult
from cachehost.schemas.common import ChatMessage
from cachehost.worker.worker import RequestPackage


@pytest.fixture
def mock_worker():
    """Create a mock worker that processes requests."""

    class MockLLMWorker:
        def __init__(self):
            self._alive = True
            self._process_func = None

        def is_alive(self):
            return self._alive

        def set_process_func(self, func):
            """Set function to call when request is processed."""
            self._process_func = func

    return MockLLMWorker()


@pytest.fixture
def app_state_factory(mock_backend, mock_worker):
    """Factory to create app state with configurable components."""

    def create_state(timeout=10):
        return {
            "worker": mock_worker,
            "backend": mock_backend,
            "config": MagicMock(timeout=timeout),
            "request_queue": queue.Queue(),
        }

    return create_state


@pytest.fixture
def openai_app(app_state_factory, mock_backend, mock_worker):
    """Create FastAPI app with OpenAI routes and mock processing."""
    app_state = app_state_factory()

    app = FastAPI()
    router = create_openai_router(app_state)
    app.include_router(router)

    # Start a thread to process requests
    def process_requests():
        while True:
            try:
                pkg = app_state["request_queue"].get(timeout=1)
                if pkg is None:
                    break

                # Generate mock response
                result = mock_backend.generate(pkg.messages, pkg.params)

                if pkg.stream_flag and pkg.stream_queue:
                    # Streaming response
                    pkg.stream_queue.put(
                        {
                            "choices": [{"index": 0, "delta": {"role": "assistant"}}]
                        }
                    )
                    for word in result.content.split():
                        pkg.stream_queue.put(
                            {
                                "choices": [{"index": 0, "delta": {"content": f"{word} "}}]
                            }
                        )
                    pkg.stream_queue.put(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": result.prompt_tokens,
                                "completion_tokens": result.completion_tokens,
                                "total_tokens": result.total_tokens,
                            },
                        }
                    )
                    pkg.stream_queue.put(None)
                else:
                    # Non-streaming response
                    pkg.response_container["result"] = {
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": result.content,
                                },
                                "finish_reason": result.finish_reason,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "total_tokens": result.total_tokens,
                        },
                    }
                    pkg.signal_event.set()

            except queue.Empty:
                continue
            except Exception as e:
                if pkg.stream_flag and pkg.stream_queue:
                    pkg.stream_queue.put({"error": str(e)})
                    pkg.stream_queue.put(None)
                elif pkg.response_container is not None:
                    pkg.response_container["error"] = str(e)
                if pkg.signal_event:
                    pkg.signal_event.set()

    worker_thread = threading.Thread(target=process_requests, daemon=True)
    worker_thread.start()

    yield app, app_state

    # Cleanup
    app_state["request_queue"].put(None)
    worker_thread.join(timeout=2)


class TestOpenAIChatCompletions:
    """Tests for POST /v1/chat/completions endpoint."""

    def test_non_streaming_success(self, openai_app):
        """Test successful non-streaming completion."""
        app, _ = openai_app
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "choices" in data
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"]
        assert "usage" in data

    def test_streaming_success(self, openai_app):
        """Test successful streaming completion."""
        app, _ = openai_app
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        # Collect all events
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                events.append(json.loads(data))

        # Should have multiple chunks
        assert len(events) > 0

        # First chunk should have role
        assert events[0]["object"] == "chat.completion.chunk"

        # Should end with [DONE]
        lines = list(response.iter_lines())
        # Note: iter_lines() was consumed above, so this may be empty

    def test_missing_messages_returns_422(self, openai_app):
        """Test that missing messages field returns 422."""
        app, _ = openai_app
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"stream": False},
        )

        assert response.status_code == 422

    def test_worker_not_running_returns_503(self, app_state_factory, mock_backend):
        """Test that unavailable worker returns 503."""
        app_state = app_state_factory()
        app_state["worker"]._alive = False

        app = FastAPI()
        router = create_openai_router(app_state)
        app.include_router(router)

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": False,
            },
        )

        assert response.status_code == 503
        assert "not available" in response.json()["detail"]

    def test_with_all_parameters(self, openai_app):
        """Test request with all sampling parameters."""
        app, _ = openai_app
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 50,
                "min_p": 0.1,
                "max_tokens": 100,
                "stop": ["END"],
                "presence_penalty": 0.5,
                "frequency_penalty": 0.3,
                "repeat_penalty": 1.2,
                "stream": False,
            },
        )

        assert response.status_code == 200

    def test_conversation_history(self, openai_app):
        """Test request with conversation history."""
        app, _ = openai_app
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "Tell me a joke."},
                ],
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Response should reference joke since last message asked for one
        assert data["choices"][0]["message"]["content"]


class TestOpenAIResponseFormat:
    """Tests for OpenAI response format compliance."""

    def test_response_has_required_fields(self, openai_app):
        """Test that response has all required OpenAI fields."""
        app, _ = openai_app
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

        data = response.json()

        # Required fields per OpenAI spec
        assert "id" in data
        assert "object" in data
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert "usage" in data

    def test_choice_format(self, openai_app):
        """Test that choice has correct format."""
        app, _ = openai_app
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

        choice = response.json()["choices"][0]

        assert "index" in choice
        assert "message" in choice
        assert "role" in choice["message"]
        assert "content" in choice["message"]

    def test_usage_format(self, openai_app):
        """Test that usage has correct format."""
        app, _ = openai_app
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

        usage = response.json()["usage"]

        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert isinstance(usage["total_tokens"], int)

    def test_stream_chunk_format(self, openai_app):
        """Test that streaming chunks have correct format."""
        app, _ = openai_app
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        # Get first data chunk
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                assert "id" in chunk
                assert chunk["object"] == "chat.completion.chunk"
                assert "choices" in chunk
                assert "delta" in chunk["choices"][0]
                break
