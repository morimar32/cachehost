"""Tests for Anthropic-compatible API endpoints."""

import json
import queue
import threading
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cachehost.api.anthropic_routes import create_anthropic_router


@pytest.fixture
def mock_worker():
    """Create a mock worker."""

    class MockLLMWorker:
        def __init__(self):
            self._alive = True

        def is_alive(self):
            return self._alive

    return MockLLMWorker()


@pytest.fixture
def anthropic_app(mock_backend, mock_worker):
    """Create FastAPI app with Anthropic routes and mock processing."""
    app_state = {
        "worker": mock_worker,
        "backend": mock_backend,
        "config": MagicMock(timeout=10),
        "request_queue": queue.Queue(),
    }

    app = FastAPI()
    router = create_anthropic_router(app_state)
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


class TestAnthropicModels:
    """Tests for GET /models and /v1/models endpoints."""

    def test_list_models_v1_endpoint(self, anthropic_app):
        """Test GET /v1/models returns model list."""
        app, _ = anthropic_app
        client = TestClient(app)

        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "object" in data
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        assert "id" in data["data"][0]
        assert "object" in data["data"][0]
        assert data["data"][0]["object"] == "model"

    def test_list_models_no_prefix_endpoint(self, anthropic_app):
        """Test GET /models returns model list (no /v1 prefix)."""
        app, _ = anthropic_app
        client = TestClient(app)

        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["object"] == "list"


class TestAnthropicMessages:
    """Tests for POST /v1/messages endpoint."""

    def test_non_streaming_success(self, anthropic_app):
        """Test successful non-streaming message."""
        app, _ = anthropic_app
        client = TestClient(app)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-sonnet",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 100,
            },
            headers={"anthropic-version": "2023-06-01"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert "content" in data
        assert len(data["content"]) > 0
        assert data["content"][0]["type"] == "text"
        assert "usage" in data

    def test_non_streaming_no_prefix(self, anthropic_app):
        """Test successful message at /messages (no /v1 prefix)."""
        app, _ = anthropic_app
        client = TestClient(app)

        response = client.post(
            "/messages",
            json={
                "model": "claude-3-sonnet",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 100,
            },
            headers={"anthropic-version": "2023-06-01"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"

    def test_streaming_success(self, anthropic_app):
        """Test successful streaming message."""
        app, _ = anthropic_app
        client = TestClient(app)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-sonnet",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 100,
                "stream": True,
            },
            headers={"anthropic-version": "2023-06-01"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        # Collect events
        events = []
        for line in response.iter_lines():
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                events.append({"type": event_type, "data": data})

        # Should have message_start
        event_types = [e["type"] for e in events]
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "message_stop" in event_types

    def test_missing_max_tokens_uses_default(self, anthropic_app):
        """Test that missing max_tokens uses default value."""
        app, _ = anthropic_app
        client = TestClient(app)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-sonnet",
                "messages": [{"role": "user", "content": "Hello!"}],
                # max_tokens defaults to 4096
            },
            headers={"anthropic-version": "2023-06-01"},
        )

        assert response.status_code == 200

    def test_missing_model_uses_default(self, anthropic_app):
        """Test that missing model uses default value."""
        app, _ = anthropic_app
        client = TestClient(app)

        response = client.post(
            "/v1/messages",
            json={
                "messages": [{"role": "user", "content": "Hello!"}],
                # model defaults to "local"
            },
            headers={"anthropic-version": "2023-06-01"},
        )

        assert response.status_code == 200

    def test_worker_not_running_returns_503(self, mock_backend):
        """Test that unavailable worker returns 503."""
        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = False

        app_state = {
            "worker": mock_worker,
            "backend": mock_backend,
            "config": MagicMock(timeout=10),
            "request_queue": queue.Queue(),
        }

        app = FastAPI()
        router = create_anthropic_router(app_state)
        app.include_router(router)

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-sonnet",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 100,
            },
            headers={"anthropic-version": "2023-06-01"},
        )

        assert response.status_code == 503

    def test_system_message_handling(self, anthropic_app):
        """Test that system message is handled correctly."""
        app, _ = anthropic_app
        client = TestClient(app)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-sonnet",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 100,
                "system": "You are a helpful assistant.",
            },
            headers={"anthropic-version": "2023-06-01"},
        )

        assert response.status_code == 200


class TestAnthropicResponseFormat:
    """Tests for Anthropic response format compliance."""

    def test_response_has_required_fields(self, anthropic_app):
        """Test that response has all required Anthropic fields."""
        app, _ = anthropic_app
        client = TestClient(app)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-sonnet",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
            },
            headers={"anthropic-version": "2023-06-01"},
        )

        data = response.json()

        # Required fields per Anthropic spec
        assert "id" in data
        assert data["id"].startswith("msg_")
        assert "type" in data
        assert data["type"] == "message"
        assert "role" in data
        assert data["role"] == "assistant"
        assert "content" in data
        assert "model" in data
        assert "usage" in data

    def test_content_block_format(self, anthropic_app):
        """Test that content blocks have correct format."""
        app, _ = anthropic_app
        client = TestClient(app)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-sonnet",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
            },
            headers={"anthropic-version": "2023-06-01"},
        )

        content = response.json()["content"]

        assert isinstance(content, list)
        assert len(content) > 0
        assert content[0]["type"] == "text"
        assert "text" in content[0]

    def test_usage_format(self, anthropic_app):
        """Test that usage has correct Anthropic format."""
        app, _ = anthropic_app
        client = TestClient(app)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-sonnet",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
            },
            headers={"anthropic-version": "2023-06-01"},
        )

        usage = response.json()["usage"]

        # Anthropic uses input_tokens/output_tokens
        assert "input_tokens" in usage
        assert "output_tokens" in usage
        assert isinstance(usage["input_tokens"], int)
        assert isinstance(usage["output_tokens"], int)

    def test_stream_event_format(self, anthropic_app):
        """Test that streaming events have correct format."""
        app, _ = anthropic_app
        client = TestClient(app)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-sonnet",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
                "stream": True,
            },
            headers={"anthropic-version": "2023-06-01"},
        )

        # Parse events
        events = {}
        current_event = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                current_event = line[7:]
            elif line.startswith("data: ") and current_event:
                data = json.loads(line[6:])
                events[current_event] = data

        # Check message_start event
        if "message_start" in events:
            assert "message" in events["message_start"]
            assert events["message_start"]["type"] == "message_start"

        # Check content_block_start event
        if "content_block_start" in events:
            assert events["content_block_start"]["type"] == "content_block_start"
            assert "content_block" in events["content_block_start"]


class TestAnthropicMessageRoles:
    """Tests for Anthropic message role handling."""

    def test_user_role_accepted(self, anthropic_app):
        """Test that user role is accepted."""
        app, _ = anthropic_app
        client = TestClient(app)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-sonnet",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            },
        )

        assert response.status_code == 200

    def test_assistant_role_accepted(self, anthropic_app):
        """Test that assistant role is accepted in history."""
        app, _ = anthropic_app
        client = TestClient(app)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-sonnet",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                ],
                "max_tokens": 100,
            },
        )

        assert response.status_code == 200

    def test_invalid_role_rejected(self, anthropic_app):
        """Test that invalid roles are rejected."""
        app, _ = anthropic_app
        client = TestClient(app)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-sonnet",
                "messages": [{"role": "system", "content": "You are helpful"}],
                "max_tokens": 100,
            },
        )

        # Anthropic doesn't allow system role in messages (use system field)
        assert response.status_code == 422
