"""Tests for the health endpoint."""

import pytest
from fastapi.testclient import TestClient

from cachehost.api.common_routes import create_common_router


@pytest.fixture
def healthy_app_state(mock_backend):
    """Create app state with a healthy worker and backend."""
    mock_worker = type("MockWorker", (), {"is_alive": lambda self: True})()

    return {
        "worker": mock_worker,
        "backend": mock_backend,
        "config": None,
    }


@pytest.fixture
def unhealthy_app_state():
    """Create app state with no worker or backend."""
    return {
        "worker": None,
        "backend": None,
        "config": None,
    }


@pytest.fixture
def degraded_app_state(mock_backend):
    """Create app state with dead worker."""
    mock_worker = type("MockWorker", (), {"is_alive": lambda self: False})()

    return {
        "worker": mock_worker,
        "backend": mock_backend,
        "config": None,
    }


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_healthy_status(self, healthy_app_state):
        """Test healthy status when worker and backend are available."""
        from fastapi import FastAPI

        app = FastAPI()
        router = create_common_router(healthy_app_state)
        app.include_router(router)

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["model_name"] == "mock_model"
        assert data["backend"] == "mock"

    def test_degraded_status_no_worker(self, unhealthy_app_state):
        """Test degraded status when worker is not available."""
        from fastapi import FastAPI

        app = FastAPI()
        router = create_common_router(unhealthy_app_state)
        app.include_router(router)

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
        assert "detail" in data

    def test_degraded_status_dead_worker(self, degraded_app_state):
        """Test degraded status when worker is not alive."""
        from fastapi import FastAPI

        app = FastAPI()
        router = create_common_router(degraded_app_state)
        app.include_router(router)

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False

    def test_degraded_status_no_backend(self):
        """Test degraded status when backend is None."""
        from fastapi import FastAPI

        mock_worker = type("MockWorker", (), {"is_alive": lambda self: True})()
        app_state = {
            "worker": mock_worker,
            "backend": None,
            "config": None,
        }

        app = FastAPI()
        router = create_common_router(app_state)
        app.include_router(router)

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
