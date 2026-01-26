"""Common API routes (health, etc.)."""

from fastapi import APIRouter

router = APIRouter()


def create_common_router(app_state: dict) -> APIRouter:
    """
    Create common routes router.

    Args:
        app_state: Application state dict containing worker, backend, etc.

    Returns:
        Configured APIRouter
    """
    router = APIRouter()

    @router.get("/health")
    async def health_check():
        """Health check endpoint."""
        worker = app_state.get("worker")
        backend = app_state.get("backend")

        if worker and worker.is_alive() and backend:
            return {
                "status": "ok",
                "model_loaded": True,
                "model_name": backend.model_name,
                "backend": backend.backend_name,
            }
        return {
            "status": "degraded",
            "model_loaded": False,
            "detail": "LLM instance or worker not available.",
        }

    return router
