"""Anthropic API routes."""

import asyncio
import json
import queue
import threading
import time
import uuid
from typing import Any, Dict, Generator, List

from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse

from ..backends.protocol import GenerationParams
from ..logging_config import get_logger
from ..schemas.anthropic import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicResponse,
    AnthropicUsage,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    TextBlock,
)
from ..schemas.common import ChatMessage
from ..worker.worker import RequestPackage

logger = get_logger("api.anthropic")


def _convert_anthropic_messages(
    messages: List[AnthropicMessage], system: str | None
) -> List[ChatMessage]:
    """Convert Anthropic messages to internal ChatMessage format."""
    result: List[ChatMessage] = []

    # Add system message if present
    if system:
        result.append(ChatMessage(role="system", content=system))

    for msg in messages:
        if isinstance(msg.content, str):
            result.append(ChatMessage(role=msg.role, content=msg.content))
        else:
            # Extract text content from content blocks
            text_parts = []
            for block in msg.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
                elif hasattr(block, "content") and isinstance(block.content, str):
                    # Tool result with string content
                    text_parts.append(block.content)
            result.append(ChatMessage(role=msg.role, content="\n".join(text_parts)))

    return result


def create_anthropic_router(app_state: dict) -> APIRouter:
    """
    Create Anthropic-compatible routes router.

    Args:
        app_state: Application state dict containing worker, backend, config, etc.

    Returns:
        Configured APIRouter
    """
    router = APIRouter()

    def get_model_name() -> str:
        """Get model name from backend."""
        backend = app_state.get("backend")
        return backend.model_name if backend else "unknown"

    async def stream_generator(
        req_id: str, response_queue: queue.Queue, model_name: str, request: AnthropicRequest
    ) -> Generator[str, None, None]:
        """Generate SSE stream in Anthropic format."""
        logger.debug(f"[{req_id}] Anthropic stream generator started")

        msg_id = f"msg_{uuid.uuid4().hex}"
        content_parts: List[str] = []
        output_tokens = 0

        try:
            # Send message_start event
            start_message = AnthropicResponse(
                id=msg_id,
                content=[],
                model=model_name,
                stop_reason=None,
                usage=AnthropicUsage(input_tokens=0, output_tokens=0),
            )
            start_event = MessageStartEvent(message=start_message)
            yield f"event: message_start\ndata: {json.dumps(start_event.dict())}\n\n"

            # Send content_block_start event
            block_start = ContentBlockStartEvent(
                index=0, content_block=TextBlock(text="")
            )
            yield f"event: content_block_start\ndata: {json.dumps(block_start.dict())}\n\n"

            while True:
                chunk_data = response_queue.get()

                if chunk_data is None:
                    logger.debug(f"[{req_id}] Stream end signal received")
                    break

                # Handle errors
                if isinstance(chunk_data, dict) and "error" in chunk_data:
                    logger.error(f"[{req_id}] Stream error: {chunk_data['error']}")
                    error_event = {
                        "type": "error",
                        "error": {
                            "type": "server_error",
                            "message": chunk_data["error"],
                        },
                    }
                    yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
                    break

                # Extract content from chunk
                if isinstance(chunk_data, dict) and "choices" in chunk_data:
                    delta = chunk_data["choices"][0].get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        content_parts.append(content)
                        output_tokens += 1  # Approximate

                        delta_event = ContentBlockDeltaEvent(
                            index=0,
                            delta={"type": "text_delta", "text": content},
                        )
                        yield f"event: content_block_delta\ndata: {json.dumps(delta_event.dict())}\n\n"

                await asyncio.sleep(0.001)

            # Send content_block_stop event
            block_stop = ContentBlockStopEvent(index=0)
            yield f"event: content_block_stop\ndata: {json.dumps(block_stop.dict())}\n\n"

            # Send message_delta event
            msg_delta = MessageDeltaEvent(
                delta={"stop_reason": "end_turn"},
                usage={"output_tokens": output_tokens},
            )
            yield f"event: message_delta\ndata: {json.dumps(msg_delta.dict())}\n\n"

            # Send message_stop event
            msg_stop = MessageStopEvent()
            yield f"event: message_stop\ndata: {json.dumps(msg_stop.dict())}\n\n"

        except Exception as e:
            logger.error(f"[{req_id}] Exception in stream_generator: {e}")
            error_event = {
                "type": "error",
                "error": {"type": "server_error", "message": str(e)},
            }
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

        finally:
            logger.debug(f"[{req_id}] Anthropic stream generator finished")

    @router.post("/v1/messages", response_model=None)
    async def create_message(
        request: AnthropicRequest,
        x_api_key: str | None = Header(None, alias="x-api-key"),
        anthropic_version: str | None = Header(None, alias="anthropic-version"),
    ):
        """Anthropic messages API endpoint."""
        req_id = f"req-{uuid.uuid4().hex[:8]}"
        logger.info(f"[{req_id}] Received Anthropic request. Stream: {request.stream}")

        worker = app_state.get("worker")
        config = app_state.get("config")

        if not worker or not worker.is_alive():
            raise HTTPException(
                status_code=503, detail="LLM service not available or worker not running."
            )

        # Convert Anthropic messages to internal format
        messages = _convert_anthropic_messages(request.messages, request.system)

        # Build generation params
        params = GenerationParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.95,
            top_k=request.top_k or 40,
            stop=request.stop_sequences,
        )

        request_package = RequestPackage(
            id=req_id,
            messages=messages,
            stream_flag=request.stream,
            params=params,
        )

        request_queue = app_state.get("request_queue")
        model_name = get_model_name()
        timeout = config.timeout if config else 300

        if request.stream:
            response_stream_queue: queue.Queue = queue.Queue()
            request_package.stream_queue = response_stream_queue
            request_queue.put(request_package)

            return StreamingResponse(
                stream_generator(req_id, response_stream_queue, model_name, request),
                media_type="text/event-stream",
            )
        else:
            response_container: Dict[str, Any] = {}
            signal_event = threading.Event()

            request_package.response_container = response_container
            request_package.signal_event = signal_event
            request_queue.put(request_package)

            if not signal_event.wait(timeout=timeout):
                logger.error(f"[{req_id}] Timeout waiting for LLM worker")
                raise HTTPException(
                    status_code=504, detail="Request timed out waiting for LLM worker."
                )

            if "error" in response_container:
                logger.error(f"[{req_id}] Worker error: {response_container['error']}")
                raise HTTPException(
                    status_code=500,
                    detail=f"LLM processing error: {response_container['error']}",
                )

            raw_completion = response_container.get("result")
            if not raw_completion:
                logger.error(f"[{req_id}] No result from worker")
                raise HTTPException(status_code=500, detail="No result from LLM worker.")

            try:
                content = raw_completion["choices"][0]["message"]["content"]
                usage = raw_completion.get("usage", {})

                return AnthropicResponse(
                    id=f"msg_{uuid.uuid4().hex}",
                    content=[TextBlock(text=content)],
                    model=model_name,
                    stop_reason="end_turn",
                    usage=AnthropicUsage(
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                    ),
                )
            except Exception as e:
                logger.error(f"[{req_id}] Error formatting response: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Error formatting LLM response: {e}"
                )

    return router
