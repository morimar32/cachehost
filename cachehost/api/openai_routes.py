"""OpenAI-compatible API routes."""

import asyncio
import json
import queue
import threading
import time
import uuid
from typing import Any, Dict, Generator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..backends.protocol import GenerationParams
from ..logging_config import get_logger
from ..schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
)
from ..worker.worker import RequestPackage

logger = get_logger("api.openai")


def create_openai_router(app_state: dict) -> APIRouter:
    """
    Create OpenAI-compatible routes router.

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
        req_id: str, response_queue: queue.Queue, model_name: str
    ) -> Generator[str, None, None]:
        """Generate SSE stream from response queue."""
        logger.debug(f"[{req_id}] Stream generator started")
        try:
            while True:
                chunk_data = response_queue.get()

                if chunk_data is None:
                    logger.debug(f"[{req_id}] Stream generator received end signal")
                    break

                # Handle final usage info
                if isinstance(chunk_data, dict) and chunk_data.get("_is_final_usage_info_"):
                    logger.debug(f"[{req_id}] Received final usage info")
                    continue

                # Handle errors
                if isinstance(chunk_data, dict) and "error" in chunk_data:
                    logger.error(f"[{req_id}] Stream error: {chunk_data['error']}")
                    error_response = {
                        "id": f"err-{uuid.uuid4().hex}",
                        "object": "error",
                        "model": model_name,
                        "error": {
                            "message": chunk_data["error"],
                            "type": "llm_processing_error",
                        },
                    }
                    yield f"data: {json.dumps(error_response)}\n\n"
                    break

                # Format chunk
                try:
                    if not isinstance(chunk_data, dict) or "choices" not in chunk_data:
                        logger.warning(f"[{req_id}] Unexpected chunk format: {chunk_data}")
                        continue

                    raw_delta = chunk_data["choices"][0].get("delta", {})
                    if "content" not in raw_delta and "role" not in raw_delta:
                        raw_delta["content"] = ""

                    stream_choice = ChatCompletionResponseStreamChoice(
                        index=chunk_data["choices"][0].get("index", 0),
                        delta=DeltaMessage(**raw_delta),
                        finish_reason=chunk_data["choices"][0].get("finish_reason"),
                    )

                    stream_resp = ChatCompletionStreamResponse(
                        id=chunk_data.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
                        model=model_name,
                        created=chunk_data.get("created", int(time.time())),
                        choices=[stream_choice],
                    )

                    if "usage" in chunk_data and chunk_data["usage"]:
                        stream_resp.usage = UsageInfo(**chunk_data["usage"])

                    json_payload = json.dumps(stream_resp.dict(exclude_none=True))
                    yield f"data: {json_payload}\n\n"

                except Exception as e:
                    logger.error(f"[{req_id}] Error formatting stream chunk: {e}")
                    error_event = {"error": f"Error formatting stream chunk: {e}"}
                    yield f"data: {json.dumps(error_event)}\n\n"

                await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"[{req_id}] Exception in stream_generator: {e}")
            error_event = {"error": f"Stream generation failed: {e}"}
            try:
                yield f"data: {json.dumps(error_event)}\n\n"
            except:
                pass

        finally:
            yield "data: [DONE]\n\n"
            logger.debug(f"[{req_id}] Stream generator finished")

    @router.post("/v1/chat/completions", response_model=None)
    async def create_chat_completion(request: ChatCompletionRequest):
        """OpenAI-compatible chat completion endpoint."""
        req_id = f"req-{uuid.uuid4().hex[:8]}"
        logger.info(f"[{req_id}] Received request. Stream: {request.stream}")

        worker = app_state.get("worker")
        config = app_state.get("config")

        if not worker or not worker.is_alive():
            raise HTTPException(
                status_code=503, detail="LLM service not available or worker not running."
            )

        # Convert request params to GenerationParams
        params = GenerationParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            typical_p=request.typical_p,
            stop=request.stop,
            max_tokens=request.max_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            repeat_penalty=request.repeat_penalty,
            tfs_z=request.tfs_z,
            mirostat_mode=request.mirostat_mode,
            mirostat_tau=request.mirostat_tau,
            mirostat_eta=request.mirostat_eta,
            grammar=request.grammar,
        )

        request_package = RequestPackage(
            id=req_id,
            messages=request.messages,
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
                stream_generator(req_id, response_stream_queue, model_name),
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
                response_message = ChatMessage(
                    role=raw_completion["choices"][0]["message"]["role"],
                    content=raw_completion["choices"][0]["message"]["content"],
                )
                response_choice = ChatCompletionResponseChoice(
                    index=raw_completion["choices"][0].get("index", 0),
                    message=response_message,
                    finish_reason=raw_completion["choices"][0].get("finish_reason"),
                )
                usage_info = UsageInfo(**raw_completion["usage"])

                return ChatCompletionResponse(
                    id=raw_completion.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
                    object="chat.completion",
                    created=raw_completion.get("created", int(time.time())),
                    model=model_name,
                    choices=[response_choice],
                    usage=usage_info,
                )
            except Exception as e:
                logger.error(f"[{req_id}] Error formatting response: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Error formatting LLM response: {e}"
                )

    return router
