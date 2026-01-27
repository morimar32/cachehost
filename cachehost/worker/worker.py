"""Worker thread for processing LLM requests."""

import queue
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..backends.protocol import GenerationParams, LLMBackend
from ..cache.manager import CacheManager
from ..logging_config import get_logger
from ..schemas.common import ChatMessage

logger = get_logger("worker")

# ANSI color codes for terminal output
_GREEN = "\033[92m"
_RED = "\033[91m"
_RESET = "\033[0m"


def _supports_color() -> bool:
    """Check if the terminal supports color output."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


@dataclass
class RequestPackage:
    """Package containing all information for an LLM request."""
    id: str
    messages: List[ChatMessage]
    stream_flag: bool
    params: GenerationParams
    response_container: Optional[Dict[str, Any]] = None
    stream_queue: Optional[queue.Queue] = None
    signal_event: Optional[threading.Event] = None


class LLMWorker:
    """
    Worker thread that processes LLM requests sequentially.

    This worker:
    1. Receives requests from a queue
    2. Looks up cached states for message prefixes
    3. Generates responses (streaming or non-streaming)
    4. Caches the final state after generation
    """

    def __init__(
        self,
        backend: LLMBackend,
        cache_manager: CacheManager,
        request_queue: queue.Queue,
    ):
        """
        Initialize the worker.

        Args:
            backend: LLM backend for generation
            cache_manager: Cache manager for state persistence
            request_queue: Queue to receive requests from
        """
        self.backend = backend
        self.cache_manager = cache_manager
        self.request_queue = request_queue
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start the worker thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Worker thread already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.debug("Worker thread started")

    def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the worker thread.

        Args:
            timeout: Maximum time to wait for thread to join
        """
        self._running = False
        # Send shutdown signal
        self.request_queue.put(None)

        if self._thread:
            logger.debug("Waiting for worker thread to join...")
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Worker thread did not join in time")
            else:
                logger.debug("Worker thread stopped")

    def is_alive(self) -> bool:
        """Check if worker thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        """Main worker loop."""
        logger.debug("Worker thread running")

        while self._running:
            try:
                request_package = self.request_queue.get()

                # Check for shutdown signal
                if request_package is None:
                    logger.debug("Worker received shutdown signal")
                    break

                self._process_request(request_package)

            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}")

        logger.debug("Worker thread exiting")

    def _process_request(self, pkg: RequestPackage) -> None:
        """Process a single request."""
        req_id = pkg.id
        logger.debug(f"[{req_id}] Processing request. History length: {len(pkg.messages)}")

        try:
            # 1. Cache lookup - find longest matching prefix
            cached_state, prefix_length = self.cache_manager.find_longest_prefix_match(
                pkg.messages
            )

            if cached_state is not None:
                try:
                    self.backend.load_state(cached_state)
                    # Green text for cache hit
                    if _supports_color():
                        logger.info(f"{_GREEN}[{req_id}] KV cache loaded (prefix length {prefix_length}){_RESET}")
                    else:
                        logger.info(f"[{req_id}] KV cache loaded (prefix length {prefix_length})")
                except Exception as e:
                    logger.warning(f"[{req_id}] Failed to load cached state: {e}. Starting fresh.")
                    self.backend.reset()
            else:
                self.backend.reset()
                # Red text for cache miss
                if _supports_color():
                    logger.info(f"{_RED}[{req_id}] No cache found, starting fresh{_RESET}")
                else:
                    logger.info(f"[{req_id}] No cache found, starting fresh")

            # 2. Generate response
            assistant_content = ""

            if pkg.stream_flag and pkg.stream_queue:
                assistant_content = self._generate_streaming(pkg)
            else:
                assistant_content = self._generate_non_streaming(pkg)

            # 3. Cache final state
            if assistant_content:
                final_assistant_message = ChatMessage(role="assistant", content=assistant_content)
                history_after_generation = pkg.messages + [final_assistant_message]

                try:
                    state_to_save = self.backend.save_state()
                    self.cache_manager.save_state(history_after_generation, state_to_save)
                    logger.debug(f"[{req_id}] Saved state after generation")
                except Exception as e:
                    logger.warning(f"[{req_id}] Failed to save state: {e}")
            else:
                logger.warning(f"[{req_id}] No assistant response generated, skipping cache save")

        except Exception as e:
            logger.error(f"[{req_id}] Error processing request: {e}")
            if pkg.stream_flag and pkg.stream_queue:
                pkg.stream_queue.put({"error": str(e)})
                pkg.stream_queue.put(None)
            elif pkg.response_container is not None:
                pkg.response_container["error"] = str(e)

        finally:
            if pkg.signal_event:
                pkg.signal_event.set()
            self.request_queue.task_done()
            logger.debug(f"[{req_id}] Request processing complete")

    def _generate_streaming(self, pkg: RequestPackage) -> str:
        """Generate streaming response and return full content."""
        content_parts: List[str] = []
        stream_queue = pkg.stream_queue

        for chunk in self.backend.generate_stream(pkg.messages, pkg.params):
            # Build chunk data for the stream queue
            chunk_data = {
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": chunk.finish_reason,
                }]
            }

            if chunk.role:
                chunk_data["choices"][0]["delta"]["role"] = chunk.role
            if chunk.content:
                chunk_data["choices"][0]["delta"]["content"] = chunk.content
                content_parts.append(chunk.content)

            if chunk.total_tokens > 0:
                chunk_data["usage"] = {
                    "prompt_tokens": chunk.prompt_tokens,
                    "completion_tokens": chunk.completion_tokens,
                    "total_tokens": chunk.total_tokens,
                }

            stream_queue.put(chunk_data)

        # Signal end of stream
        stream_queue.put(None)

        return "".join(content_parts)

    def _generate_non_streaming(self, pkg: RequestPackage) -> str:
        """Generate non-streaming response and return content."""
        result = self.backend.generate(pkg.messages, pkg.params)

        if pkg.response_container is not None:
            # Store raw response for API formatting
            pkg.response_container["result"] = {
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.content,
                    },
                    "finish_reason": result.finish_reason,
                }],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens,
                },
            }

        return result.content
