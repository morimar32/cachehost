"""llama-cpp-python backend implementation."""

from typing import Any, Iterator, List, Optional

from llama_cpp import Llama, LlamaGrammar

from ..config import Config
from ..logging_config import get_logger
from ..schemas.common import ChatMessage
from .protocol import GenerationChunk, GenerationParams, GenerationResult

logger = get_logger("backend.llama_cpp")


class LlamaCppBackend:
    """
    LLM backend using llama-cpp-python.

    This wraps the Llama class from llama-cpp-python and implements
    the LLMBackend protocol for state caching.
    """

    def __init__(self, config: Config):
        """
        Initialize the backend with configuration.

        Args:
            config: Server configuration
        """
        self.config = config
        self._llm: Optional[Llama] = None
        self._model_name = config.model_name

    @property
    def model_name(self) -> str:
        """Return the name of the loaded model."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend identifier."""
        return "llama_cpp"

    def load(self) -> None:
        """Load the model."""
        logger.debug(f"Loading model from: {self.config.model_path}")
        logger.debug(f"Context size: {self.config.n_ctx}")

        self._llm = Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            verbose=False,
            seed=self.config.seed,
            offload_kqv=True,
            n_gpu_layers=-1,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repeat_penalty,
            temperature=self.config.temperature,
            min_p=self.config.min_p,
        )

        logger.debug("Model loaded successfully")

    def reset(self) -> None:
        """Reset the model state."""
        if self._llm:
            self._llm.reset()

    def save_state(self) -> Any:
        """Save and return the current KV-cache state."""
        if not self._llm:
            raise RuntimeError("Model not loaded")
        return self._llm.save_state()

    def load_state(self, state: Any) -> None:
        """Load a previously saved KV-cache state."""
        if not self._llm:
            raise RuntimeError("Model not loaded")
        self._llm.load_state(state)

    def _prepare_messages(self, messages: List[ChatMessage]) -> List[dict]:
        """Convert ChatMessage list to llama-cpp format."""
        return [msg.dict() for msg in messages]

    def _build_generation_params(self, params: GenerationParams) -> dict:
        """Build llama-cpp generation parameters from GenerationParams."""
        gen_params = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "min_p": params.min_p,
            "typical_p": params.typical_p,
            "stop": params.stop,
            "max_tokens": params.max_tokens,
            "presence_penalty": params.presence_penalty,
            "frequency_penalty": params.frequency_penalty,
            "repeat_penalty": params.repeat_penalty,
            "tfs_z": params.tfs_z,
            "mirostat_mode": params.mirostat_mode,
            "mirostat_tau": params.mirostat_tau,
            "mirostat_eta": params.mirostat_eta,
        }

        if params.grammar:
            try:
                gen_params["grammar"] = LlamaGrammar.from_string(params.grammar)
            except Exception as e:
                logger.warning(f"Error parsing grammar: {e}. Proceeding without grammar.")

        return gen_params

    def generate(
        self, messages: List[ChatMessage], params: GenerationParams
    ) -> GenerationResult:
        """Generate a complete response (non-streaming)."""
        if not self._llm:
            raise RuntimeError("Model not loaded")

        processed_messages = self._prepare_messages(messages)
        gen_params = self._build_generation_params(params)

        result = self._llm.create_chat_completion(
            messages=processed_messages,
            stream=False,
            **gen_params,
        )

        content = ""
        finish_reason = None
        if result and result.get("choices"):
            content = result["choices"][0]["message"]["content"]
            finish_reason = result["choices"][0].get("finish_reason")

        usage = result.get("usage", {})

        return GenerationResult(
            content=content,
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            raw_response=result,
        )

    def generate_stream(
        self, messages: List[ChatMessage], params: GenerationParams
    ) -> Iterator[GenerationChunk]:
        """Generate a streaming response."""
        if not self._llm:
            raise RuntimeError("Model not loaded")

        processed_messages = self._prepare_messages(messages)
        gen_params = self._build_generation_params(params)

        chunks = self._llm.create_chat_completion(
            messages=processed_messages,
            stream=True,
            **gen_params,
        )

        sent_role = False
        for chunk in chunks:
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            content = delta.get("content", "")
            role = delta.get("role") if not sent_role else None
            if role:
                sent_role = True

            usage = chunk.get("usage", {})

            yield GenerationChunk(
                content=content,
                role=role,
                finish_reason=finish_reason,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                is_final=finish_reason is not None,
            )

    def shutdown(self) -> None:
        """Clean up resources."""
        logger.debug("Shutting down llama-cpp backend")
        self._llm = None
