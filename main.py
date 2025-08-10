import asyncio
import hashlib
import json
import os
import pickle # Added for LlamaState serialization
import queue
import shutil
import threading
import time
import uuid
import atexit
import argparse
import uvicorn
from typing import List, Dict, Any, Optional, Generator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from llama_cpp import Llama, LlamaGrammar
# Assuming LlamaState might be needed for type hinting if available, but not strictly necessary for functionality.
# from llama_cpp import LlamaState # If LlamaState is directly importable

# --- Configuration ---
parser = argparse.ArgumentParser(description="Llama.cpp OpenAI-Compatible API with KV Cache")
parser.add_argument("-m", "--model-path", required=True, help="Path to the LLM model file")
parser.add_argument("--n-ctx", type=int, default=4096, help="Context size for the model")
parser.add_argument("--top-k", type=int, default=20, help="Top K sampling parameter (default: 20)")
parser.add_argument("--repeat-penalty", type=float, default=1.0, help="Repeat penalty parameter (default: 1.0)")
parser.add_argument("--temperature", type=float, default=0.7, help="Temperature parameter (default: 0.7)")
parser.add_argument("--min-p", type=float, default=0.05, help="Min P sampling parameter (default: 0.05)")
parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility (default: 3407)")
args = parser.parse_args()

LLM_MODEL_PATH = args.model_path
if not os.path.exists(LLM_MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {LLM_MODEL_PATH}")

LLM_N_CTX = args.n_ctx
LLM_N_THREADS = os.cpu_count()
LLM_TOP_K = args.top_k
LLM_REPEAT_PENALTY = args.repeat_penalty
LLM_TEMPERATURE = args.temperature
LLM_MIN_P = args.min_p
LLM_SEED = args.seed

MODEL_NAME = os.path.splitext(os.path.basename(LLM_MODEL_PATH))[0].replace('.', '_')
CACHE_BASE_DIR = "./cache"
MODEL_CACHE_DIR = os.path.join(CACHE_BASE_DIR, MODEL_NAME)

# --- Global Variables ---
REQUEST_QUEUE = queue.Queue() 
llm_instance: Optional[Llama] = None
llm_worker_thread: Optional[threading.Thread] = None

# --- Pydantic Models (OpenAI Compatible) ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None 
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    typical_p: float = 1.0
    stream: bool = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None 
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repeat_penalty: float = 1.1
    tfs_z: float = 1.0
    mirostat_mode: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    grammar: Optional[str] = None 

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str 
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str 
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None 

# --- Helper Functions ---

def get_cache_filepath(history_prefix: List[ChatMessage]) -> str:
    if not history_prefix: 
        combined_content = MODEL_NAME + "__initial_state__"
    else:
        combined_content = MODEL_NAME + "".join(msg.content for msg in history_prefix)
    
    hasher = hashlib.sha256()
    hasher.update(combined_content.encode('utf-8'))
    hashed_content = hasher.hexdigest()
    return os.path.join(MODEL_CACHE_DIR, f"{hashed_content}.pkl") 

def init_cache_directory():
    if os.path.exists(CACHE_BASE_DIR):
        shutil.rmtree(CACHE_BASE_DIR)
        print(f"Cleared cache directory: {CACHE_BASE_DIR}")
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    print(f"Initialized cache directory: {MODEL_CACHE_DIR}")

def cleanup_cache_directory():
    if os.path.exists(CACHE_BASE_DIR):
        shutil.rmtree(CACHE_BASE_DIR)
        print(f"Cleaned up cache directory: {CACHE_BASE_DIR}")

atexit.register(cleanup_cache_directory)

# --- LLM Worker ---
def llm_worker_main():
    global llm_instance
    if not llm_instance:
        print("LLM instance not initialized in worker.") 
        return

    print("LLM Worker thread started.")
    while True:
        try:
            request_package = REQUEST_QUEUE.get()
            if request_package is None: 
                print("LLM Worker received shutdown signal.")
                break

            req_id = request_package.get("id", "unknown_req")
            messages: List[ChatMessage] = request_package["messages"]
            stream_flag: bool = request_package["stream_flag"]
            response_container: Optional[Dict[str, Any]] = request_package.get("response_container")
            stream_queue: Optional[queue.Queue] = request_package.get("stream_queue") 
            signal_event: Optional[threading.Event] = request_package.get("signal_event")
            request_params = request_package["params"]

            print(f"[{req_id}] Worker processing request. History length: {len(messages)}")

            # 1. Cache Lookup
            loaded_from_cache_path: Optional[str] = None
            
            for i in range(len(messages), -1, -1):
                history_prefix = messages[:i]
                cache_file = get_cache_filepath(history_prefix)
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, "rb") as f: 
                            state_object_to_load = pickle.load(f)
                        llm_instance.load_state(state_object_to_load) 
                        loaded_from_cache_path = cache_file
                        print(f"[{req_id}] Loaded state from: {cache_file} for prefix of length {len(history_prefix)}")
                        break 
                    except Exception as e:
                        print(f"[{req_id}] Error loading state from {cache_file}: {e}. Attempting to continue without cache.")
                        llm_instance.reset() 
                        loaded_from_cache_path = None 
            
            if not loaded_from_cache_path:
                llm_instance.reset() 
                print(f"[{req_id}] No suitable cache found or initial request, starting from fresh state.")

            # 2. Generation
            assistant_response_content_parts: List[str] = []
            completion_result: Optional[Dict[str, Any]] = None
            
            generation_params = {
                "temperature": request_params.temperature,
                "top_p": request_params.top_p,
                "top_k": request_params.top_k,
                "min_p": request_params.min_p,
                "typical_p": request_params.typical_p,
                "stop": request_params.stop,
                "max_tokens": request_params.max_tokens,
                "presence_penalty": request_params.presence_penalty,
                "frequency_penalty": request_params.frequency_penalty,
                "repeat_penalty": request_params.repeat_penalty,
                "tfs_z": request_params.tfs_z,
                "mirostat_mode": request_params.mirostat_mode,
                "mirostat_tau": request_params.mirostat_tau,
                "mirostat_eta": request_params.mirostat_eta,
            }
            if request_params.grammar:
                try:
                    generation_params["grammar"] = LlamaGrammar.from_string(request_params.grammar)
                except Exception as e:
                    print(f"[{req_id}] Error parsing grammar: {e}. Proceeding without grammar.")

            processed_messages = [msg.dict() for msg in messages]

            if stream_flag and stream_queue:
                chunks = llm_instance.create_chat_completion(
                    messages=processed_messages, 
                    stream=True,
                    **generation_params
                )
                
                last_chunk_usage = None
                for i, chunk in enumerate(chunks):
                    stream_queue.put(chunk) 
                    choice = chunk.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    if "content" in delta:
                        assistant_response_content_parts.append(delta["content"])
                    if i == 0 and 'role' not in delta : 
                        stream_queue.put({
                            "id": chunk["id"], "object": chunk["object"], "created": chunk["created"],
                            "model": MODEL_NAME, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                        })
                    if chunk.get("usage"): 
                        last_chunk_usage = chunk["usage"]

                stream_queue.put(None) 
                if last_chunk_usage: 
                     stream_queue.put({"usage": last_chunk_usage, "_is_final_usage_info_": True})

            else: # Non-streaming
                completion_result = llm_instance.create_chat_completion(
                    messages=processed_messages, 
                    stream=False,
                    **generation_params
                )
                if response_container is not None:
                    response_container['result'] = completion_result
                
                if completion_result and completion_result.get("choices"):
                    assistant_response_content_parts.append(completion_result["choices"][0]["message"]["content"])

            # 3. Cache Final State (after generation)
            final_assistant_response_content = "".join(assistant_response_content_parts)
            if final_assistant_response_content: 
                final_assistant_message = ChatMessage(role="assistant", content=final_assistant_response_content)
                history_after_generation = messages + [final_assistant_message] 
                final_cache_file = get_cache_filepath(history_after_generation)
                try:
                    state_object_to_save = llm_instance.save_state() 
                    with open(final_cache_file, "wb") as f: 
                        pickle.dump(state_object_to_save, f)
                    print(f"[{req_id}] Saved state to: {final_cache_file}")
                except Exception as e:
                    print(f"[{req_id}] Error saving state to {final_cache_file}: {e}")
            else:
                print(f"[{req_id}] No assistant response generated, skipping caching final state.")

        except Exception as e:
            print(f"[{req_id}] Error in LLM worker: {e}")
            if stream_flag and stream_queue:
                stream_queue.put({"error": str(e)}) 
                stream_queue.put(None)
            elif response_container is not None:
                response_container['error'] = str(e)
        finally:
            if signal_event:
                signal_event.set() 
            REQUEST_QUEUE.task_done()
            print(f"[{req_id}] Worker finished processing request.")


# --- FastAPI Application ---
app = FastAPI(title="Llama.cpp OpenAI-Compatible API with KV Cache")

@app.on_event("startup")
async def startup_event():
    global llm_instance, llm_worker_thread
    print("Server startup...")
    init_cache_directory()
    
    print(f"Loading LLM model from: {LLM_MODEL_PATH}")
    print(f"Using context size (n_ctx): {LLM_N_CTX}")
    print(f"Model name for caching: {MODEL_NAME}")

    try:
        llm_instance = Llama(
            model_path=LLM_MODEL_PATH,
            n_ctx=LLM_N_CTX,
            n_threads=LLM_N_THREADS,
            verbose=True,
            seed=LLM_SEED,
            offload_kqv=True,
            n_gpu_layers=-1,
            # LLM sampling parameters
            top_k=LLM_TOP_K,
            repeat_penalty=LLM_REPEAT_PENALTY,
            temperature=LLM_TEMPERATURE,
            min_p=LLM_MIN_P
        )
    except Exception as e:
        print(f"Failed to initialize Llama model: {e}")
        raise RuntimeError(f"Could not initialize Llama model: {e}") from e

    llm_worker_thread = threading.Thread(target=llm_worker_main, daemon=True)
    llm_worker_thread.start()
    print("LLM Worker thread initialized and started.")
    print("Server startup complete. Ready for requests.")

@app.on_event("shutdown")
async def shutdown_event():
    print("Server shutdown initiated...")
    REQUEST_QUEUE.put(None) 
    if llm_worker_thread:
        print("Waiting for LLM worker thread to join...")
        llm_worker_thread.join(timeout=10) 
        if llm_worker_thread.is_alive():
            print("LLM worker thread did not join in time.")
    print("Server shutdown complete.")


async def stream_generator(req_id: str, response_queue: queue.Queue) -> Generator[str, None, None]: 
    print(f"[{req_id}] Stream generator started.")
    try:
        while True:
            chunk_data = response_queue.get()
            if chunk_data is None: 
                print(f"[{req_id}] Stream generator received None sentinel (normal end).")
                break
            
            if isinstance(chunk_data, dict) and chunk_data.get("_is_final_usage_info_"):
                print(f"[{req_id}] Stream generator received final usage info: {chunk_data.get('usage')}")
                continue 

            if isinstance(chunk_data, dict) and "error" in chunk_data:
                 print(f"[{req_id}] Stream generator received error: {chunk_data['error']}")
                 error_response = {
                     "id": f"err-{uuid.uuid4().hex}",
                     "object": "error",
                     "model": MODEL_NAME,
                     "error": {
                         "message": chunk_data['error'],
                         "type": "llm_processing_error"
                     }
                 }
                 yield f"data: {json.dumps(error_response)}\n\n"
                 break

            try:
                if not isinstance(chunk_data, dict) or "choices" not in chunk_data:
                    print(f"[{req_id}] Stream generator received unexpected chunk format: {chunk_data}")
                    continue

                raw_delta = chunk_data["choices"][0].get("delta", {})
                if "content" not in raw_delta and "role" in raw_delta : 
                     raw_delta["content"] = "" 
                elif "content" not in raw_delta and "role" not in raw_delta: 
                    raw_delta["content"] = "" 

                stream_choice = ChatCompletionResponseStreamChoice(
                    index=chunk_data["choices"][0].get("index", 0),
                    delta=DeltaMessage(**raw_delta),
                    finish_reason=chunk_data["choices"][0].get("finish_reason")
                )
                stream_resp_obj = ChatCompletionStreamResponse(
                    id=chunk_data.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
                    model=MODEL_NAME, 
                    created=chunk_data.get("created", int(time.time())),
                    choices=[stream_choice]
                )
                if "usage" in chunk_data and chunk_data["usage"] is not None:
                    stream_resp_obj.usage = UsageInfo(**chunk_data["usage"])

                # CHANGED: Use .dict() and json.dumps() for Pydantic V1 compatibility
                json_payload = json.dumps(stream_resp_obj.dict(exclude_none=True))
                yield f"data: {json_payload}\n\n"
            except Exception as e:
                print(f"[{req_id}] Error formatting stream chunk: {e}. Chunk: {chunk_data}")
                error_event = {"error": f"Error formatting stream chunk: {e}"}
                yield f"data: {json.dumps(error_event)}\n\n"

            await asyncio.sleep(0.001) 
    except Exception as e:
        print(f"[{req_id}] Exception in stream_generator: {e}")
        error_event = {"error": f"Stream generation failed: {e}"}
        try:
            yield f"data: {json.dumps(error_event)}\n\n"
        except: 
            pass
    finally:
        yield f"data: [DONE]\n\n"
        print(f"[{req_id}] Stream generator finished.")


@app.post("/v1/chat/completions", response_model=None) 
async def create_chat_completion(request: ChatCompletionRequest, http_request: Request):
    req_id = f"req-{uuid.uuid4().hex[:8]}"
    print(f"[{req_id}] Received request /v1/chat/completions. Stream: {request.stream}")

    if not llm_instance or not llm_worker_thread or not llm_worker_thread.is_alive():
        raise HTTPException(status_code=503, detail="LLM service not available or worker not running.")

    request_package = {
        "id": req_id,
        "messages": request.messages, 
        "stream_flag": request.stream,
        "params": request, 
    }

    if request.stream:
        response_stream_queue = queue.Queue() 
        request_package["stream_queue"] = response_stream_queue
        REQUEST_QUEUE.put(request_package)
        return StreamingResponse(
            stream_generator(req_id, response_stream_queue),
            media_type="text/event-stream"
        )
    else: 
        response_container: Dict[str, Any] = {} 
        signal_event = threading.Event() 
        
        request_package["response_container"] = response_container
        request_package["signal_event"] = signal_event
        REQUEST_QUEUE.put(request_package)
        
        if not signal_event.wait(timeout=300): 
             print(f"[{req_id}] Timeout waiting for LLM worker.")
             raise HTTPException(status_code=504, detail="Request timed out waiting for LLM worker.")

        if 'error' in response_container:
            print(f"[{req_id}] LLM worker reported an error: {response_container['error']}")
            raise HTTPException(status_code=500, detail=f"LLM processing error: {response_container['error']}")

        raw_completion = response_container.get('result')
        if not raw_completion:
            print(f"[{req_id}] No result from LLM worker.")
            raise HTTPException(status_code=500, detail="No result from LLM worker.")

        try:
            response_message = ChatMessage(
                role=raw_completion["choices"][0]["message"]["role"],
                content=raw_completion["choices"][0]["message"]["content"]
            )
            response_choice = ChatCompletionResponseChoice(
                index=raw_completion["choices"][0]["index"],
                message=response_message,
                finish_reason=raw_completion["choices"][0].get("finish_reason")
            )
            usage_info = UsageInfo(**raw_completion["usage"])
            
            openai_response = ChatCompletionResponse(
                id=raw_completion.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
                object="chat.completion",
                created=raw_completion.get("created", int(time.time())),
                model=MODEL_NAME, 
                choices=[response_choice],
                usage=usage_info
            )
            return openai_response
        except Exception as e:
            print(f"[{req_id}] Error formatting non-streaming response: {e}. Raw: {raw_completion}")
            raise HTTPException(status_code=500, detail=f"Error formatting LLM response: {e}")

@app.get("/health")
async def health_check():
    if llm_instance and llm_worker_thread and llm_worker_thread.is_alive():
        return {"status": "ok", "model_loaded": True, "model_name": MODEL_NAME}
    return {"status": "degraded", "model_loaded": False, "detail": "LLM instance or worker not available."}

if __name__ == "__main__":
    if not LLM_MODEL_PATH:
        print("FATAL: LLM_MODEL_PATH environment variable must be set.")
        print("Example: export LLM_MODEL_PATH=/path/to/your/model.gguf")
        exit(1)
    
    print(f"Starting server for model: {MODEL_NAME}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
