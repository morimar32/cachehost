# CacheHost
CacheHost is a LLM host that automatically builds and incremental cache of the context in order to avoid reprocessing the context when using iterative dev coding tools locally, such as Roo Code. Llama.cpp is used to run the model.

## Installation

```bash
# CPU (llama.cpp)
pip install .[llama-cpu]

# CUDA (requires --extra-index-url for prebuilt wheels)
pip install .[llama-cuda] --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# ROCm
pip install .[llama-rocm] --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/rocm

# Apple Silicon (MLX)
pip install .[mlx]

# Development (editable install with tests)
pip install -e .[llama-cpu,dev]
```

## Usage

```bash
# Via console script
cachehost -m /path/to/your/model.gguf

# Or via python module
python -m cachehost -m /path/to/your/model.gguf

# Or directly (backward compatible)
python3 main.py -m /path/to/your/model.gguf
```

## MacOS Setup

[Llama.cpp MacOS setup](https://github.com/abetlen/llama-cpp-python/blob/main/docs/install/macos.md)
