# CLAUDE.md

## Project Overview

vLLM is a high-throughput LLM inference and serving library. Originally from UC Berkeley's Sky Computing Lab, it is now a community project. Core innovations: **PagedAttention** for efficient KV-cache memory management, continuous batching, and an OpenAI-compatible API server.

## Runtime & Build System

- Python 3.9+; CUDA required for GPU builds
- Build: `setuptools` + `CMakeLists.txt` for custom CUDA/C++ extensions
- Package manager: pip / conda (no lockfile in repo)

## Development Commands

```bash
# Install from source (CPU-only, no CUDA extensions)
pip install -e . --no-build-isolation

# Install with CUDA extensions (requires CUDA toolkit)
pip install -e . --no-build-isolation -v

# Run tests (requires GPU unless using CPU-only fixtures)
pytest tests/

# Run specific test
pytest tests/test_sampling_params.py -v

# Lint (project uses ruff + mypy)
ruff check vllm/
mypy vllm/ --ignore-missing-imports

# Start OpenAI-compatible server
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

## Project Structure

```
vllm/                         # Core library
  engine/                     # LLM engine (async + sync variants)
  entrypoints/                # CLI + OpenAI API server (FastAPI)
  executor/                   # Multi-GPU / distributed executors
  worker/                     # Per-GPU worker processes
  model_executor/             # Model loading, forward pass routing
  attention/                  # PagedAttention backends (CUDA, ROCm, CPU, TPU)
  core/                       # Scheduler, block manager (KV-cache)
  distributed/                # tensor/pipeline parallelism (torch.dist)
  multimodal/                 # Multi-modal input processing
  lora/                       # LoRA adapter support
  spec_decode/                # Speculative decoding
  compilation/                # torch.compile integration
  v1/                         # V1 architecture (new async execution loop)
  platforms/                  # Platform detection (CUDA, ROCm, CPU, TPU, etc.)
  transformers_utils/         # HuggingFace model config utilities
  config.py                   # All engine/model configuration dataclasses
  envs.py                     # Environment variable config
  sampling_params.py          # SamplingParams dataclass

tests/                        # pytest test suite
benchmarks/                   # Throughput/latency benchmarking scripts
docs/                         # Sphinx documentation source
.buildkite/                   # CI pipelines (nightly benchmarks)
CMakeLists.txt                # C++/CUDA extension build
```

## Key Architectural Concepts

- **PagedAttention**: KV-cache stored in non-contiguous physical blocks, managed by the block manager in `core/`
- **Continuous batching**: The scheduler in `core/scheduler.py` dynamically adds/removes sequences; no fixed batch boundaries
- **Workers**: Each GPU runs a `Worker` process; tensor parallelism splits each layer across workers; pipeline parallelism splits layers across groups
- **V1 engine**: `vllm/v1/` is an in-progress architectural rewrite with a cleaner async execution loop — prefer contributing to V1 for new features

## Supported Backends

| Hardware | Notes |
|---|---|
| NVIDIA CUDA | Primary, fully supported |
| AMD ROCm | Supported via HIP |
| Intel CPU/GPU | Via OpenVINO / IPEX |
| AWS Neuron | Via `neuronx` |
| Google TPU | Via PyTorch/XLA |
| Apple Metal (CPU) | Experimental |

## Contributing

- Full contribution guide: `https://docs.vllm.ai/en/latest/contributing/overview.html`
- New model support: add to `vllm/model_executor/models/` following existing patterns
- New attention backend: implement `AttentionBackend` ABC in `vllm/attention/backends/`
- Run `ruff check` and `mypy` before opening a PR
- Large changes (new backends, architectural changes) should open an RFC issue first
- Tests for new models should use small model fixtures and run without a real GPU via mocking where possible

## Key Conventions

- `SamplingParams` and `EngineArgs` are the public API — keep them stable and backward-compatible
- Environment variables are all declared in `vllm/envs.py` — never use `os.environ` directly in library code
- All configuration flows through `vllm/config.py` dataclasses — avoid ad-hoc dicts
- `vllm/v1/` and `vllm/` (legacy) coexist; use `vllm/platforms/` for runtime hardware detection, not hardcoded strings
