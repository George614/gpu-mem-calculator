# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 🚀 **Inference Memory Calculation**: Support for HuggingFace, vLLM, TGI, TensorRT-LLM, SGLang inference engines
  - KV cache quantization (NONE, INT8, FP8, INT4)
  - Tensor parallelism for distributed inference
  - Throughput estimation and batch size optimization
  - Memory feasibility analysis for inference workloads
  - Engine-specific configuration parameters:
    - **TGI**: max_total_tokens, max_input_tokens, max_batch_total_tokens, tgi_quantize, tgi_dtype, sharded, num_shard
    - **vLLM**: block_size, swap_space_gb, enable_prefix_caching, enforce_eager, max_num_batched_tokens, max_num_seqs, vllm_quantization
    - **TensorRT-LLM**: trt_max_batch_size, trt_max_input_len, trt_max_seq_len, trt_max_beam_width
    - **SGLang**: chunk_size, max_running_requests, disable_radix_cache, enable_p2p, attention_backend, enable_torch_compile, radix_cache_max_seq_len, speculative_algo, multi_lora_enabled
  - SGLang RadixAttention memory optimization with tree-based KV cache sharing
  - Chunked prefill and speculative decoding support for SGLang
- 🌐 **Multi-Node Training**: Network overhead calculation and hybrid parallelism optimization
  - AllReduce, AllGather, ReduceScatter, pipeline communication estimation
  - Interconnect support (InfiniBand, NVLink, Ethernet 10G/25G/100G/200G)
  - Automatic TP+PP+DP strategy optimization
  - ZeRO stage impact analysis
- 📦 **Framework Configuration Exporters**: Export to Accelerate, Lightning, Axolotl formats
  - HuggingFace Accelerate config generation
  - PyTorch Lightning Trainer configuration
  - Axolotl YAML config for fine-tuning
  - DeepSpeed config export
  - Generic YAML/JSON export
- 🧪 **Comprehensive Test Suite**: 239 tests with 95% coverage on new modules
  - Training calculator tests (48 tests)
  - Inference engine tests (97 tests)
  - Multi-node calculator tests (50 tests)
  - Exporter tests (44 tests)

### Changed
- Updated README with new feature documentation and API examples
- Updated FAQ with inference, multi-node, and exporter information
- Updated Getting Started guide with new feature examples

## [0.1.0] - 2026

### Added
- Initial release of GPU Memory Calculator
- Support for multiple training engines:
  - PyTorch DDP (Distributed Data Parallel)
  - DeepSpeed ZeRO (stages 1, 2, and 3)
  - Megatron-LM (tensor and pipeline parallelism)
  - Megatron + DeepSpeed hybrid
  - PyTorch FSDP (Fully Sharded Data Parallel)
- Command-line interface (CLI) with rich formatting
- Web interface using FastAPI and modern HTML/CSS/JavaScript
- Model presets for popular LLMs:
  - LLaMA 2 (7B, 13B, 70B)
  - GPT-3 (175B)
  - Mixtral 8x7B
  - GLM-4 variants
  - Qwen1.5-MoE
  - DeepSeek-MoE
- JSON-based configuration system
- Memory breakdown by component:
  - Model parameters
  - Gradients
  - Optimizer states
  - Activations
- GPU feasibility analysis
- CPU/NVMe offloading support for DeepSpeed
- Activation checkpointing support
- Human-readable parameter formats (e.g., "7B" for 7 billion)
- Comprehensive documentation with formula explanations
- Test suite with pytest
- Code quality tools (Black, Ruff, MyPy)

### Features
- Calculate memory requirements for any model size
- Quick calculation mode for rapid estimates
- Preset models for one-command calculations
- Detailed memory utilization reporting
- Support for mixed precision training (FP16, BF16, FP32)
- Various optimizer support (Adam, AdamW, SGD)
- Parallelism configuration (data, tensor, pipeline)
- Interactive web UI with visual breakdowns
- Export configurations to JSON
- Validation of configuration files

### Documentation
- Comprehensive README with usage examples
- Configuration file format documentation
- Memory formula explanations with references
- API usage examples
- Web UI feature guide
- Links to authoritative sources (DeepSpeed, Megatron-LM, research papers)

[Unreleased]: https://github.com/George614/gpu-mem-calculator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/George614/gpu-mem-calculator/releases/tag/v0.1.0
