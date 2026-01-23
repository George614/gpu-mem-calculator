# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Enhanced documentation with contribution guidelines
- Code of Conduct for community engagement
- Changelog for tracking project history
- GitHub badges for better visibility

## [0.1.0] - 2024

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
