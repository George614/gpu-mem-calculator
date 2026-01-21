# GPU Memory Calculator for LLM Training

A versatile Python application for calculating GPU memory requirements for training Large Language Models with support for multiple training engines including PyTorch DDP, DeepSpeed ZeRO, Megatron-LM, and FSDP.

<p align="center">
  <img src="screenshot.png" alt="GPU Memory Calculator Screenshot" width="800">
</p>

## Features

- **Multiple Training Engines**: Support for PyTorch DDP, DeepSpeed ZeRO (stages 1-3), Megatron-LM, Megatron+DeepSpeed, and PyTorch FSDP
- **Dual Interface**: Both CLI and Web UI for flexible usage
- **Preset Models**: Quick-load configurations for popular models (LLaMA 2, GPT-3, etc.)
- **Detailed Breakdown**: Memory breakdown by component (parameters, gradients, optimizer states, activations)
- **Feasibility Analysis**: Check if your configuration fits on available GPU memory
- **Easy Config**: JSON-based configuration files with human-readable parameter formats

## Installation

### From source

```bash
git clone https://github.com/George614/gpu-mem_calculator.git
cd gpu_mem_calculator
pip install -e .
```

### For Web UI support

```bash
pip install -e ".[web]"
```

### Development installation

```bash
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

#### Using model presets (Recommended)

The calculator includes pre-configured model presets for popular LLMs:

```bash
# List all available presets
gpu-mem-calc presets

# Calculate with a preset
gpu-mem-calc calculate --preset llama2-7b
gpu-mem-calc calculate --preset mixtral-8x7b --format json

# List presets in table format
gpu-mem-calc presets --format table
```

Available presets include:
- **Dense Models**: LLaMA 2 (7B, 13B, 70B), GPT-3 (175B)
- **MoE Models**: Mixtral 8x7B, GLM-4 (9B), GLM-4.7 (355B), GLM-4.5 Air (106B),
  Qwen1.5-MoE-A2.7B, DeepSeek-MoE (16B)

#### Calculate from config file

```bash
gpu-mem-calc calculate --config configs/llama2_7b_deepspeed.json
```

#### Quick calculation from model size

```bash
# Calculate memory for 7B model with 8x80GB GPUs using DeepSpeed
gpu-mem-calc quick 7 --gpus 8 --engine deepspeed

# With custom GPU memory
gpu-mem-calc quick 70 --gpus 64 --gpu-mem 80 --engine megatron
```

#### Validate configuration

```bash
gpu-mem-calc validate configs/my_config.json
```

### Web Interface

Start the web server:

```bash
python -m gpu_mem_calculator.web.app
```

Or using uvicorn directly:

```bash
uvicorn gpu_mem_calculator.web.app:app --reload
```

Then open your browser to `http://localhost:8000`

### Python API

```python
from gpu_mem_calculator.core.calculator import GPUMemoryCalculator
from gpu_mem_calculator.core.models import (
    ModelConfig,
    TrainingConfig,
    ParallelismConfig,
    EngineConfig,
    GPUConfig,
)

# Create configuration
model_config = ModelConfig(
    name="llama2-7b",
    num_parameters=7_000_000_000,
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    vocab_size=32000,
    max_seq_len=4096,
)

training_config = TrainingConfig(
    batch_size=4,
    gradient_accumulation_steps=4,
    dtype="bf16",
    optimizer="adamw",
)

parallelism_config = ParallelismConfig(
    data_parallel_size=8,
)

engine_config = EngineConfig(
    type="deepspeed",
    zero_stage=3,
    offload_optimizer="cpu",
)

gpu_config = GPUConfig(
    num_gpus=8,
    gpu_memory_gb=80,
)

# Calculate memory
calculator = GPUMemoryCalculator(
    model_config=model_config,
    training_config=training_config,
    parallelism_config=parallelism_config,
    engine_config=engine_config,
    gpu_config=gpu_config,
)

result = calculator.calculate()

print(f"Memory per GPU: {result.total_memory_per_gpu_gb:.2f} GB")
print(f"Fits on GPU: {result.fits_on_gpu}")
print(f"Utilization: {result.memory_utilization_percent:.1f}%")
```

## Configuration File Format

```json
{
  "model": {
    "name": "llama2-7b",
    "num_parameters": "7B",
    "num_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "vocab_size": 32000,
    "max_seq_len": 4096
  },
  "training": {
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "optimizer": "adamw",
    "dtype": "bf16",
    "activation_checkpointing": 1
  },
  "parallelism": {
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "data_parallel_size": 8,
    "sequence_parallel": false
  },
  "engine": {
    "type": "deepspeed",
    "zero_stage": 3,
    "offload_optimizer": "cpu",
    "offload_param": "none"
  },
  "hardware": {
    "num_gpus": 8,
    "gpu_memory_gb": 80
  }
}
```

## Supported Training Engines

### PyTorch DDP (Baseline)
Standard Distributed Data Parallel training without memory optimizations.

### DeepSpeed ZeRO
- **ZeRO-1**: Shard optimizer states
- **ZeRO-2**: Shard optimizer states + gradients
- **ZeRO-3**: Shard everything (parameters, gradients, optimizer states)
- Supports CPU/NVMe offloading

### Megatron-LM
Tensor and pipeline parallelism with activation checkpointing support.

### Megatron + DeepSpeed
Combines Megatron-LM's model parallelism with DeepSpeed ZeRO's optimizer sharding.

### PyTorch FSDP
Fully Sharded Data Parallel with multiple sharding strategies.

## Memory Formulas

The calculator uses the following formulas based on training engine:

**Base Components:**
- Model Parameters: `num_params × bytes_per_param` (FP16/BF16 = 2 bytes, FP32 = 4 bytes)
- Gradients: `num_params × bytes_per_param` (same precision as parameters during training)
- Optimizer States: `num_params × 12` for Adam/AdamW (FP32 param copy + momentum + variance)
- Activations: `batch_size × seq_len × hidden_size × num_layers × 16` (heuristic approximation)

**DeepSpeed ZeRO Stages:**

*ZeRO-1* (shard optimizer states):
```
total_per_gpu = 4 × params + (12 × params) / num_gpus
```

*ZeRO-2* (shard optimizer + gradients):
```
total_per_gpu = 2 × params + (2 × params) / num_gpus + (12 × params) / num_gpus
```

*ZeRO-3* (shard everything):
```
total_per_gpu = largest_layer_memory + (16 × params) / num_gpus
where largest_layer_memory = 4 × largest_layer_params
```

Note: ZeRO-3 uses 16 bytes (not 18) when offloading is disabled, representing:
- 12 bytes: Optimizer states (FP32)
- 2 bytes: Sharded FP16 parameters
- 2 bytes: Sharded FP16 gradients

**References:**
- [DeepSpeed Memory Documentation](https://deepspeed.readthedocs.io/en/latest/memory.html)
- [Microsoft Research ZeRO Blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
- [EleutherAI Transformer Math 101](https://blog.eleuther.ai/transformer-math/)

## Example Configurations

### LLaMA 2 7B with DeepSpeed ZeRO-3
```bash
gpu-mem-calc calculate --config configs/llama2_7b_deepspeed.json
```

### GPT-3 175B with Megatron-LM
```bash
gpu-mem-calc calculate --config configs/gpt3_175b_megatron.json
```

### Custom 1B model with PyTorch DDP
```bash
gpu-mem-calc calculate --config configs/pytorch_ddp_example.json
```

## Web UI Features

- **Interactive Form**: Easy-to-use interface for tweaking hyperparameters
- **Unified Presets**: Same model presets available in CLI and web interface
- **Real-time Validation**: Instant feedback on configuration validity
- **Visual Breakdown**: Bar chart showing memory component distribution
- **Feasibility Indicators**: Color-coded memory utilization status
- **Export Options**: Save config as JSON or copy to clipboard

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ cli/ web/
ruff check src/ cli/ web/
```

### Type Checking

```bash
mypy src/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

The memory calculations in this tool are based on authoritative sources:

**Core Memory Formulas:**
- [EleutherAI Transformer Math 101](https://blog.eleuther.ai/transformer-math/) - Comprehensive breakdown of transformer memory requirements
- [Microsoft Research ZeRO Blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) - ZeRO optimization techniques
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2204.13323) - Activation checkpointing strategies

**Engine Documentation:**
- [DeepSpeed Memory Documentation](https://deepspeed.readthedocs.io/en/latest/memory.html) - Official DeepSpeed memory formulas
- [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Tensor and pipeline parallelism
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html) - Fully sharded data parallel
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) - Distributed data parallel

**Related Tools:**
- [llm-analysis](https://github.com/cli99/llm-analysis) - LLM memory analysis
- [vram-calculator](https://github.com/furiousteabag/vram-calculator) - VRAM calculation utilities

## License

MIT License

## Acknowledgments

This tool was inspired by and builds upon the excellent work of:
- [DeepSpeed Memory Estimator](https://deepspeed.readthedocs.io/en/latest/memory.html) - ZeRO memory optimization formulas
- [llm-analysis](https://github.com/cli99/llm-analysis) - LLM memory analysis methodology
- [vram-calculator](https://github.com/furiousteabag/vram-calculator) - VRAM calculation approach

Special thanks to the EleutherAI community for their comprehensive [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) guide, which provides detailed formulas for transformer memory calculations.
