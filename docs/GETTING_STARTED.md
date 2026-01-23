# Getting Started Guide

Welcome to GPU Memory Calculator! This guide will help you get up and running quickly.

## 🚀 Quick Start (5 minutes)

### Step 1: Install

```bash
pip install git+https://github.com/George614/gpu-mem-calculator.git
```

### Step 2: Try a preset

```bash
gpu-mem-calc calculate --preset llama2-7b
```

You'll see output like:
```
Model: LLaMA 2 7B
Total Parameters: 7.00B
GPUs: 8 × 80GB
Memory per GPU: 24.5 GB
Fits on GPU: ✓ Yes
Utilization: 30.6%
```

### Step 3: Explore the web interface

```bash
uvicorn gpu_mem_calculator.web.app:app --reload
```

Open http://localhost:8000 in your browser and start experimenting!

## 📚 Basic Concepts

### Model Parameters

The number of parameters in your model (e.g., 7 billion for LLaMA 2 7B). This is the most important factor in memory consumption.

### Training Engine

Different engines offer different memory optimizations:
- **DDP**: Standard PyTorch distributed training
- **DeepSpeed**: Advanced memory optimization with ZeRO
- **Megatron**: Model parallelism for huge models
- **FSDP**: PyTorch's fully sharded approach

### Precision

- **FP32**: 4 bytes per parameter (highest precision, most memory)
- **FP16/BF16**: 2 bytes per parameter (good balance)
- **Mixed precision**: Train in FP16/BF16, keep optimizer in FP32

### Parallelism

- **Data Parallel**: Same model on each GPU, different data batches
- **Tensor Parallel**: Split model layers across GPUs
- **Pipeline Parallel**: Split model depth across GPUs

## 🎯 Common Scenarios

### Scenario 1: "Will my model fit?"

You have 8× A100 (80GB) GPUs and want to train LLaMA 2 7B:

```bash
gpu-mem-calc calculate --preset llama2-7b
```

Result: ✅ Yes, it fits with 30% utilization. You have headroom to increase batch size!

### Scenario 2: "How many GPUs do I need?"

You want to train GPT-3 (175B parameters):

```bash
gpu-mem-calc calculate --preset gpt3-175b
```

Result: You'll need 64× 80GB GPUs with Megatron-LM or 128× with standard training.

### Scenario 3: "Should I use DeepSpeed?"

Compare different approaches:

```bash
# PyTorch DDP
gpu-mem-calc quick 7 --gpus 8 --engine pytorch

# DeepSpeed ZeRO-3
gpu-mem-calc quick 7 --gpus 8 --engine deepspeed
```

Result: DeepSpeed can save 3-4× memory compared to DDP!

### Scenario 4: "What if I use CPU offloading?"

Create a config file with CPU offloading enabled:

```json
{
  "model": {"num_parameters": "7B", ...},
  "engine": {
    "type": "deepspeed",
    "zero_stage": 3,
    "offload_optimizer": "cpu",
    "offload_param": "cpu"
  }
}
```

Then calculate:
```bash
gpu-mem-calc calculate --config my_config.json
```

## 🔧 Working with Configurations

### Create a configuration file

Start with an example:
```bash
cp configs/llama2_7b_deepspeed.json my_config.json
```

Edit `my_config.json` to match your needs:
```json
{
  "model": {
    "name": "my-model",
    "num_parameters": "13B",
    "num_layers": 40,
    "hidden_size": 5120,
    "num_attention_heads": 40,
    "vocab_size": 32000,
    "max_seq_len": 4096
  },
  "training": {
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "optimizer": "adamw",
    "dtype": "bf16"
  },
  "parallelism": {
    "data_parallel_size": 8
  },
  "engine": {
    "type": "deepspeed",
    "zero_stage": 3
  },
  "hardware": {
    "num_gpus": 8,
    "gpu_memory_gb": 80
  }
}
```

### Validate your configuration

```bash
gpu-mem-calc validate my_config.json
```

### Calculate memory

```bash
gpu-mem-calc calculate --config my_config.json
```

## 💡 Tips & Tricks

### 1. Use presets as starting points

List all available presets:
```bash
gpu-mem-calc presets --format table
```

Use a preset and save the config:
```bash
gpu-mem-calc calculate --preset mixtral-8x7b --format json > my_config.json
```

### 2. Start with quick calculations

Before creating detailed configs, get quick estimates:
```bash
gpu-mem-calc quick 70 --gpus 16 --gpu-mem 80 --engine deepspeed
```

### 3. Enable activation checkpointing

If memory is tight, use activation checkpointing in your config:
```json
{
  "training": {
    "activation_checkpointing": 1
  }
}
```

This can save 30-50% of activation memory at the cost of ~20% slower training.

### 4. Experiment with batch size

Larger batches improve GPU utilization but use more memory:
```json
{
  "training": {
    "batch_size": 8,
    "gradient_accumulation_steps": 2
  }
}
```

Total effective batch size = `batch_size × gradient_accumulation_steps × num_gpus`

### 5. Use the web interface for experimentation

The web UI makes it easy to try different configurations:
```bash
python -m gpu_mem_calculator.web.app
```

## 🎓 Next Steps

### Learn about new features

#### 🆕 Inference Memory Calculation
Calculate GPU memory for LLM inference with different engines:
```python
from gpu_mem_calculator.inference.calculator import InferenceMemoryCalculator
from gpu_mem_calculator.core.models import InferenceConfig, InferenceEngineType

calculator = InferenceMemoryCalculator(model_config, inference_config, gpu_config)

# Compare engines
result_vllm = calculator.calculate(InferenceEngineType.VLLM)
result_tgi = calculator.calculate(InferenceEngineType.TGI)
result_trt = calculator.calculate(InferenceEngineType.TENSORRT_LLM)

print(f"vLLM: {result_vllm.total_memory_per_gpu_gb:.2f} GB")
print(f"Max batch: {result_vllm.max_supported_batch_size}")
```

#### 🆕 Multi-Node Training
Calculate network overhead for distributed training:
```python
from gpu_mem_calculator.core.multinode import MultiNodeCalculator
from gpu_mem_calculator.core.models import NodeConfig, InterconnectType

node_config = NodeConfig(
    num_nodes=4,
    gpus_per_node=8,
    interconnect_type=InterconnectType.INFINIBAND,
)

calculator = MultiNodeCalculator(model_config, training_config,
                                  parallelism_config, node_config, engine_config)
overhead = calculator.calculate_network_overhead()

print(f"Network overhead: {overhead.total_overhead_gb:.2f} GB")
print(f"Time overhead: {overhead.estimated_overhead_ms_per_step:.2f} ms/step")
```

#### 🆕 Export Framework Configs
Generate configurations for training frameworks:
```python
from gpu_mem_calculator.exporters.manager import ExportManager, ExportFormat

manager = ExportManager(model_config, training_config, parallelism_config,
                       engine_config, node_config)

# Export to different frameworks
manager.export_to_file(ExportFormat.ACCELERATE, "accelerate_config.yaml")
manager.export_to_file(ExportFormat.LIGHTNING, "lightning_config.json")
manager.export_to_file(ExportFormat.AXOLOTL, "axolotl_config.yml")
```

### Learn more about engines

Read about different training engines:
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)

### Understand the formulas

Check out the [Memory Formulas](../README.md#memory-formulas) section in the README to understand how calculations work.

### Explore advanced features

- Multiple GPU types
- Mixed precision training
- Custom model architectures
- Hybrid parallelism strategies

### Join the community

- ⭐ Star the repo on GitHub
- 💬 Join [GitHub Discussions](https://github.com/George614/gpu-mem-calculator/discussions)
- 🐛 Report issues or request features
- 🤝 Contribute improvements

## 📞 Get Help

Stuck? Here's how to get help:

1. Check the [FAQ](FAQ.md)
2. Read the [README](../README.md)
3. Ask in [GitHub Discussions](https://github.com/George614/gpu-mem-calculator/discussions)
4. Open an [issue](https://github.com/George614/gpu-mem-calculator/issues)

Happy calculating! 🎉
