# FAQ - Frequently Asked Questions

## General Questions

### What is GPU Memory Calculator?

GPU Memory Calculator is a tool that helps you estimate the GPU memory requirements for training large language models (LLMs). It supports multiple training frameworks including PyTorch DDP, DeepSpeed ZeRO, Megatron-LM, and FSDP.

### Who should use this tool?

- **Researchers** planning LLM training experiments
- **ML Engineers** optimizing training infrastructure
- **Students** learning about distributed training
- **Anyone** who wants to avoid costly OOM (Out of Memory) errors

### Is this tool free?

Yes! GPU Memory Calculator is completely free and open-source under the MIT license.

## Installation & Setup

### What are the system requirements?

- Python 3.10 or higher
- Any operating system (Linux, macOS, Windows)
- No GPU required to run the calculator itself

### How do I install the tool?

```bash
pip install git+https://github.com/George614/gpu-mem-calculator.git
```

For web interface support:
```bash
pip install -e ".[web]"
```

### Why can't I import the package after installation?

Make sure you're using the correct package name:
```python
from gpu_mem_calculator.core.calculator import GPUMemoryCalculator
```

## Usage Questions

### How accurate are the calculations?

The calculations are based on established formulas from:
- DeepSpeed official documentation
- NVIDIA Megatron-LM research
- EleutherAI's Transformer Math 101
- Academic papers on distributed training

Actual memory usage may vary by 5-10% depending on framework implementation details, but the estimates are generally reliable for planning purposes.

### What models are supported?

The calculator works with any transformer-based model. We provide presets for:
- LLaMA 2 (7B, 13B, 70B)
- GPT-3 (175B)
- Mixtral 8x7B
- GLM-4 variants
- Qwen MoE models
- DeepSeek MoE

You can also define custom models by specifying model parameters.

### Can I use this for inference?

Currently, the tool focuses on training memory calculations. Inference memory is generally much lower and less complex. We may add inference support in future versions.

### Which training engine should I use?

- **PyTorch DDP**: Simple, good for smaller models on a few GPUs
- **DeepSpeed ZeRO**: Best for very large models, excellent memory efficiency
- **Megatron-LM**: Best for models that benefit from model parallelism
- **FSDP**: PyTorch's native solution, good middle ground

### What's the difference between ZeRO stages?

- **ZeRO-1**: Shards optimizer states across GPUs (saves ~4x memory)
- **ZeRO-2**: Shards optimizer states + gradients (saves ~8x memory)
- **ZeRO-3**: Shards everything including model parameters (saves ~16x memory)

Higher stages save more memory but add communication overhead.

## Configuration Questions

### How do I create a configuration file?

See the examples in the `configs/` directory:
```bash
cat configs/llama2_7b_deepspeed.json
```

Or use the web interface to generate configurations interactively.

### What does "activation checkpointing" do?

Activation checkpointing (also called gradient checkpointing) reduces memory by recomputing activations during the backward pass instead of storing them. This trades compute for memory.

### Should I enable CPU offloading?

CPU offloading can help when GPU memory is limited, but it significantly slows down training. Only use it if:
- You can't fit the model in GPU memory otherwise
- Training speed is not critical
- You have fast CPU-GPU interconnects

## Troubleshooting

### The calculator says my config won't fit, but I want to try anyway

The calculator provides estimates. You can:
1. Enable activation checkpointing
2. Use a higher ZeRO stage
3. Enable CPU offloading
4. Reduce batch size
5. Use gradient accumulation instead of larger batches
6. Add more GPUs

### The web interface won't start

Make sure you installed the web dependencies:
```bash
pip install -e ".[web]"
```

Then start the server:
```bash
uvicorn gpu_mem_calculator.web.app:app --reload
```

### My actual training uses more/less memory than predicted

Several factors can affect actual memory usage:
- Framework overhead (PyTorch/DeepSpeed internal buffers)
- CUDA kernels and temporary tensors
- Communication buffers for distributed training
- Logging and monitoring tools
- Mixed precision training implementation details

The calculator provides base estimates; always leave 10-20% headroom for overhead.

## Contributing

### How can I contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines. Common contributions:
- Adding new model presets
- Improving documentation
- Reporting bugs
- Suggesting features
- Submitting bug fixes

### Can I add support for my custom training framework?

Yes! You can extend the calculator by:
1. Creating a new engine calculator class
2. Implementing the memory formulas
3. Adding tests
4. Submitting a pull request

### I found a bug, what should I do?

Please [open an issue](https://github.com/George614/gpu-mem-calculator/issues/new/choose) with:
- Description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Your configuration file or command

## Technical Questions

### How is optimizer memory calculated?

For Adam/AdamW:
- 4 bytes: FP32 copy of parameters
- 4 bytes: Momentum (first moment)
- 4 bytes: Variance (second moment)
- Total: 12 bytes per parameter

For SGD:
- 4 bytes: Momentum
- Total: 4 bytes per parameter (if momentum enabled)

### How are activations calculated?

We use a heuristic approximation:
```
activation_memory = batch_size × seq_len × hidden_size × num_layers × bytes_per_activation
```

With activation checkpointing, this is divided by the checkpoint factor.

### Why does ZeRO-3 have a "largest layer" component?

ZeRO-3 shards model parameters but needs to gather a full layer during forward/backward passes. The largest layer determines the peak memory usage.

### What about communication overhead?

The calculator focuses on memory requirements. Communication overhead affects training speed but not memory usage (except for temporary communication buffers, which are typically small).

## Getting Help

### Where can I get more help?

- 📖 Read the [README](../README.md)
- 💬 Ask in [GitHub Discussions](https://github.com/George614/gpu-mem-calculator/discussions)
- 🐛 Report bugs in [Issues](https://github.com/George614/gpu-mem-calculator/issues)
- 📧 Contact maintainers through GitHub

### Is there a community chat?

We primarily use GitHub Discussions for community interaction. Feel free to start a discussion for questions, ideas, or just to share your experiences!

---

**Don't see your question here?** [Ask in Discussions](https://github.com/George614/gpu-mem-calculator/discussions) or [open an issue](https://github.com/George614/gpu-mem-calculator/issues)!
