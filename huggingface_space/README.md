---
title: GPU Memory Calculator
emoji: 🎮
colorFrom: blue
colorTo: purple
sdk: docker
pinned: true
license: mit
tags: [llm, gpu, deep-learning, pytorch, training, inference, memory-calculator, deepspeed, megatron, fsdp, vllm, quantization, machine-learning, ai, tools]
---

# 🎮 GPU Memory Calculator for LLM Training & Inference

**Instantly calculate GPU memory requirements for training and running Large Language Models.** Plan your infrastructure, avoid OOM errors, and optimize costs before you start.

[![GitHub Stars](https://img.shields.io/github/stars/George614/gpu-mem-calculator?style=social)](https://github.com/George614/gpu-mem-calculator)
[![GitHub Issues](https://img.shields.io/github/issues/George614/gpu-mem-calculator)](https://github.com/George614/gpu-mem-calculator/issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Why Use This Tool?

- **💰 Save Money** - Know exactly what GPUs you need before spending thousands
- **⚡ Avoid OOM** - Validate your config fits in memory before training
- **📊 Compare Strategies** - DeepSpeed vs Megatron vs FSDP at a glance
- **🎯 Plan Infrastructure** - From 7B to 175B+ parameter models
- **⚙️ Export Configs** - Generate working configs for your training framework

## ✨ Features

### Training Memory Calculation
Calculate memory for all major training frameworks:
- **PyTorch DDP** - Baseline distributed training
- **DeepSpeed ZeRO** (Stages 0-3) with CPU/NVMe offloading
- **Megatron-LM** - Tensor + Pipeline parallelism
- **PyTorch FSDP** - Fully sharded data parallel
- **Megatron + DeepSpeed** - Hybrid approach

### Inference Memory Estimation
Optimize your deployment with:
- **HuggingFace Transformers** - Baseline inference
- **vLLM** - PagedAttention optimization
- **TGI** - Text Generation Inference
- **TensorRT-LLM** - Maximum throughput
- **SGLang** - RadixAttention caching

### Smart Features
- 🎯 **Model Presets** - LLaMA 2, GPT-3, Mixtral, GLM, Qwen, DeepSeek-MoE
- 📦 **Export Configs** - Accelerate, Lightning, Axolotl, DeepSpeed, YAML, JSON
- 🔢 **Batch Optimizer** - Auto-find max batch size for your hardware
- 🌐 **Multi-Node** - Calculate network overhead for distributed training
- 💾 **KV Cache** - Quantization options (INT4/INT8/FP8/None)

## 🎯 Supported Models

| Model | Parameters | Use Case |
|-------|-----------|----------|
| LLaMA 2 | 7B, 13B, 70B | General purpose |
| GPT-3 | 175B | Large scale training |
| Mixtral 8x7B | 47B | Mixture of Experts |
| GLM-4 | 9B - 355B | Chinese/English |
| Qwen MoE | 2.7B | Efficient inference |
| DeepSeek-MoE | 16B | sparse training |

## 📖 How to Use

1. **Select a Model** - Choose from presets or enter custom parameters
2. **Pick Your Engine** - Training (DeepSpeed/Megatron/FSDP) or Inference (vLLM/TGI/SGLang)
3. **Configure** - Adjust batch size, GPUs, precision, offloading
4. **Calculate** - Get instant memory breakdown
5. **Export** - Generate working configs for your framework

## 💡 Example Use Cases

- **"Can I train a 7B model on 4x A100s?"** → Calculate and find out
- **"What's the max batch size for DeepSpeed ZeRO-3?"** → Batch optimizer tells you
- **"vLLM vs TGI - which uses less memory?"** → Compare instantly
- **"How many GPUs for 175B with Megatron?"** → Plan your cluster

## 🔗 Links & Resources

- **[GitHub Repository](https://github.com/George614/gpu-mem-calculator)** - Star us on GitHub! ⭐
- **[Full Documentation](https://github.com/George614/gpu-mem-calculator#readme)** - Complete guide
- **[Report Issues](https://github.com/George614/gpu-mem-calculator/issues)** - Bug reports & feature requests
- **[Contributing Guide](https://github.com/George614/gpu-mem-calculator/blob/main/CONTRIBUTING.md)** - Pull requests welcome!

## 📚 Technical Details

Built with:
- **FastAPI** - High-performance web framework
- **Pydantic** - Data validation and settings
- **Python 3.12** - Latest Python for maximum performance

Formulas verified against:
- [EleutherAI Transformer Math](https://blog.eleuther.ai/transformer-math/)
- [Microsoft DeepSpeed ZeRO](https://www.microsoft.com/en-us/research/blog/zero-deepspeed/)
- [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

## 📊 License

MIT License - Free for commercial and personal use.

---

**Made with ❤️ by the AI community**

[![GitHub stars](https://img.shields.io/github/stars/George614/gpu-mem-calculator?style=flat-square&logo=github&label=Star%20on%20GitHub)](https://github.com/George614/gpu-mem-calculator)

