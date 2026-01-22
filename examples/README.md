# Examples

This directory contains practical examples demonstrating how to use the GPU Memory Calculator.

## Available Examples

### 1. Basic Usage (`basic_usage.py`)

Demonstrates the fundamental usage of the calculator API:
- Creating model, training, and hardware configurations
- Calculating memory requirements
- Interpreting results
- Getting optimization recommendations

Run it:
```bash
python examples/basic_usage.py
```

### 2. Compare Engines (`compare_engines.py`)

Compares memory requirements across different training engines:
- PyTorch DDP
- DeepSpeed ZeRO (stages 1, 2, 3)
- Megatron-LM
- PyTorch FSDP

Useful for deciding which engine to use for your model.

Run it:
```bash
python examples/compare_engines.py
```

## Running Examples

Make sure you have the package installed:
```bash
pip install -e .
```

Then run any example:
```bash
cd /path/to/gpu-mem-calculator
python examples/basic_usage.py
```

## Creating Your Own

Use these examples as templates for your own calculations. Key steps:

1. **Define your model**:
   ```python
   model_config = ModelConfig(
       name="my-model",
       num_parameters=7_000_000_000,
       num_layers=32,
       hidden_size=4096,
       ...
   )
   ```

2. **Configure training**:
   ```python
   training_config = TrainingConfig(
       batch_size=4,
       dtype="bf16",
       optimizer="adamw",
   )
   ```

3. **Set up hardware**:
   ```python
   gpu_config = GPUConfig(
       num_gpus=8,
       gpu_memory_gb=80,
   )
   ```

4. **Calculate**:
   ```python
   calculator = GPUMemoryCalculator(...)
   result = calculator.calculate()
   ```

## More Resources

- [Getting Started Guide](../docs/GETTING_STARTED.md)
- [FAQ](../docs/FAQ.md)
- [Main README](../README.md)
