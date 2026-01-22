"""
Example: Comparing different training engines

This example compares memory requirements across different training engines
for the same model configuration.
"""

from gpu_mem_calculator.core.calculator import GPUMemoryCalculator
from gpu_mem_calculator.core.models import (
    ModelConfig,
    TrainingConfig,
    ParallelismConfig,
    EngineConfig,
    GPUConfig,
)


def calculate_for_engine(engine_type, zero_stage=None):
    """Calculate memory for a specific engine configuration"""
    
    # Same model configuration for all engines
    model_config = ModelConfig(
        name="llama2-13b",
        num_parameters=13_000_000_000,
        num_layers=40,
        hidden_size=5120,
        num_attention_heads=40,
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
    
    # Configure engine based on type
    if engine_type == "deepspeed":
        engine_config = EngineConfig(
            type="deepspeed",
            zero_stage=zero_stage,
        )
    elif engine_type == "megatron":
        engine_config = EngineConfig(
            type="megatron",
        )
        parallelism_config.tensor_parallel_size = 2
        parallelism_config.pipeline_parallel_size = 1
    elif engine_type == "fsdp":
        engine_config = EngineConfig(
            type="fsdp",
            sharding_strategy="FULL_SHARD",
        )
    else:  # pytorch (DDP)
        engine_config = EngineConfig(
            type="pytorch",
        )
    
    gpu_config = GPUConfig(
        num_gpus=8,
        gpu_memory_gb=80,
    )
    
    calculator = GPUMemoryCalculator(
        model_config=model_config,
        training_config=training_config,
        parallelism_config=parallelism_config,
        engine_config=engine_config,
        gpu_config=gpu_config,
    )
    
    return calculator.calculate()


def main():
    """Compare different training engines"""
    
    print("=" * 70)
    print("Comparing Training Engines for LLaMA 2 13B")
    print("Hardware: 8× A100 80GB GPUs")
    print("=" * 70)
    print()
    
    # Test different configurations
    configs = [
        ("PyTorch DDP", "pytorch", None),
        ("DeepSpeed ZeRO-1", "deepspeed", 1),
        ("DeepSpeed ZeRO-2", "deepspeed", 2),
        ("DeepSpeed ZeRO-3", "deepspeed", 3),
        ("FSDP", "fsdp", None),
        ("Megatron-LM", "megatron", None),
    ]
    
    results = []
    for name, engine, zero_stage in configs:
        result = calculate_for_engine(engine, zero_stage)
        results.append((name, result))
    
    # Print comparison table
    print(f"{'Engine':<20} {'Memory/GPU':<15} {'Fits?':<10} {'Utilization':<15}")
    print("-" * 70)
    
    for name, result in results:
        fits = "✓ Yes" if result.fits_on_gpu else "✗ No"
        print(
            f"{name:<20} "
            f"{result.total_memory_per_gpu_gb:>6.2f} GB      "
            f"{fits:<10} "
            f"{result.memory_utilization_percent:>6.1f}%"
        )
    
    print("=" * 70)
    print()
    
    # Find most memory efficient
    valid_results = [(name, r) for name, r in results if r.fits_on_gpu]
    if valid_results:
        most_efficient = min(valid_results, key=lambda x: x[1].total_memory_per_gpu_gb)
        print(f"🏆 Most memory efficient: {most_efficient[0]}")
        print(f"   Uses only {most_efficient[1].total_memory_per_gpu_gb:.2f} GB per GPU")
    
    # Find configurations that don't fit
    invalid_results = [(name, r) for name, r in results if not r.fits_on_gpu]
    if invalid_results:
        print()
        print("⚠️  Configurations that don't fit:")
        for name, result in invalid_results:
            print(f"   - {name}: Needs {result.total_memory_per_gpu_gb:.2f} GB per GPU")


if __name__ == "__main__":
    main()
