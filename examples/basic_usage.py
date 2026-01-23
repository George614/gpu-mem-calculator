"""
Example: Basic usage of GPU Memory Calculator Python API

This example demonstrates how to use the calculator programmatically
to estimate memory requirements for training LLaMA 2 7B.
"""

from gpu_mem_calculator.core.calculator import GPUMemoryCalculator
from gpu_mem_calculator.core.models import (
    ModelConfig,
    TrainingConfig,
    ParallelismConfig,
    EngineConfig,
    GPUConfig,
)


def main():
    """Calculate memory for LLaMA 2 7B with DeepSpeed ZeRO-3"""
    
    # Configure the model
    model_config = ModelConfig(
        name="llama2-7b",
        num_parameters=7_000_000_000,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        vocab_size=32000,
        max_seq_len=4096,
    )
    
    # Configure training parameters
    training_config = TrainingConfig(
        batch_size=4,
        gradient_accumulation_steps=4,
        dtype="bf16",
        optimizer="adamw",
        activation_checkpointing=1,  # Enable activation checkpointing
    )
    
    # Configure parallelism
    parallelism_config = ParallelismConfig(
        data_parallel_size=8,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )
    
    # Configure training engine (DeepSpeed ZeRO-3)
    engine_config = EngineConfig(
        type="deepspeed",
        zero_stage=3,
        offload_optimizer="none",  # Try "cpu" if GPU memory is limited
        offload_param="none",
    )
    
    # Configure hardware
    gpu_config = GPUConfig(
        num_gpus=8,
        gpu_memory_gb=80,  # A100 80GB
    )
    
    # Create calculator and compute memory
    calculator = GPUMemoryCalculator(
        model_config=model_config,
        training_config=training_config,
        parallelism_config=parallelism_config,
        engine_config=engine_config,
        gpu_config=gpu_config,
    )
    
    result = calculator.calculate()
    
    # Print results
    print("=" * 60)
    print(f"Model: {model_config.name}")
    print(f"Total Parameters: {model_config.num_parameters / 1e9:.1f}B")
    print("=" * 60)
    print(f"Memory per GPU: {result.total_memory_per_gpu_gb:.2f} GB")
    print(f"Total memory across all GPUs: {result.total_memory_per_gpu_gb * gpu_config.num_gpus:.2f} GB")
    print(f"Fits on GPU: {'✓ Yes' if result.fits_on_gpu else '✗ No'}")
    print(f"GPU Utilization: {result.memory_utilization_percent:.1f}%")
    print("=" * 60)
    print("\nMemory Breakdown:")
    print(f"  Parameters: {result.breakdown.model_params_gb:.2f} GB")
    print(f"  Gradients: {result.breakdown.gradients_gb:.2f} GB")
    print(f"  Optimizer States: {result.breakdown.optimizer_states_gb:.2f} GB")
    print(f"  Activations: {result.breakdown.activations_gb:.2f} GB")
    print("=" * 60)
    
    # Recommendations
    if not result.fits_on_gpu:
        print("\n⚠️  Configuration doesn't fit in GPU memory!")
        print("Suggestions:")
        print("  1. Enable CPU offloading (set offload_optimizer='cpu')")
        print("  2. Reduce batch size")
        print("  3. Add more GPUs")
        print("  4. Enable activation checkpointing (if not already enabled)")
    else:
        if result.memory_utilization_percent < 50:
            print("\n💡 GPU utilization is low. You can:")
            print("  1. Increase batch size for faster training")
            print("  2. Use fewer GPUs to reduce costs")
        else:
            print("\n✅ Good GPU utilization!")


if __name__ == "__main__":
    main()
