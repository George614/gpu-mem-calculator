"""Memory calculation formulas.

This module contains the fundamental formulas for calculating GPU memory
requirements for LLM training.
"""

from dataclasses import dataclass

from gpu_mem_calculator.utils.precision import get_precision_from_dtype


@dataclass
class Precision:
    """Precision information for a data type.

    This is re-exported from utils.precision for convenience.
    """

    name: str
    bits_per_param: int
    bytes_per_param: float
    is_integer: bool = False


def calculate_parameter_memory(
    num_params: int,
    dtype: str,
    num_gpus: int = 1,
) -> float:
    """Calculate memory in GB for model parameters.

    Args:
        num_params: Number of model parameters
        dtype: Data type (e.g., "fp32", "fp16", "bf16", "int8", "int4")
        num_gpus: Number of GPUs for distribution

    Returns:
        Memory in GB
    """
    from gpu_mem_calculator.utils.precision import gb_from_params

    # Parameters are distributed across GPUs in data parallel training
    # But for tensor/pipeline parallel, each GPU holds a portion
    # We'll handle parallelism in the engine implementations
    return gb_from_params(num_params, dtype)


def calculate_gradient_memory(
    num_params: int,
    dtype: str,
) -> float:
    """Calculate memory in GB for gradients.

    Gradients are typically stored in the same precision as parameters
    for training (though updated in FP32).

    Args:
        num_params: Number of model parameters
        dtype: Data type for gradients

    Returns:
        Memory in GB
    """
    from gpu_mem_calculator.utils.precision import gb_from_params

    # Gradients are same size as parameters during training
    return gb_from_params(num_params, dtype)


def calculate_optimizer_memory(
    num_params: int,
    optimizer: str,
) -> float:
    """Calculate memory in GB for optimizer states.

    Args:
        num_params: Number of model parameters
        optimizer: Optimizer type (adam, adamw, sgd, adamw_8bit)

    Returns:
        Memory in GB (for FP32 optimizer states)
    """
    from gpu_mem_calculator.utils.precision import gb_from_bytes

    # Optimizer states are typically stored in FP32
    bytes_per_param = 4.0  # FP32

    match optimizer.lower():
        case "adam" | "adamw":
            # Adam: momentum (4 bytes) + variance (4 bytes) = 8 bytes per param
            optimizer_bytes_per_param = 8.0
        case "adamw_8bit":
            # 8-bit Adam: ~2 bytes per param (quantized states)
            optimizer_bytes_per_param = 2.0
        case "sgd":
            # SGD: just momentum (4 bytes) if using momentum, 0 if not
            # Assuming momentum is used
            optimizer_bytes_per_param = 4.0
        case _:
            # Default to Adam
            optimizer_bytes_per_param = 8.0

    total_bytes = num_params * optimizer_bytes_per_param
    return gb_from_bytes(total_bytes)


def calculate_activation_memory(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    tensor_parallel_size: int = 1,
    activation_checkpointing: int = 0,
) -> float:
    """Calculate approximate memory in GB for activations.

    This is a rough estimate based on transformer architecture.
    Actual activation memory depends on many factors.

    Formula approximation from:
    https://arxiv.org/abs/2204.13323

    Args:
        batch_size: Batch size per GPU
        seq_len: Sequence length
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        tensor_parallel_size: Tensor parallelism degree
        activation_checkpointing: Checkpointing level (0-4)

    Returns:
        Memory in GB
    """
    from gpu_mem_calculator.utils.precision import gb_from_bytes

    # Approximate activation memory per token
    # This includes attention outputs, MLP activations, layer norms, etc.
    # Rough estimate: ~10-20 bytes per token per layer per hidden dimension

    bytes_per_token_per_layer = hidden_size * 16  # Rough estimate

    # Total activation memory
    total_bytes = (
        batch_size
        * seq_len
        * num_layers
        * bytes_per_token_per_layer
        / tensor_parallel_size
    )

    # Adjust for activation checkpointing
    # Level 0: No checkpointing (100% memory)
    # Level 1: Checkpoint attention output (~80% memory)
    # Level 2: Checkpoint attention input (~60% memory)
    # Level 3: Checkpoint more (~40% memory)
    # Level 4: Full checkpointing (~20% memory)
    checkpoint_factors = [1.0, 0.8, 0.6, 0.4, 0.2]
    checkpoint_factor = checkpoint_factors[min(activation_checkpointing, 4)]

    total_bytes *= checkpoint_factor

    return gb_from_bytes(total_bytes)


def calculate_overhead(
    total_memory: float,
    overhead_factor: float = 0.2,
) -> float:
    """Calculate additional memory overhead.

    This accounts for CUDA context, fragmentation, temporary buffers, etc.

    Args:
        total_memory: Total calculated memory in GB
        overhead_factor: Fraction to add for overhead (default 20%)

    Returns:
        Overhead memory in GB
    """
    return total_memory * overhead_factor


def estimate_largest_layer_params(
    hidden_size: int,
    num_attention_heads: int,
    intermediate_size: int | None = None,
) -> int:
    """Estimate the largest layer parameters for ZeRO-3 calculations.

    The largest layer is typically the MLP layer or attention projection.

    Args:
        hidden_size: Hidden dimension size
        num_attention_heads: Number of attention heads
        intermediate_size: MLP intermediate size (default 4 * hidden_size)

    Returns:
        Estimated number of parameters in the largest layer
    """
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size

    # MLP layer: hidden_size * intermediate_size * 2 (for up and down projections)
    mlp_params = hidden_size * intermediate_size * 2

    # Attention output projection: hidden_size * hidden_size
    attn_params = hidden_size * hidden_size

    return max(mlp_params, attn_params)
