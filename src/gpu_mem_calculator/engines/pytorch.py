"""PyTorch DDP (Distributed Data Parallel) engine implementation.

This is the baseline implementation without any memory optimizations.

Reference: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
Reference: https://blog.eleuther.ai/transformer-math/
"""

from gpu_mem_calculator.core.formulas import (
    calculate_activation_memory,
    calculate_optimizer_memory,
    calculate_overhead,
    calculate_parameter_memory,
    calculate_gradient_memory,
)
from gpu_mem_calculator.core.models import (
    EngineConfig,
    GPUConfig,
    MemoryBreakdown,
    MemoryResult,
    ModelConfig,
    ParallelismConfig,
    TrainingConfig,
)
from gpu_mem_calculator.engines.base import BaseEngine


class PyTorchDDPEngine(BaseEngine):
    """PyTorch DDP memory calculation.

    DDP replicates the model on each GPU, so memory is not sharded.
    Each GPU holds a full copy of the model, gradients, and optimizer states.
    """

    def calculate_memory(self) -> MemoryResult:
        """Calculate memory requirements for PyTorch DDP training.

        For DDP:
        - Model parameters: Full model on each GPU
        - Gradients: Full gradients on each GPU
        - Optimizer states: Full optimizer states on each GPU (FP32)
        - Activations: Batch size dependent, split by data parallel

        Returns:
            MemoryResult with complete memory breakdown
        """
        # 1. Model parameters (in the specified dtype)
        model_params_gb = calculate_parameter_memory(
            num_params=self.model_config.num_parameters,
            dtype=self.training_config.dtype.value,
        )

        # 2. Gradients (same precision as parameters for mixed precision)
        gradients_gb = calculate_gradient_memory(
            num_params=self.model_config.num_parameters,
            dtype=self.training_config.dtype.value,
        )

        # 3. Optimizer states (always FP32 for Adam/AdamW)
        optimizer_gb = calculate_optimizer_memory(
            num_params=self.model_config.num_parameters,
            optimizer=self.training_config.optimizer.value,
        )

        # 4. Activations (depends on batch size and model architecture)
        activations_gb = calculate_activation_memory(
            batch_size=self.training_config.batch_size,
            seq_len=self.model_config.max_seq_len,
            hidden_size=self.model_config.hidden_size,
            num_layers=self.model_config.num_layers,
            num_attention_heads=self.model_config.num_attention_heads,
            tensor_parallel_size=self.parallelism_config.tensor_parallel_size,
            activation_checkpointing=self.training_config.activation_checkpointing,
            moe_enabled=self.model_config.moe_enabled,
            num_experts=self.model_config.num_experts,
            top_k=self.model_config.top_k,
            expert_intermediate_size=self.model_config.expert_intermediate_size,
        )

        # 5. Calculate overhead
        base_memory = model_params_gb + gradients_gb + optimizer_gb + activations_gb
        overhead_gb = calculate_overhead(base_memory)

        # Create breakdown
        breakdown = MemoryBreakdown(
            model_params_gb=model_params_gb,
            gradients_gb=gradients_gb,
            optimizer_states_gb=optimizer_gb,
            activations_gb=activations_gb,
            overhead_gb=overhead_gb,
        )

        return self._create_result(breakdown)
