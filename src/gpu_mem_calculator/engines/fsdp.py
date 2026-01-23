"""FSDP (Fully Sharded Data Parallel) engine implementation.

Implements memory calculations for PyTorch FSDP.

Reference: https://pytorch.org/docs/stable/fsdp.html
Reference: https://blog.eleuther.ai/transformer-math/
"""

from gpu_mem_calculator.core.formulas import (
    calculate_activation_memory,
    calculate_overhead,
    estimate_largest_layer_params,
)
from gpu_mem_calculator.core.models import (
    MemoryBreakdown,
    MemoryResult,
)
from gpu_mem_calculator.engines.base import BaseEngine
from gpu_mem_calculator.utils.precision import gb_from_bytes


class FSDPEngine(BaseEngine):
    """PyTorch FSDP memory calculation.

    FSDP shards model parameters, gradients, and optimizer states
    across data parallel GPUs, similar to DeepSpeed ZeRO-3.

    Sharding strategies:
    - NO_SHARD: Equivalent to DDP (no sharding)
    - SHARD_GRAD_OP: Shard gradients and optimizer states (like ZeRO-2)
    - FULL_SHARD: Shard everything (like ZeRO-3)
    """

    def calculate_memory(self) -> MemoryResult:
        """Calculate memory requirements for FSDP training.

        Returns:
            MemoryResult with complete memory breakdown
        """
        sharding_strategy = self.engine_config.sharding_strategy

        # Get largest layer params for FULL_SHARD
        if self.model_config.largest_layer_params is None:
            largest_layer_params = estimate_largest_layer_params(
                hidden_size=self.model_config.hidden_size,
                num_attention_heads=self.model_config.num_attention_heads,
            )
        else:
            largest_layer_params = self.model_config.largest_layer_params

        match sharding_strategy:
            case "no_shard":
                return self._calculate_no_shard()
            case "shard_grad_op":
                return self._calculate_shard_grad_op()
            case "full_shard":
                return self._calculate_full_shard(largest_layer_params)
            case _:
                # Default to full shard
                return self._calculate_full_shard(largest_layer_params)

    def _calculate_no_shard(self) -> MemoryResult:
        """Calculate memory for NO_SHARD (same as DDP).

        No sharding - each GPU holds a full copy of the model.
        """
        # Import PyTorch DDP engine
        from gpu_mem_calculator.engines.pytorch import PyTorchDDPEngine

        ddp_engine = PyTorchDDPEngine(
            model_config=self.model_config,
            training_config=self.training_config,
            parallelism_config=self.parallelism_config,
            engine_config=self.engine_config,
            gpu_config=self.gpu_config,
        )
        return ddp_engine.calculate_memory()

    def _calculate_shard_grad_op(self) -> MemoryResult:
        """Calculate memory for SHARD_GRAD_OP.

        Shards gradients and optimizer states across GPUs.
        Similar to DeepSpeed ZeRO-2.

        Reference: https://pytorch.org/tutorials/intermediate/FSDP_advanced.html
        Reference: https://blog.eleuther.ai/transformer-math/

        Memory formula:
        - Model parameters: Full model on each GPU (not sharded)
        - Gradients: Sharded across GPUs
        - Optimizer states: Sharded across GPUs (12 bytes per param for Adam/AdamW)

        Note: Optimizer states = 12 bytes per param for Adam/AdamW
        - 4 bytes: FP32 parameter copy
        - 4 bytes: Momentum (FP32)
        - 4 bytes: Variance (FP32)
        """
        num_params = self.model_config.num_parameters
        num_gpus = self.total_num_gpus

        # Model parameters (full model on each GPU)
        model_params_gb = gb_from_bytes(num_params * 2)  # FP16/BF16

        # Gradients (sharded)
        gradients_gb = gb_from_bytes((num_params * 2) / num_gpus)

        # Optimizer states (sharded) - 12 bytes per param for Adam/AdamW
        optimizer_gb = gb_from_bytes((num_params * 12) / num_gpus)  # FP32

        # Activations
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

        # Overhead
        base_memory = model_params_gb + gradients_gb + optimizer_gb + activations_gb
        overhead_gb = calculate_overhead(base_memory)

        breakdown = MemoryBreakdown(
            model_params_gb=model_params_gb,
            gradients_gb=gradients_gb,
            optimizer_states_gb=optimizer_gb,
            activations_gb=activations_gb,
            overhead_gb=overhead_gb,
        )

        return self._create_result(breakdown)

    def _calculate_full_shard(self, largest_layer_params: int) -> MemoryResult:
        """Calculate memory for FULL_SHARD.

        Shards parameters, gradients, and optimizer states.
        Similar to DeepSpeed ZeRO-3.

        Reference: https://pytorch.org/tutorials/intermediate/FSDP_advanced.html
        Reference: https://blog.eleuther.ai/transformer-math/

        Memory formula:
        - Largest layer: 4 * largest_layer_params (fp16 params + fp16 grads)
        - Remaining parameters and gradients: Sharded across GPUs (2 bytes fp16 each)
        - Optimizer states: Sharded across GPUs (12 bytes per param for Adam/AdamW in FP32)

        Total per GPU: largest_layer_memory + 2 * params / num_gpus +
                       2 * params / num_gpus + 12 * params / num_gpus
                    = largest_layer_memory + 16 * params / num_gpus

        Note: FSDP typically uses 12 bytes for optimizer states (not 16 like DeepSpeed ZeRO-3)
        because FSDP doesn't keep an additional FP32 gradient copy in the optimizer states.
        """
        num_params = self.model_config.num_parameters
        num_gpus = self.total_num_gpus

        # Largest layer memory (fp16 params + fp16 grads gathered during compute)
        largest_layer_memory_gb = gb_from_bytes(largest_layer_params * 4)

        # Sharded parameters (fp16)
        params_per_gpu_gb = gb_from_bytes((num_params * 2) / num_gpus)

        # Sharded gradients (fp16)
        gradients_per_gpu_gb = gb_from_bytes((num_params * 2) / num_gpus)

        # Sharded optimizer states (FP32 for Adam/AdamW)
        # 12 bytes per param: 4 bytes fp32 params copy + 4 bytes momentum + 4 bytes variance
        optimizer_per_gpu_gb = gb_from_bytes((num_params * 12) / num_gpus)

        # Model params in breakdown: largest layer (gathered) + sharded params
        # This represents the total parameter memory on each GPU
        model_params_gb = largest_layer_memory_gb + params_per_gpu_gb

        # Activations
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

        # Overhead
        base_memory = (
            largest_layer_memory_gb
            + params_per_gpu_gb
            + gradients_per_gpu_gb
            + optimizer_per_gpu_gb
            + activations_gb
        )
        overhead_gb = calculate_overhead(base_memory)

        breakdown = MemoryBreakdown(
            model_params_gb=model_params_gb,
            gradients_gb=gradients_per_gpu_gb,
            optimizer_states_gb=optimizer_per_gpu_gb,
            activations_gb=activations_gb,
            overhead_gb=overhead_gb,
        )

        return self._create_result(breakdown)
