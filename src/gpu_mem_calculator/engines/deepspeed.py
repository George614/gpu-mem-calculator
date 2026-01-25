"""DeepSpeed ZeRO engine implementation.

Implements memory calculations for DeepSpeed ZeRO stages 1, 2, and 3.
Based on: https://deepspeed.readthedocs.io/en/latest/memory.html
"""

from gpu_mem_calculator.core.formulas import (
    calculate_activation_memory,
    calculate_overhead,
    estimate_largest_layer_params,
)
from gpu_mem_calculator.core.models import (
    MemoryBreakdown,
    MemoryResult,
    OffloadDevice,
)
from gpu_mem_calculator.engines.base import BaseEngine
from gpu_mem_calculator.utils.precision import gb_from_bytes


class DeepSpeedEngine(BaseEngine):
    """DeepSpeed ZeRO memory calculation.

    Implements ZeRO stages:
    - ZeRO-1: Shard optimizer states
    - ZeRO-2: Shard optimizer states + gradients
    - ZeRO-3: Shard optimizer states + gradients + parameters
    """

    def calculate_memory(self) -> MemoryResult:
        """Calculate memory requirements for DeepSpeed ZeRO training.

        Returns:
            MemoryResult with complete memory breakdown
        """
        zero_stage = self.engine_config.zero_stage or 0
        offload_optimizer = self.engine_config.offload_optimizer
        offload_param = self.engine_config.offload_param

        # Get largest layer params for ZeRO-3
        if self.model_config.largest_layer_params is None:
            largest_layer_params = estimate_largest_layer_params(
                hidden_size=self.model_config.hidden_size,
                num_attention_heads=self.model_config.num_attention_heads,
            )
        else:
            largest_layer_params = self.model_config.largest_layer_params

        match zero_stage:
            case 0:
                return self._calculate_zero0()
            case 1:
                return self._calculate_zero1(offload_optimizer)
            case 2:
                return self._calculate_zero2(offload_optimizer)
            case 3:
                return self._calculate_zero3(
                    offload_optimizer,
                    offload_param,
                    largest_layer_params,
                )
            case _:
                # Default to ZeRO-2
                return self._calculate_zero2(offload_optimizer)

    def _calculate_zero0(self) -> MemoryResult:
        """Calculate memory for ZeRO-0 (disabled, same as PyTorch DDP)."""
        # Import here to avoid circular dependency
        from gpu_mem_calculator.engines.pytorch import PyTorchDDPEngine

        # ZeRO-0 is the same as PyTorch DDP
        ddp_engine = PyTorchDDPEngine(
            model_config=self.model_config,
            training_config=self.training_config,
            parallelism_config=self.parallelism_config,
            engine_config=self.engine_config,
            gpu_config=self.gpu_config,
        )
        return ddp_engine.calculate_memory()

    def _calculate_zero1(
        self,
        offload_optimizer: OffloadDevice,
    ) -> MemoryResult:
        """Calculate memory for ZeRO-1 (shard optimizer states).

        ZeRO-1 shards optimizer states across data parallel GPUs.

        Reference: https://deepspeed.readthedocs.io/en/latest/memory.html
        Reference: https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/

        Memory formula:
        - offload_optimizer=cpu: 2 * params (fp16 params only on GPU)
        - offload_optimizer=none: 4 * params (fp16 params + fp32 params) +
          12 * params / num_gpus (sharded optimizer states)

        Note: Optimizer states = 12 bytes per param for Adam/AdamW
        - 4 bytes: FP32 parameter copy
        - 4 bytes: Momentum (FP32)
        - 4 bytes: Variance (FP32)
        """
        num_params = self.model_config.num_parameters
        num_gpus = max(1, self.total_num_gpus)  # Defensive guard against division by zero

        # Model parameters (fp16/bf16 on GPU)
        model_params_gb = gb_from_bytes(num_params * 2)  # FP16/BF16 = 2 bytes

        # Gradients (fp16 on GPU)
        gradients_gb = gb_from_bytes(num_params * 2)

        # Optimizer states (sharded across GPUs, possibly offloaded to CPU)
        # 12 bytes per param for Adam/AdamW (FP32 params copy + momentum + variance)
        if offload_optimizer == OffloadDevice.CPU:
            # Offloaded to CPU, minimal GPU memory for optimizer
            optimizer_gb = 0.0
            cpu_memory_gb = gb_from_bytes(num_params * 12)  # Full optimizer on CPU
        else:
            # Sharded across GPUs: 12 bytes / num_gpus per GPU
            optimizer_gb = gb_from_bytes((num_params * 12) / num_gpus)
            cpu_memory_gb = 0.0

        # Activations (same as baseline)
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

        return self._create_result(breakdown, cpu_memory_gb)

    def _calculate_zero2(
        self,
        offload_optimizer: OffloadDevice,
    ) -> MemoryResult:
        """Calculate memory for ZeRO-2 (shard optimizer + gradients).

        ZeRO-2 shards optimizer states AND gradients across data parallel GPUs.

        Reference: https://deepspeed.readthedocs.io/en/latest/memory.html
        Reference: https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/

        Memory formula:
        - offload_optimizer=cpu: 2 * params (fp16 params) +
          (2 * params / num_gpus) (sharded fp16 grads)
        - offload_optimizer=none: 2 * params (fp16 params) +
          2 * params / num_gpus (sharded fp16 grads) +
          12 * params / num_gpus (sharded optimizer states)

        Note: Unlike ZeRO-1, ZeRO-2 shards gradients across GPUs
        """
        num_params = self.model_config.num_parameters
        num_gpus = max(1, self.total_num_gpus)  # Defensive guard against division by zero

        # Model parameters (fp16/bf16 on GPU) - NOT sharded in ZeRO-2
        model_params_gb = gb_from_bytes(num_params * 2)  # FP16/BF16 = 2 bytes

        # Gradients (fp16 on GPU) - SHARDED in ZeRO-2
        gradients_gb = gb_from_bytes((num_params * 2) / num_gpus)

        # Optimizer states (sharded across GPUs, possibly offloaded to CPU)
        # 12 bytes per param for Adam/AdamW (FP32 params copy + momentum + variance)
        if offload_optimizer == OffloadDevice.CPU:
            # Offloaded to CPU, minimal GPU memory for optimizer
            optimizer_gb = 0.0
            cpu_memory_gb = gb_from_bytes(num_params * 12)  # Full optimizer on CPU
        else:
            # Sharded across GPUs: 12 bytes / num_gpus per GPU
            optimizer_gb = gb_from_bytes((num_params * 12) / num_gpus)
            cpu_memory_gb = 0.0

        # Activations (same as baseline)
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

        return self._create_result(breakdown, cpu_memory_gb)

    def _calculate_zero3(
        self,
        offload_optimizer: OffloadDevice,
        offload_param: OffloadDevice,
        largest_layer_params: int,
    ) -> MemoryResult:
        """Calculate memory for ZeRO-3 (shard params + optimizer + gradients).

        ZeRO-3 shards everything across GPUs.

        Reference: https://deepspeed.readthedocs.io/en/latest/memory.html
        Reference: https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/

        Memory formula:
        - largest_layer_memory = 4 * largest_layer_params (fp16 params + fp16 grads)

        Case 1 (no offload):
          largest_layer_memory + 18 * params / num_gpus
          (where 18 = 16 bytes optimizer states + 2 bytes fp16 params)

        Case 2 (param + optimizer offload to CPU):
          largest_layer_memory (main limit is CPU RAM)

        Case 3 (optimizer offload to CPU only):
          largest_layer_memory + 2 * params / num_gpus

        Note: Optimizer states = 16 bytes per param for Adam/AdamW (FP32)
        - 4 bytes: FP32 parameter copy
        - 4 bytes: Momentum (FP32)
        - 4 bytes: Variance (FP32)
        - 4 bytes: Gradient (FP32 copy for optimizer update)
        """
        num_params = self.model_config.num_parameters
        num_gpus = max(1, self.total_num_gpus)  # Defensive guard against division by zero

        # Largest layer memory (fp16 params + fp16 grads gathered on one GPU)
        largest_layer_memory_gb = gb_from_bytes(largest_layer_params * 4)

        # Calculate memory based on offload configuration
        if offload_param == OffloadDevice.CPU and offload_optimizer == OffloadDevice.CPU:
            # Case 2: Both params and optimizer offloaded to CPU
            # Only need largest layer on GPU at a time
            params_per_gpu_gb = 0.0
            gradients_per_gpu_gb = 0.0
            optimizer_gb = 0.0
            cpu_memory_gb = gb_from_bytes(num_params * 18)  # Full model on CPU
        elif offload_optimizer == OffloadDevice.CPU:
            # Case 3: Only optimizer offloaded to CPU
            params_per_gpu_gb = gb_from_bytes((num_params * 2) / num_gpus)
            gradients_per_gpu_gb = gb_from_bytes((num_params * 2) / num_gpus)
            optimizer_gb = 0.0
            cpu_memory_gb = gb_from_bytes(num_params * 16)  # Optimizer on CPU
        else:
            # Case 1: No offload
            params_per_gpu_gb = gb_from_bytes((num_params * 2) / num_gpus)
            gradients_per_gpu_gb = gb_from_bytes((num_params * 2) / num_gpus)
            optimizer_gb = gb_from_bytes((num_params * 16) / num_gpus)  # FP32
            cpu_memory_gb = 0.0

        # Model params = largest layer for ZeRO-3
        model_params_gb = largest_layer_memory_gb

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
            model_params_gb
            + params_per_gpu_gb
            + gradients_per_gpu_gb
            + optimizer_gb
            + activations_gb
        )
        overhead_gb = calculate_overhead(base_memory)

        # For ZeRO-3, model_params is the largest layer (gathered during compute)
        # params_per_gpu is the sharded parameter storage, counted separately in gradients/optimizer
        # Don't double-count by adding params_per_gpu_gb to model_params_gb
        breakdown = MemoryBreakdown(
            model_params_gb=model_params_gb,  # Just largest layer for ZeRO-3
            gradients_gb=gradients_per_gpu_gb,
            optimizer_states_gb=optimizer_gb,
            activations_gb=activations_gb,
            overhead_gb=overhead_gb,
        )

        return self._create_result(breakdown, cpu_memory_gb)
