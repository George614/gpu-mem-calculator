"""Megatron-LM engine implementation.

Implements memory calculations for Megatron-LM with tensor, pipeline,
and sequence parallelism.

Reference: https://github.com/NVIDIA/Megatron-LM
Reference: https://arxiv.org/abs/1909.08053
Reference: https://blog.eleuther.ai/transformer-math/
"""

from gpu_mem_calculator.core.formulas import (
    calculate_activation_memory,
    calculate_gradient_memory,
    calculate_optimizer_memory,
    calculate_overhead,
    calculate_parameter_memory,
)
from gpu_mem_calculator.core.models import (
    MemoryBreakdown,
    MemoryResult,
    ModelConfig,
    TrainingConfig,
)
from gpu_mem_calculator.engines.base import BaseEngine
from gpu_mem_calculator.utils.precision import gb_from_bytes


def calculate_megatron_activations(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    tp_size: int,
    pp_size: int,
    seq_parallel: bool,
) -> float:
    """Calculate activation memory for Megatron-LM parallelism.

    This is a shared utility function used by both MegatronLMEngine
    and MegatronDeepSpeedEngine.

    Megatron-LM activations are affected by parallelism strategy:
    - Tensor parallelism: splits hidden dimension
    - Pipeline parallelism: only current stage's activations
    - Sequence parallelism: splits sequence dimension

    Args:
        model_config: Model configuration
        training_config: Training configuration
        tp_size: Tensor parallelism size
        pp_size: Pipeline parallelism size
        seq_parallel: Whether sequence parallelism is enabled

    Returns:
        Activation memory in GB
    """
    # Defensive guards
    tp_size = max(1, tp_size)
    pp_size = max(1, pp_size)

    # Base activation memory
    base_activations = calculate_activation_memory(
        batch_size=training_config.batch_size,
        seq_len=model_config.max_seq_len,
        hidden_size=model_config.hidden_size,
        num_layers=model_config.num_layers,
        num_attention_heads=model_config.num_attention_heads,
        tensor_parallel_size=tp_size,
        activation_checkpointing=training_config.activation_checkpointing,
        moe_enabled=model_config.moe_enabled,
        num_experts=model_config.num_experts,
        top_k=model_config.top_k,
        expert_intermediate_size=model_config.expert_intermediate_size,
    )

    # Adjust for pipeline parallelism
    # Each PP stage only holds num_layers / pp_size layers
    pp_factor = 1.0 / pp_size

    # Adjust for sequence parallelism
    # Sequence parallelism splits sequence dimension across TP GPUs
    # Note: Sequence parallelism requires tensor parallelism (tp_size > 1)
    if seq_parallel:
        if tp_size > 1:
            seq_factor = 1.0 / tp_size
        else:
            # SP requires TP > 1, but we don't raise an error here
            # to allow for exploratory configurations. In production,
            # this should be validated at configuration time.
            seq_factor = 1.0  # No sequence parallelism effect without TP
    else:
        seq_factor = 1.0

    return base_activations * pp_factor * seq_factor


class MegatronLMEngine(BaseEngine):
    """Megatron-LM memory calculation.

    Megatron-LM uses tensor parallelism to split individual layers across GPUs,
    and optionally pipeline parallelism to split layers across GPUs.
    """

    def calculate_memory(self) -> MemoryResult:
        """Calculate memory requirements for Megatron-LM training.

        Megatron-LM memory characteristics:
        - Parameters are sharded across tensor parallel GPUs
        - Gradients are sharded across tensor parallel GPUs
        - Optimizer states can be sharded or replicated
        - Activations depend on tensor/pipeline/sequence parallelism

        Returns:
            MemoryResult with complete memory breakdown
        """
        tp_size = self.parallelism_config.tensor_parallel_size
        pp_size = self.parallelism_config.pipeline_parallel_size
        seq_parallel = self.parallelism_config.sequence_parallel

        # 1. Model parameters (sharded by tensor parallelism)
        # Each TP GPU holds 1/tp of the parameters
        params_per_gpu = self.model_config.num_parameters / tp_size
        model_params_gb = calculate_parameter_memory(
            num_params=int(params_per_gpu),
            dtype=self.training_config.dtype.value,
        )

        # 2. Gradients (sharded by tensor parallelism)
        gradients_gb = calculate_gradient_memory(
            num_params=int(params_per_gpu),
            dtype=self.training_config.dtype.value,
        )

        # 3. Optimizer states
        # In Megatron-LM, optimizer states are typically sharded similarly to parameters
        # for tensor parallelism, but this can vary based on configuration
        optimizer_gb = calculate_optimizer_memory(
            num_params=int(params_per_gpu),
            optimizer=self.training_config.optimizer.value,
        )

        # 4. Activations
        # Activations are affected by:
        # - Tensor parallelism: splits activations across TP GPUs
        # - Pipeline parallelism: only holds activations for current stage
        # - Sequence parallelism: splits sequence dimension
        activations_gb = calculate_megatron_activations(
            model_config=self.model_config,
            training_config=self.training_config,
            tp_size=tp_size,
            pp_size=pp_size,
            seq_parallel=seq_parallel,
        )

        # 5. Overhead
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


class MegatronDeepSpeedEngine(BaseEngine):
    """Megatron-LM + DeepSpeed combined engine.

    This combines Megatron-LM's tensor/pipeline parallelism with
    DeepSpeed ZeRO's optimizer/gradient sharding.
    """

    def calculate_memory(self) -> MemoryResult:
        """Calculate memory for Megatron-LM + DeepSpeed.

        This uses:
        - Megatron-LM for tensor/pipeline parallelism and activation memory
        - DeepSpeed ZeRO for optimizer/gradient sharding

        Returns:
            MemoryResult with complete memory breakdown
        """
        # Import DeepSpeed engine

        # First calculate activation memory using Megatron-LM approach
        tp_size = self.parallelism_config.tensor_parallel_size
        pp_size = self.parallelism_config.pipeline_parallel_size
        seq_parallel = self.parallelism_config.sequence_parallel

        activations_gb = calculate_megatron_activations(
            model_config=self.model_config,
            training_config=self.training_config,
            tp_size=tp_size,
            pp_size=pp_size,
            seq_parallel=seq_parallel,
        )

        # For parameters, gradients, optimizer - use DeepSpeed ZeRO logic
        # combined with Megatron tensor/pipeline parallelism
        tp_size = max(1, self.parallelism_config.tensor_parallel_size)
        pp_size = max(1, self.parallelism_config.pipeline_parallel_size)
        dp_size = max(1, self.parallelism_config.data_parallel_size)
        num_params = self.model_config.num_parameters

        zero_stage = self.engine_config.zero_stage or 2
        offload_optimizer = self.engine_config.offload_optimizer

        # Model parameters (sharded by TP and PP via Megatron, then by DP via ZeRO)
        # Megatron shards model across TP × PP GPUs
        # ZeRO-3 further shards across DP GPUs
        if zero_stage >= 3:
            # ZeRO-3: parameters are sharded across all parallel dimensions
            # Total sharding = TP × PP × DP
            total_sharding = tp_size * pp_size * dp_size
            model_params_gb = gb_from_bytes((num_params * 2) / total_sharding)
        else:
            # ZeRO-0/1/2: parameters sharded only by TP × PP (Megatron sharding)
            megatron_sharding = tp_size * pp_size
            model_params_gb = gb_from_bytes((num_params * 2) / megatron_sharding)

        # Gradients (sharded by TP × PP from Megatron)
        # ZeRO-2+ further shards gradients across DP
        if zero_stage >= 2:
            total_grad_sharding = tp_size * pp_size * dp_size
            gradients_gb = gb_from_bytes((num_params * 2) / total_grad_sharding)
        else:
            megatron_sharding = tp_size * pp_size
            gradients_gb = gb_from_bytes((num_params * 2) / megatron_sharding)

        # Optimizer states (12 bytes per param for Adam/AdamW in FP32)
        if offload_optimizer.value == "cpu":
            optimizer_gb = 0.0
        else:
            if zero_stage >= 1:
                # ZeRO-1+ shards optimizer across DP, Megatron across TP × PP
                total_opt_sharding = tp_size * pp_size * dp_size
                optimizer_gb = gb_from_bytes((num_params * 12) / total_opt_sharding)
            else:
                megatron_sharding = tp_size * pp_size
                optimizer_gb = gb_from_bytes((num_params * 12) / megatron_sharding)

        # Overhead (already in GB, so calculate directly)
        base_memory = model_params_gb + gradients_gb + optimizer_gb + activations_gb
        overhead_gb = base_memory * 0.2  # 20% overhead

        breakdown = MemoryBreakdown(
            model_params_gb=model_params_gb,
            gradients_gb=gradients_gb,
            optimizer_states_gb=optimizer_gb,
            activations_gb=activations_gb,
            overhead_gb=overhead_gb,
        )

        return self._create_result(breakdown)
