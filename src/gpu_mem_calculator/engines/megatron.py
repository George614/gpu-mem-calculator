"""Megatron-LM engine implementation.

Implements memory calculations for Megatron-LM with tensor, pipeline,
and sequence parallelism.
"""

from gpu_mem_calculator.core.formulas import (
    calculate_activation_memory,
    calculate_overhead,
    calculate_optimizer_memory,
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
from gpu_mem_calculator.utils.precision import gb_from_bytes


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
        dp_size = self.parallelism_config.data_parallel_size
        seq_parallel = self.parallelism_config.sequence_parallel

        # Effective number of GPUs for model sharding
        model_parallel_size = tp_size * pp_size

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
        activations_gb = self._calculate_megatron_activations(
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

    def _calculate_megatron_activations(
        self,
        tp_size: int,
        pp_size: int,
        seq_parallel: bool,
    ) -> float:
        """Calculate activation memory for Megatron-LM.

        Megatron-LM activations are affected by parallelism strategy:
        - Tensor parallelism: splits hidden dimension
        - Pipeline parallelism: only current stage's activations
        - Sequence parallelism: splits sequence dimension

        Args:
            tp_size: Tensor parallelism size
            pp_size: Pipeline parallelism size
            seq_parallel: Whether sequence parallelism is enabled

        Returns:
            Activation memory in GB
        """
        from gpu_mem_calculator.core.formulas import calculate_activation_memory

        # Base activation memory
        base_activations = calculate_activation_memory(
            batch_size=self.training_config.batch_size,
            seq_len=self.model_config.max_seq_len,
            hidden_size=self.model_config.hidden_size,
            num_layers=self.model_config.num_layers,
            num_attention_heads=self.model_config.num_attention_heads,
            tensor_parallel_size=tp_size,
            activation_checkpointing=self.training_config.activation_checkpointing,
            moe_enabled=self.model_config.moe_enabled,
            num_experts=self.model_config.num_experts,
            top_k=self.model_config.top_k,
            expert_intermediate_size=self.model_config.expert_intermediate_size,
        )

        # Adjust for pipeline parallelism
        # Each PP stage only holds num_layers / pp_size layers
        pp_factor = 1.0 / pp_size

        # Adjust for sequence parallelism
        # If enabled, splits sequence dimension across TP GPUs
        if seq_parallel and tp_size > 1:
            seq_factor = 1.0 / tp_size
        else:
            seq_factor = 1.0

        return base_activations * pp_factor * seq_factor


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
        from gpu_mem_calculator.engines.deepspeed import DeepSpeedEngine

        # First calculate activation memory using Megatron-LM approach
        tp_size = self.parallelism_config.tensor_parallel_size
        pp_size = self.parallelism_config.pipeline_parallel_size
        seq_parallel = self.parallelism_config.sequence_parallel

        activations_gb = self._calculate_megatron_activations(
            tp_size=tp_size,
            pp_size=pp_size,
            seq_parallel=seq_parallel,
        )

        # For parameters, gradients, optimizer - use DeepSpeed ZeRO logic
        # But account for tensor parallelism (parameters are already split by TP)
        tp_size = self.parallelism_config.tensor_parallel_size
        params_per_gpu = self.model_config.num_parameters / tp_size

        zero_stage = self.engine_config.zero_stage or 2
        offload_optimizer = self.engine_config.offload_optimizer

        # Model parameters (sharded by TP, then possibly by ZeRO)
        if zero_stage >= 3:
            # ZeRO-3 shards further
            dp_size = self.parallelism_config.data_parallel_size
            model_params_gb = gb_from_bytes((params_per_gpu * 2) / dp_size)
        else:
            # ZeRO-0/1/2 keeps parameters on each TP GPU
            model_params_gb = gb_from_bytes(params_per_gpu * 2)

        # Gradients
        if zero_stage >= 2:
            dp_size = self.parallelism_config.data_parallel_size
            gradients_gb = gb_from_bytes((params_per_gpu * 2) / dp_size)
        else:
            gradients_gb = gb_from_bytes(params_per_gpu * 2)

        # Optimizer states
        if offload_optimizer.value == "cpu":
            optimizer_gb = 0.0
        else:
            if zero_stage >= 1:
                dp_size = self.parallelism_config.data_parallel_size
                optimizer_gb = gb_from_bytes((params_per_gpu * 8) / dp_size)
            else:
                optimizer_gb = gb_from_bytes(params_per_gpu * 8)

        # Overhead
        base_memory = model_params_gb + gradients_gb + optimizer_gb + activations_gb
        overhead_gb = gb_from_bytes(base_memory * 0.2)

        breakdown = MemoryBreakdown(
            model_params_gb=model_params_gb,
            gradients_gb=gradients_gb,
            optimizer_states_gb=optimizer_gb,
            activations_gb=activations_gb,
            overhead_gb=overhead_gb,
        )

        return self._create_result(breakdown)

    def _calculate_megatron_activations(
        self,
        tp_size: int,
        pp_size: int,
        seq_parallel: bool,
    ) -> float:
        """Calculate activation memory for Megatron-LM."""
        from gpu_mem_calculator.core.formulas import calculate_activation_memory

        # Base activation memory
        base_activations = calculate_activation_memory(
            batch_size=self.training_config.batch_size,
            seq_len=self.model_config.max_seq_len,
            hidden_size=self.model_config.hidden_size,
            num_layers=self.model_config.num_layers,
            num_attention_heads=self.model_config.num_attention_heads,
            tensor_parallel_size=tp_size,
            activation_checkpointing=self.training_config.activation_checkpointing,
            moe_enabled=self.model_config.moe_enabled,
            num_experts=self.model_config.num_experts,
            top_k=self.model_config.top_k,
            expert_intermediate_size=self.model_config.expert_intermediate_size,
        )

        # Adjust for pipeline parallelism
        pp_factor = 1.0 / pp_size

        # Adjust for sequence parallelism
        if seq_parallel and tp_size > 1:
            seq_factor = 1.0 / tp_size
        else:
            seq_factor = 1.0

        return base_activations * pp_factor * seq_factor
