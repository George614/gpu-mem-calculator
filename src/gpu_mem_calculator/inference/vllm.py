"""vLLM inference engine memory calculation."""

from gpu_mem_calculator.core.models import (
    InferenceMemoryBreakdown,
    InferenceMemoryResult,
)
from gpu_mem_calculator.inference.base import BaseInferenceEngine


class VLLMEngine(BaseInferenceEngine):
    """vLLM inference engine with PagedAttention memory management.

    vLLM uses PagedAttention to efficiently manage KV cache memory
    with block-based allocation.
    """

    def calculate_memory(self) -> InferenceMemoryResult:
        """Calculate memory requirements for vLLM inference.

        vLLM memory breakdown:
        - Model parameters: Loaded once, shared across all GPUs
        - KV cache: Managed in blocks with PagedAttention
        - Activations: Temporary during forward pass
        - Overhead: vLLM scheduler, worker overhead, block tables

        Returns:
            InferenceMemoryResult with complete memory breakdown
        """
        batch_size = self.inference_config.batch_size
        tensor_parallel_size = self.inference_config.tensor_parallel_size

        # 1. Model parameters (sharded across tensor parallel GPUs)
        model_params_bytes = self._calculate_model_params_bytes()
        model_params_per_gpu_gb = (model_params_bytes / tensor_parallel_size) / (1024**3)

        # 2. KV cache with PagedAttention (block-based allocation)
        kv_cache_bytes = self._calculate_vllm_kv_cache(batch_size)
        kv_cache_gb = kv_cache_bytes / (1024**3)

        # 3. Activations (temporary, per batch)
        activations_bytes = self._calculate_activations(batch_size)
        activations_gb = activations_bytes / (1024**3)

        # 4. vLLM overhead (scheduler, block manager, etc.)
        overhead_gb = self._calculate_vllm_overhead()

        breakdown = InferenceMemoryBreakdown(
            model_params_gb=model_params_per_gpu_gb,
            kv_cache_gb=kv_cache_gb,
            activations_gb=activations_gb,
            overhead_gb=overhead_gb,
        )

        return self._create_result(breakdown)

    def _calculate_vllm_kv_cache(self, batch_size: int) -> int:
        """Calculate KV cache memory for vLLM with PagedAttention.

        vLLM uses block-based KV cache management, which is more efficient
        than contiguous allocation. Each block contains multiple token slots.

        Args:
            batch_size: Batch size

        Returns:
            KV cache memory in bytes
        """
        block_size = self.inference_config.block_size or 16

        # Calculate total tokens needed
        seq_len = self._get_effective_seq_len()
        total_tokens = batch_size * seq_len

        # Calculate number of blocks needed
        num_blocks = (total_tokens + block_size - 1) // block_size

        # Add 20% buffer for dynamic allocation during generation
        num_blocks = int(num_blocks * 1.2)

        # KV cache memory with block allocation
        kv_bytes_per_token = self._get_kv_cache_bytes_per_token()
        total_kv_bytes = num_blocks * block_size * kv_bytes_per_token

        return total_kv_bytes

    def _calculate_activations(self, batch_size: int) -> int:
        """Calculate activation memory for vLLM.

        vLLM optimizes activation memory with kernel fusion
        and efficient attention implementation.

        Args:
            batch_size: Batch size

        Returns:
            Activation memory in bytes
        """
        seq_len = self._get_effective_seq_len()
        hidden_size = self.model_config.hidden_size
        num_layers = self.model_config.num_layers

        # Simplified activation calculation
        # vLLM uses kernel fusion to reduce activation memory
        # Rough estimate: batch * seq * hidden * layers * bytes_per_value
        bytes_per_value = 2  # FP16/BF16

        # Base activation memory
        activation_bytes = (
            batch_size
            * seq_len
            * hidden_size
            * num_layers
            * bytes_per_value
            * 2  # Forward pass only (no backward)
        )

        # vLLM optimization: ~50% reduction with kernel fusion
        activation_bytes = int(activation_bytes * 0.5)

        return activation_bytes

    def _calculate_vllm_overhead(self) -> float:
        """Calculate vLLM-specific overhead.

        Includes:
        - Scheduler memory
        - Block table management
        - Worker process overhead
        - CUDA graphs and preallocated buffers

        Returns:
            Overhead in GB
        """
        # Base overhead: ~100-200MB for scheduler and block manager
        base_overhead_gb = 0.15

        # Additional overhead for block tables
        # Each block entry: 8 bytes (pointer)
        block_size = self.inference_config.block_size or 16
        seq_len = self._get_effective_seq_len()
        batch_size = self.inference_config.batch_size

        num_blocks = (batch_size * seq_len + block_size - 1) // block_size * 1.2
        block_table_bytes = num_blocks * 8
        block_table_gb = block_table_bytes / (1024**3)

        # Preallocated buffers for CUDA kernels (~50MB)
        buffer_overhead_gb = 0.05

        return base_overhead_gb + block_table_gb + buffer_overhead_gb
