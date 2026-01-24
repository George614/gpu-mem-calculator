"""SGLang inference engine memory calculation."""

from gpu_mem_calculator.core.models import (
    InferenceMemoryBreakdown,
    InferenceMemoryResult,
)
from gpu_mem_calculator.inference.base import BaseInferenceEngine


class SGLangEngine(BaseInferenceEngine):
    """SGLang inference engine with RadixAttention memory management.

    SGLang uses RadixAttention to efficiently manage KV cache memory
    with tree-based cache sharing and chunked prefill.
    """

    def calculate_memory(self) -> InferenceMemoryResult:
        """Calculate memory requirements for SGLang inference.

        SGLang memory breakdown:
        - Model parameters: Loaded once, shared across all GPUs
        - KV cache: Managed with RadixAttention (tree-based sharing)
        - Activations: Temporary during forward pass
        - Overhead: Scheduler, RadixCache tree, worker overhead

        Returns:
            InferenceMemoryResult with complete memory breakdown
        """
        batch_size = self.inference_config.batch_size
        tensor_parallel_size = self.inference_config.tensor_parallel_size

        # 1. Model parameters (sharded across tensor parallel GPUs)
        model_params_bytes = self._calculate_model_params_bytes()
        model_params_per_gpu_gb = (model_params_bytes / tensor_parallel_size) / (1024**3)

        # 2. KV cache with RadixAttention
        kv_cache_bytes = self._calculate_sglang_kv_cache(batch_size)
        kv_cache_gb = kv_cache_bytes / (1024**3)

        # 3. Activations (temporary, per batch)
        activations_bytes = self._calculate_activations(batch_size)
        activations_gb = activations_bytes / (1024**3)

        # 4. SGLang overhead (RadixCache, scheduler, etc.)
        overhead_gb = self._calculate_sglang_overhead()

        breakdown = InferenceMemoryBreakdown(
            model_params_gb=model_params_per_gpu_gb,
            kv_cache_gb=kv_cache_gb,
            activations_gb=activations_gb,
            overhead_gb=overhead_gb,
        )

        return self._create_result(breakdown)

    def _calculate_sglang_kv_cache(self, batch_size: int) -> int:
        """Calculate KV cache memory for SGLang with RadixAttention.

        SGLang's RadixAttention uses a tree-based structure for KV cache
        sharing, which is more memory-efficient than traditional approaches.

        Args:
            batch_size: Batch size

        Returns:
            KV cache memory in bytes
        """
        if not self.inference_config.use_kv_cache:
            return 0

        # Use chunk size for memory estimation (default: 8192)
        chunk_size = self.inference_config.chunk_size or 8192

        # Calculate effective sequence length
        seq_len = self._get_effective_seq_len()

        # RadixCache shares common prefixes across requests
        # This provides significant memory savings for concurrent requests
        max_running_requests = self.inference_config.max_running_requests or batch_size * 4

        # Base KV cache memory
        kv_bytes_per_token = self._get_kv_cache_bytes_per_token()

        # RadixCache provides ~30% memory savings from prefix sharing
        cache_sharing_factor = 0.7 if not self.inference_config.disable_radix_cache else 1.0

        # Calculate total tokens with chunking
        total_tokens = batch_size * min(seq_len, chunk_size)

        # Apply RadixCache sharing factor
        total_kv_bytes = total_tokens * kv_bytes_per_token * cache_sharing_factor

        # Account for max running requests (concurrent requests share cache)
        concurrent_requests_factor = min(max_running_requests / batch_size, 2.0)
        total_kv_bytes = int(total_kv_bytes * concurrent_requests_factor)

        return total_kv_bytes

    def _calculate_activations(self, batch_size: int) -> int:
        """Calculate activation memory for SGLang.

        SGLang uses chunked prefill and optimized attention kernels
        to reduce activation memory.

        Args:
            batch_size: Batch size

        Returns:
            Activation memory in bytes
        """
        seq_len = self._get_effective_seq_len()
        chunk_size = self.inference_config.chunk_size or 8192

        # Chunked prefill processes sequences in chunks
        effective_seq_len = min(seq_len, chunk_size)

        hidden_size = self.model_config.hidden_size
        num_layers = self.model_config.num_layers

        # Base activation memory
        bytes_per_value = 2  # FP16/BF16

        activation_bytes = (
            batch_size
            * effective_seq_len
            * hidden_size
            * num_layers
            * bytes_per_value
            * 2  # Forward pass only (no backward)
        )

        # SGLang optimizations:
        # - Chunked prefill: ~40% reduction
        # - FlashInfer/triton kernels: ~30% reduction
        # - torch.compile if enabled: ~20% additional reduction
        activation_bytes = int(activation_bytes * 0.6)  # 40% reduction from chunked prefill

        if self.inference_config.enable_torch_compile:
            activation_bytes = int(activation_bytes * 0.8)  # Additional 20% reduction

        return activation_bytes

    def _calculate_sglang_overhead(self) -> float:
        """Calculate SGLang-specific overhead.

        Includes:
        - RadixCache tree structure
        - Scheduler memory
        - P2P communication buffers (if enabled)
        - Custom all-reduce buffers (if not disabled)
        - Multi-LoRA serving overhead (if enabled)

        Returns:
            Overhead in GB
        """
        # Base overhead: ~100-200MB for scheduler and RadixCache tree
        base_overhead_gb = 0.15

        # RadixCache tree structure overhead
        # Tree nodes: ~32 bytes per node for metadata
        if not self.inference_config.disable_radix_cache:
            seq_len = self._get_effective_seq_len()
            batch_size = self.inference_config.batch_size
            num_nodes = batch_size * seq_len // 100  # Rough estimate
            tree_overhead_gb = (num_nodes * 32) / (1024**3)
        else:
            tree_overhead_gb = 0.0

        # P2P attention overhead (if enabled)
        if self.inference_config.enable_p2p:
            p2p_overhead_gb = 0.1  # ~100MB for P2P buffers
        else:
            p2p_overhead_gb = 0.0

        # Custom all-reduce overhead (if not disabled)
        if not self.inference_config.disable_custom_all_reduce:
            all_reduce_overhead_gb = 0.08  # ~80MB for all-reduce buffers
        else:
            all_reduce_overhead_gb = 0.0

        # Multi-LoRA serving overhead
        if self.inference_config.multi_lora_enabled:
            lora_overhead_gb = 0.2  # ~200MB for LoRA adapters
        else:
            lora_overhead_gb = 0.0

        # Speculative decoding overhead
        if self.inference_config.speculative_algo != "default":
            speculate_overhead_gb = 0.15  # ~150MB for draft models
        else:
            speculate_overhead_gb = 0.0

        return (
            base_overhead_gb
            + tree_overhead_gb
            + p2p_overhead_gb
            + all_reduce_overhead_gb
            + lora_overhead_gb
            + speculate_overhead_gb
        )
