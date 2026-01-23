"""TGI (Text Generation Inference) engine memory calculation."""

from gpu_mem_calculator.core.models import (
    InferenceMemoryBreakdown,
    InferenceMemoryResult,
)
from gpu_mem_calculator.inference.base import BaseInferenceEngine


class TGIEngine(BaseInferenceEngine):
    """Text Generation Inference (TGI) engine by HuggingFace.

    TGI is a production-ready inference server with optimized
    attention mechanisms and memory management.
    """

    def calculate_memory(self) -> InferenceMemoryResult:
        """Calculate memory requirements for TGI inference.

        TGI memory breakdown:
        - Model parameters: Loaded in specified dtype
        - KV cache: With optional quantization (INT8/FP8)
        - Activations: Flash Attention optimized
        - Overhead: TGI server, router, preallocation

        Returns:
            InferenceMemoryResult with complete memory breakdown
        """
        batch_size = self.inference_config.batch_size
        tensor_parallel_size = self.inference_config.tensor_parallel_size

        # 1. Model parameters (sharded across tensor parallel GPUs)
        model_params_bytes = self._calculate_model_params_bytes()
        model_params_per_gpu_gb = (model_params_bytes / tensor_parallel_size) / (1024**3)

        # 2. KV cache (with quantization support)
        kv_cache_bytes = self._calculate_kv_cache_bytes(batch_size)
        kv_cache_gb = kv_cache_bytes / (1024**3)

        # 3. Activations (Flash Attention optimized)
        activations_bytes = self._calculate_tgi_activations(batch_size)
        activations_gb = activations_bytes / (1024**3)

        # 4. TGI overhead
        overhead_gb = self._calculate_tgi_overhead()

        breakdown = InferenceMemoryBreakdown(
            model_params_gb=model_params_per_gpu_gb,
            kv_cache_gb=kv_cache_gb,
            activations_gb=activations_gb,
            overhead_gb=overhead_gb,
        )

        return self._create_result(breakdown)

    def _calculate_tgi_activations(self, batch_size: int) -> int:
        """Calculate activation memory for TGI.

        TGI uses Flash Attention which significantly reduces
        activation memory by materializing less attention matrices.

        Args:
            batch_size: Batch size

        Returns:
            Activation memory in bytes
        """
        seq_len = self._get_effective_seq_len()
        hidden_size = self.model_config.hidden_size
        num_layers = self.model_config.num_layers

        # Flash Attention reduces memory by not materializing full attention matrix
        # Rough estimate for activation memory
        bytes_per_value = 2  # FP16/BF16

        # TGI uses optimized kernels: ~40% of standard activation memory
        activation_bytes = batch_size * seq_len * hidden_size * num_layers * bytes_per_value * 0.4

        return int(activation_bytes)

    def _calculate_tgi_overhead(self) -> float:
        """Calculate TGI-specific overhead.

        Includes:
        - TGI server and router
        - nccl communication for tensor parallel
        - Preallocated buffers for efficiency
        - Dynamic batching overhead

        Returns:
            Overhead in GB
        """
        # Base server overhead: ~200MB
        base_overhead_gb = 0.2

        # Tensor parallel communication overhead
        if self.inference_config.tensor_parallel_size > 1:
            # nccl overhead scales with TP size
            tp_overhead_gb = self.inference_config.tensor_parallel_size * 0.05
        else:
            tp_overhead_gb = 0.0

        # Dynamic batching bookkeeping
        batch_overhead_gb = 0.05

        # Preallocated buffers for Flash Attention
        buffer_overhead_gb = 0.1

        return base_overhead_gb + tp_overhead_gb + batch_overhead_gb + buffer_overhead_gb
