"""TensorRT-LLM inference engine memory calculation."""

from gpu_mem_calculator.core.models import (
    InferenceMemoryBreakdown,
    InferenceMemoryResult,
)
from gpu_mem_calculator.inference.base import BaseInferenceEngine


class TensorRTLLMEngine(BaseInferenceEngine):
    """TensorRT-LLM inference engine with optimized inference kernels.

    TensorRT-LLM provides highly optimized inference through:
    - Weight-only quantization (INT4/INT8)
    - Fused attention kernels
    - In-flight batching
    - Custom CUDA kernels
    """

    def calculate_memory(self) -> InferenceMemoryResult:
        """Calculate memory requirements for TensorRT-LLM inference.

        TensorRT-LLM memory breakdown:
        - Model parameters: Can be quantized (INT4/INT8/FP8)
        - KV cache: With INT8 quantization support
        - Activations: Minimal with fused kernels
        - Overhead: TensorRT runtime, engine workspace

        Returns:
            InferenceMemoryResult with complete memory breakdown
        """
        batch_size = self.inference_config.batch_size
        tensor_parallel_size = self.inference_config.tensor_parallel_size

        # 1. Model parameters (quantized options available)
        model_params_bytes = self._calculate_model_params_bytes()
        model_params_per_gpu_gb = (model_params_bytes / tensor_parallel_size) / (1024**3)

        # 2. KV cache (INT8 optimized)
        kv_cache_bytes = self._calculate_kv_cache_bytes(batch_size)
        kv_cache_gb = kv_cache_bytes / (1024**3)

        # 3. Activations (minimal with fused kernels)
        activations_bytes = self._calculate_tensorrt_activations(batch_size)
        activations_gb = activations_bytes / (1024**3)

        # 4. TensorRT-LLM overhead
        overhead_gb = self._calculate_tensorrt_overhead()

        breakdown = InferenceMemoryBreakdown(
            model_params_gb=model_params_per_gpu_gb,
            kv_cache_gb=kv_cache_gb,
            activations_gb=activations_gb,
            overhead_gb=overhead_gb,
        )

        return self._create_result(breakdown)

    def _calculate_tensorrt_activations(self, batch_size: int) -> int:
        """Calculate activation memory for TensorRT-LLM.

        TensorRT-LLM uses heavily fused kernels which minimize
        activation memory.

        Args:
            batch_size: Batch size

        Returns:
            Activation memory in bytes
        """
        seq_len = self._get_effective_seq_len()
        hidden_size = self.model_config.hidden_size
        num_layers = self.model_config.num_layers

        # TensorRT-LLM fuses many operations, reducing activation memory
        # Rough estimate: ~30% of standard activation memory
        bytes_per_value = 2  # FP16/BF16

        activation_bytes = batch_size * seq_len * hidden_size * num_layers * bytes_per_value * 0.3

        return int(activation_bytes)

    def _calculate_tensorrt_overhead(self) -> float:
        """Calculate TensorRT-LLM-specific overhead.

        Includes:
        - TensorRT runtime
        - Engine workspace for temporary buffers
        - In-flight batching bookkeeping
        - Custom kernel overhead

        Returns:
            Overhead in GB
        """
        # TensorRT runtime: ~100MB
        runtime_overhead_gb = 0.1

        # Engine workspace: scales with model size
        workspace_overhead_gb = 0.2

        # In-flight batching structures
        batching_overhead_gb = 0.05

        return runtime_overhead_gb + workspace_overhead_gb + batching_overhead_gb
