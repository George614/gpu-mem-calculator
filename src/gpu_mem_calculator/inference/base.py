"""Base class for inference engine implementations."""

from abc import ABC, abstractmethod

from gpu_mem_calculator.core.models import (
    GPUConfig,
    InferenceConfig,
    InferenceMemoryBreakdown,
    InferenceMemoryResult,
    ModelConfig,
)


class BaseInferenceEngine(ABC):
    """Abstract base class for inference engine memory calculation.

    Each inference engine (vLLM, TGI, TensorRT-LLM, etc.)
    should implement this interface to provide engine-specific
    memory calculations.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        inference_config: InferenceConfig,
        gpu_config: GPUConfig,
    ) -> None:
        """Initialize the inference engine with configuration.

        Args:
            model_config: Model architecture configuration
            inference_config: Inference hyperparameters
            gpu_config: Hardware configuration
        """
        self.model_config = model_config
        self.inference_config = inference_config
        self.gpu_config = gpu_config

    @abstractmethod
    def calculate_memory(self) -> InferenceMemoryResult:
        """Calculate memory requirements for inference.

        This is the main method that should be implemented by each engine.

        Returns:
            InferenceMemoryResult with complete memory breakdown
        """
        pass

    def _check_feasibility(
        self,
        total_memory_per_gpu: float,
    ) -> tuple[bool, float, int | None]:
        """Check if the configuration fits on available GPU.

        Args:
            total_memory_per_gpu: Total memory required per GPU

        Returns:
            Tuple of (fits_on_gpu, utilization_percent, max_batch_size)
        """
        available_memory = (
            self.gpu_config.gpu_memory_gb * self.inference_config.gpu_memory_utilization
        )
        utilization_percent = (total_memory_per_gpu / self.gpu_config.gpu_memory_gb) * 100

        fits_on_gpu = total_memory_per_gpu <= available_memory

        # Find max batch size that fits
        max_batch_size = None
        if fits_on_gpu:
            # Try to estimate max batch size
            # This is a simplified heuristic
            current_batch = self.inference_config.batch_size
            # Defensive guard: ensure batch size is at least 1 to avoid division by zero
            if current_batch <= 0:
                max_batch_size = 1
            else:
                overhead_per_token = total_memory_per_gpu / current_batch
                if overhead_per_token > 0:
                    potential_max_batch = int(available_memory / overhead_per_token)
                    max_batch_size = max(1, potential_max_batch)
                else:
                    max_batch_size = 1

        return fits_on_gpu, utilization_percent, max_batch_size

    def _create_result(
        self,
        breakdown: InferenceMemoryBreakdown,
    ) -> InferenceMemoryResult:
        """Create an InferenceMemoryResult from breakdown.

        Args:
            breakdown: Memory breakdown by component

        Returns:
            Complete InferenceMemoryResult
        """
        total_memory_per_gpu = breakdown.total_memory_gb
        num_gpus = self.inference_config.tensor_parallel_size
        total_memory_all_gpus = total_memory_per_gpu * num_gpus

        fits_on_gpu, utilization_percent, max_batch_size = self._check_feasibility(
            total_memory_per_gpu
        )

        # Estimate throughput (simplified heuristic)
        # NOTE: This is a very rough estimate. Actual throughput varies significantly based on:
        # - GPU hardware (A100, H100, etc.)
        # - Model architecture and size
        # - Sequence length and batch size
        # - Inference optimizations (Flash Attention, kernel fusion, etc.)
        # Use this value with caution - it's primarily for relative comparisons.
        estimated_throughput = None
        if fits_on_gpu:
            batch_size = max(1, self.inference_config.batch_size)  # Defensive guard
            tokens_per_batch = batch_size * self._get_effective_seq_len()
            # Rough estimate: ~50ms per batch (highly variable)
            estimated_throughput = tokens_per_batch / 0.05  # tokens per second

        return InferenceMemoryResult(
            total_memory_per_gpu_gb=total_memory_per_gpu,
            total_memory_all_gpus_gb=total_memory_all_gpus,
            breakdown=breakdown,
            fits_on_gpu=fits_on_gpu,
            memory_utilization_percent=utilization_percent,
            max_supported_batch_size=max_batch_size,
            estimated_throughput_tokens_per_sec=estimated_throughput,
        )

    def _get_effective_seq_len(self) -> int:
        """Get effective sequence length for inference."""
        return self.inference_config.max_seq_len or self.model_config.max_seq_len

    def _get_kv_cache_bytes_per_token(self) -> int:
        """Calculate KV cache bytes per token.

        Returns:
            Bytes per token for KV cache (considering quantization)
        """
        # Base: 2 * num_layers * num_heads * head_dim * bytes_per_value
        # For each token, we store K and V for each layer
        num_layers = self.model_config.num_layers
        num_heads = self.model_config.num_attention_heads
        head_dim = self.model_config.hidden_size // num_heads

        # Determine bytes per value based on quantization
        quantization = self.inference_config.kv_cache_quantization
        bytes_per_value = {
            "none": 2,  # FP16/BF16
            "int8": 1,
            "fp8": 1,
            "int4": 0.5,
        }[quantization.value]

        # KV cache = 2 (K and V) * num_layers * num_heads * head_dim * bytes_per_value
        kv_bytes_per_token = 2 * num_layers * num_heads * head_dim * bytes_per_value

        return int(kv_bytes_per_token)

    def _calculate_model_params_bytes(self) -> int:
        """Calculate model parameter memory in bytes.

        Returns:
            Bytes needed for model parameters
        """
        dtype_bytes = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "int8": 1,
            "int4": 0.5,
        }

        # Assume model is loaded in BF16/FP16 for inference
        dtype = "bf16"
        num_params = self.model_config.num_parameters

        return int(num_params * dtype_bytes[dtype])

    def _calculate_kv_cache_bytes(self, batch_size: int) -> int:
        """Calculate KV cache memory in bytes.

        Args:
            batch_size: Batch size to calculate for

        Returns:
            Bytes needed for KV cache
        """
        if not self.inference_config.use_kv_cache:
            return 0

        seq_len = self._get_effective_seq_len()
        kv_bytes_per_token = self._get_kv_cache_bytes_per_token()

        # KV cache = batch_size * seq_len * kv_bytes_per_token
        return batch_size * seq_len * kv_bytes_per_token
