"""HuggingFace Transformers inference engine memory calculation."""

from gpu_mem_calculator.core.models import (
    InferenceMemoryBreakdown,
    InferenceMemoryResult,
)
from gpu_mem_calculator.inference.base import BaseInferenceEngine


class HuggingFaceEngine(BaseInferenceEngine):
    """HuggingFace Transformers inference engine.

    Standard HuggingFace inference with optimizations like
    Flash Attention and torch.compile.
    """

    def calculate_memory(self) -> InferenceMemoryResult:
        """Calculate memory requirements for HF Transformers inference.

        HF Transformers memory breakdown:
        - Model parameters: Standard loading
        - KV cache: Standard implementation
        - Activations: Depends on optimization level
        - Overhead: PyTorch runtime, model loading

        Returns:
            InferenceMemoryResult with complete memory breakdown
        """
        batch_size = self.inference_config.batch_size
        tensor_parallel_size = self.inference_config.tensor_parallel_size

        # 1. Model parameters (sharded if using tensor parallel)
        model_params_bytes = self._calculate_model_params_bytes()
        model_params_per_gpu_gb = (model_params_bytes / tensor_parallel_size) / (1024**3)

        # 2. KV cache (standard implementation)
        kv_cache_bytes = self._calculate_kv_cache_bytes(batch_size)
        kv_cache_gb = kv_cache_bytes / (1024**3)

        # 3. Activations (standard PyTorch)
        activations_bytes = self._calculate_hf_activations(batch_size)
        activations_gb = activations_bytes / (1024**3)

        # 4. HF overhead
        overhead_gb = self._calculate_hf_overhead()

        breakdown = InferenceMemoryBreakdown(
            model_params_gb=model_params_per_gpu_gb,
            kv_cache_gb=kv_cache_gb,
            activations_gb=activations_gb,
            overhead_gb=overhead_gb,
        )

        return self._create_result(breakdown)

    def _calculate_hf_activations(self, batch_size: int) -> int:
        """Calculate activation memory for HF Transformers.

        Standard PyTorch activation memory without kernel fusion.

        Args:
            batch_size: Batch size

        Returns:
            Activation memory in bytes
        """
        seq_len = self._get_effective_seq_len()
        hidden_size = self.model_config.hidden_size
        num_layers = self.model_config.num_layers

        # Standard activation memory (forward pass only)
        bytes_per_value = 2  # FP16/BF16

        activation_bytes = batch_size * seq_len * hidden_size * num_layers * bytes_per_value

        return int(activation_bytes)

    def _calculate_hf_overhead(self) -> float:
        """Calculate HF Transformers-specific overhead.

        Includes:
        - PyTorch runtime
        - Model loading overhead
        - Autograd graph (even for inference, some overhead remains)

        Returns:
            Overhead in GB
        """
        # Base PyTorch overhead: ~150MB
        base_overhead_gb = 0.15

        # Model loading overhead: ~50MB
        loading_overhead_gb = 0.05

        return base_overhead_gb + loading_overhead_gb
