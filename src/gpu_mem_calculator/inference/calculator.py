"""Main inference memory calculator.

Orchestrates the inference memory calculation by selecting the appropriate
inference engine and aggregating results.
"""

from gpu_mem_calculator.core.models import (
    GPUConfig,
    InferenceConfig,
    InferenceEngineType,
    InferenceMemoryResult,
    ModelConfig,
)
from gpu_mem_calculator.inference.huggingface import HuggingFaceEngine
from gpu_mem_calculator.inference.sglang import SGLangEngine
from gpu_mem_calculator.inference.tensorrt_llm import TensorRTLLMEngine
from gpu_mem_calculator.inference.tgi import TGIEngine
from gpu_mem_calculator.inference.vllm import VLLMEngine

# Type alias for inference engine types
InferenceEngineAlias = HuggingFaceEngine | VLLMEngine | TGIEngine | TensorRTLLMEngine | SGLangEngine


class InferenceMemoryCalculator:
    """Main inference memory calculator.

    This class provides a high-level interface for calculating
    GPU memory requirements for LLM inference with different engines.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        inference_config: InferenceConfig,
        gpu_config: GPUConfig | None = None,
    ) -> None:
        """Initialize the inference calculator.

        Args:
            model_config: Model architecture configuration
            inference_config: Inference hyperparameters
            gpu_config: Hardware configuration (default: 1x 80GB GPU)
        """
        self.model_config = model_config
        self.inference_config = inference_config
        self.gpu_config = gpu_config or GPUConfig()

    def calculate(self, engine_type: InferenceEngineType) -> InferenceMemoryResult:
        """Calculate inference GPU memory requirements.

        Selects the appropriate inference engine based on the specified type
        and returns the memory calculation result.

        Args:
            engine_type: The inference engine to use

        Returns:
            InferenceMemoryResult with complete memory breakdown
        """
        engine = self._get_engine(engine_type)
        return engine.calculate_memory()

    def _get_engine(self, engine_type: InferenceEngineType) -> InferenceEngineAlias:
        """Get the appropriate inference engine instance.

        Args:
            engine_type: The type of inference engine

        Returns:
            Engine instance configured with current settings
        """
        match engine_type:
            case InferenceEngineType.HUGGINGFACE:
                return HuggingFaceEngine(
                    model_config=self.model_config,
                    inference_config=self.inference_config,
                    gpu_config=self.gpu_config,
                )
            case InferenceEngineType.VLLM:
                return VLLMEngine(
                    model_config=self.model_config,
                    inference_config=self.inference_config,
                    gpu_config=self.gpu_config,
                )
            case InferenceEngineType.TGI:
                return TGIEngine(
                    model_config=self.model_config,
                    inference_config=self.inference_config,
                    gpu_config=self.gpu_config,
                )
            case InferenceEngineType.TENSORRT_LLM | InferenceEngineType.TRTLLM:
                return TensorRTLLMEngine(
                    model_config=self.model_config,
                    inference_config=self.inference_config,
                    gpu_config=self.gpu_config,
                )
            case InferenceEngineType.SGLANG:
                return SGLangEngine(
                    model_config=self.model_config,
                    inference_config=self.inference_config,
                    gpu_config=self.gpu_config,
                )
            case _:
                raise ValueError(f"Unknown inference engine type: {engine_type}")
