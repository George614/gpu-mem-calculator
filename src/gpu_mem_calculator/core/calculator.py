"""Main GPU memory calculator.

Orchestrates the memory calculation by selecting the appropriate
training engine and aggregating results.
"""

from gpu_mem_calculator.config.parser import ConfigParser
from gpu_mem_calculator.core.models import (
    EngineConfig,
    EngineType,
    GPUConfig,
    MemoryResult,
    ModelConfig,
    ParallelismConfig,
    TrainingConfig,
)
from gpu_mem_calculator.engines import (
    DeepSpeedEngine,
    FSDPEngine,
    MegatronDeepSpeedEngine,
    MegatronLMEngine,
    PyTorchDDPEngine,
)

# Type alias for engine types
EngineTypeAlias = (
    PyTorchDDPEngine
    | DeepSpeedEngine
    | MegatronLMEngine
    | FSDPEngine
    | MegatronDeepSpeedEngine
)


class GPUMemoryCalculator:
    """Main GPU memory calculator.

    This class provides a high-level interface for calculating
    GPU memory requirements for LLM training.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        parallelism_config: ParallelismConfig | None = None,
        engine_config: EngineConfig | None = None,
        gpu_config: GPUConfig | None = None,
    ) -> None:
        """Initialize the calculator.

        Args:
            model_config: Model architecture configuration
            training_config: Training hyperparameters
            parallelism_config: Parallelism settings (default: no parallelism)
            engine_config: Training engine configuration (default: PyTorch DDP)
            gpu_config: Hardware configuration (default: 1x 80GB GPU)
        """
        self.model_config = model_config
        self.training_config = training_config
        self.parallelism_config = parallelism_config or ParallelismConfig()
        self.engine_config = engine_config or EngineConfig()
        self.gpu_config = gpu_config or GPUConfig()

    def calculate(self) -> MemoryResult:
        """Calculate GPU memory requirements.

        Selects the appropriate training engine based on configuration
        and returns the memory calculation result.

        Returns:
            MemoryResult with complete memory breakdown
        """
        engine = self._get_engine()
        return engine.calculate_memory()

    def _get_engine(self) -> EngineTypeAlias:
        """Get the appropriate training engine instance.

        Returns:
            Engine instance configured with current settings
        """
        match self.engine_config.type:
            case EngineType.PYTORCH_DDP:
                return PyTorchDDPEngine(
                    model_config=self.model_config,
                    training_config=self.training_config,
                    parallelism_config=self.parallelism_config,
                    engine_config=self.engine_config,
                    gpu_config=self.gpu_config,
                )
            case EngineType.DEEPSPEED:
                return DeepSpeedEngine(
                    model_config=self.model_config,
                    training_config=self.training_config,
                    parallelism_config=self.parallelism_config,
                    engine_config=self.engine_config,
                    gpu_config=self.gpu_config,
                )
            case EngineType.MEGATRON_LM:
                return MegatronLMEngine(
                    model_config=self.model_config,
                    training_config=self.training_config,
                    parallelism_config=self.parallelism_config,
                    engine_config=self.engine_config,
                    gpu_config=self.gpu_config,
                )
            case EngineType.FSDP:
                return FSDPEngine(
                    model_config=self.model_config,
                    training_config=self.training_config,
                    parallelism_config=self.parallelism_config,
                    engine_config=self.engine_config,
                    gpu_config=self.gpu_config,
                )
            case EngineType.MEGATRON_DEEPSPEED:
                return MegatronDeepSpeedEngine(
                    model_config=self.model_config,
                    training_config=self.training_config,
                    parallelism_config=self.parallelism_config,
                    engine_config=self.engine_config,
                    gpu_config=self.gpu_config,
                )
            case _:
                # Default to PyTorch DDP
                return PyTorchDDPEngine(
                    model_config=self.model_config,
                    training_config=self.training_config,
                    parallelism_config=self.parallelism_config,
                    engine_config=self.engine_config,
                    gpu_config=self.gpu_config,
                )

    @classmethod
    def from_config_file(
        cls,
        config_path: str,
    ) -> "GPUMemoryCalculator":
        """Create calculator from configuration file.

        Args:
            config_path: Path to JSON configuration file

        Returns:
            Configured GPUMemoryCalculator instance
        """
        model_config, training_config, parallelism_config, engine_config, gpu_config = (
            ConfigParser.parse_full_config(config_path)
        )

        return cls(
            model_config=model_config,
            training_config=training_config,
            parallelism_config=parallelism_config,
            engine_config=engine_config,
            gpu_config=gpu_config,
        )

    def to_dict(self) -> dict:
        """Export calculator configuration to dictionary.

        Returns:
            Dictionary with all configuration
        """
        return {
            "model": self.model_config.model_dump(),
            "training": self.training_config.model_dump(),
            "parallelism": self.parallelism_config.model_dump(),
            "engine": self.engine_config.model_dump(),
            "hardware": self.gpu_config.model_dump(),
        }
