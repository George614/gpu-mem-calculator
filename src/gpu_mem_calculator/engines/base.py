"""Base class for training engine implementations."""

from abc import ABC, abstractmethod

from gpu_mem_calculator.core.models import (
    EngineConfig,
    GPUConfig,
    MemoryBreakdown,
    MemoryResult,
    ModelConfig,
    ParallelismConfig,
    TrainingConfig,
)


class BaseEngine(ABC):
    """Abstract base class for training engine memory calculation.

    Each training engine (PyTorch DDP, DeepSpeed, Megatron-LM, etc.)
    should implement this interface to provide engine-specific
    memory calculations.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        parallelism_config: ParallelismConfig,
        engine_config: EngineConfig,
        gpu_config: GPUConfig,
    ) -> None:
        """Initialize the engine with configuration.

        Args:
            model_config: Model architecture configuration
            training_config: Training hyperparameters
            parallelism_config: Parallelism settings
            engine_config: Engine-specific configuration
            gpu_config: Hardware configuration
        """
        self.model_config = model_config
        self.training_config = training_config
        self.parallelism_config = parallelism_config
        self.engine_config = engine_config
        self.gpu_config = gpu_config

    @abstractmethod
    def calculate_memory(self) -> MemoryResult:
        """Calculate memory requirements for this engine.

        This is the main method that should be implemented by each engine.

        Returns:
            MemoryResult with complete memory breakdown
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
            Tuple of (fits_on_gpu, utilization_percent, recommended_batch_size)
        """
        available_memory = self.gpu_config.gpu_memory_gb
        utilization_percent = (total_memory_per_gpu / available_memory) * 100

        fits_on_gpu = total_memory_per_gpu <= available_memory

        # If doesn't fit, suggest a smaller batch size
        recommended_batch_size = None
        if not fits_on_gpu:
            # Simple heuristic: scale batch size inversely with memory excess
            excess_factor = total_memory_per_gpu / available_memory
            recommended_batch_size = max(1, int(self.training_config.batch_size / excess_factor))

        return fits_on_gpu, utilization_percent, recommended_batch_size

    def _create_result(
        self,
        breakdown: MemoryBreakdown,
        cpu_memory_gb: float = 0.0,
    ) -> MemoryResult:
        """Create a MemoryResult from breakdown.

        Args:
            breakdown: Memory breakdown by component
            cpu_memory_gb: CPU memory required (default 0)

        Returns:
            Complete MemoryResult
        """
        total_memory_per_gpu = breakdown.total_memory_gb
        total_memory_all_gpus = total_memory_per_gpu * self.gpu_config.num_gpus

        fits_on_gpu, utilization_percent, recommended_batch_size = self._check_feasibility(
            total_memory_per_gpu
        )

        return MemoryResult(
            total_memory_per_gpu_gb=total_memory_per_gpu,
            total_memory_all_gpus_gb=total_memory_all_gpus,
            cpu_memory_gb=cpu_memory_gb,
            breakdown=breakdown,
            fits_on_gpu=fits_on_gpu,
            memory_utilization_percent=utilization_percent,
            recommended_batch_size=recommended_batch_size,
        )

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size with gradient accumulation."""
        return (
            self.training_config.batch_size
            * self.training_config.gradient_accumulation_steps
            * self.parallelism_config.data_parallel_size
        )

    @property
    def total_num_gpus(self) -> int:
        """Get total number of GPUs."""
        return self.gpu_config.num_gpus

    @property
    def num_gpus_per_model(self) -> int:
        """Get number of GPUs per model replica.

        This is tensor_parallel * pipeline_parallel for distributed training.
        """
        return (
            self.parallelism_config.tensor_parallel_size
            * self.parallelism_config.pipeline_parallel_size
        )
