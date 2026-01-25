"""Base class for training engine implementations."""

from abc import ABC, abstractmethod

from gpu_mem_calculator.core.models import (
    EngineConfig,
    GPUConfig,
    MemoryBreakdown,
    MemoryResult,
    ModelConfig,
    NodeConfig,
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
        node_config: NodeConfig | None = None,
    ) -> None:
        """Initialize the engine with configuration.

        Args:
            model_config: Model architecture configuration
            training_config: Training hyperparameters
            parallelism_config: Parallelism settings
            engine_config: Engine-specific configuration
            gpu_config: Hardware configuration
            node_config: Multi-node configuration (optional)
        """
        self.model_config = model_config
        self.training_config = training_config
        self.parallelism_config = parallelism_config
        self.engine_config = engine_config
        self.gpu_config = gpu_config
        self.node_config = node_config or NodeConfig()

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

        # Calculate network overhead for multi-node configurations
        network_overhead = None
        multi_node_info = None
        if self.node_config.is_multi_node:
            from gpu_mem_calculator.core.multinode import MultiNodeCalculator

            multinode_calc = MultiNodeCalculator(
                model_config=self.model_config,
                training_config=self.training_config,
                parallelism_config=self.parallelism_config,
                node_config=self.node_config,
                engine_config=self.engine_config,
            )
            network_overhead = multinode_calc.calculate_network_overhead()

            # Add multi-node info
            multi_node_info = {
                "num_nodes": self.node_config.num_nodes,
                "gpus_per_node": self.node_config.gpus_per_node,
                "interconnect_type": self.node_config.interconnect_type.value,
                "interconnect_bandwidth_gbps": self.node_config.get_interconnect_bandwidth_gbps(),
            }

        return MemoryResult(
            total_memory_per_gpu_gb=total_memory_per_gpu,
            total_memory_all_gpus_gb=total_memory_all_gpus,
            cpu_memory_gb=cpu_memory_gb,
            breakdown=breakdown,
            network_overhead=network_overhead,
            fits_on_gpu=fits_on_gpu,
            memory_utilization_percent=utilization_percent,
            recommended_batch_size=recommended_batch_size,
            multi_node_info=multi_node_info,
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

    def calculate_moe_activation_multiplier(self) -> float:
        """Calculate activation memory multiplier for MoE models.

        For MoE models, activation memory depends on top_k (active experts per token)
        rather than total number of experts. This is because only top_k experts
        are activated per token during forward/backward pass.

        Returns:
            Multiplier for activation memory (1.0 for dense models, <1 for MoE)
        """
        if not self.model_config.moe_enabled:
            return 1.0

        # For MoE: only top_k experts are active per token
        # Activation memory scales with active_experts / total_experts
        # But we also have router overhead and gating network activations

        num_experts = self.model_config.num_experts
        top_k = self.model_config.top_k

        # Base activation ratio: only top_k experts active
        activation_ratio = top_k / num_experts

        # Add router overhead (typically 5-15% extra for gating)
        router_overhead = 0.1

        # For models with shared experts (like GLM), adjust accordingly
        if self.model_config.shared_expert_intermediate_size:
            # Shared expert is always active, so add its contribution
            # Calculate the size ratio between shared and routed experts
            shared_size = self.model_config.shared_expert_intermediate_size
            # Get routed expert size, default to 4x hidden_size if not specified
            routed_size = self.model_config.expert_intermediate_size or (
                self.model_config.hidden_size * 4
            )
            # Shared expert contribution = shared_size / (routed_size * num_experts)
            # This represents the fraction of total expert capacity that shared expert uses
            if routed_size > 0 and num_experts > 0:
                size_ratio = shared_size / (routed_size * num_experts)
                activation_ratio = activation_ratio + size_ratio

        return min(1.0, activation_ratio + router_overhead)

    def calculate_moe_parameter_ratio(self) -> float:
        """Calculate effective parameter ratio for MoE models.

        For MoE models, only top_k experts are used during forward pass,
        but all expert parameters are stored in memory.

        Returns:
            Ratio of active parameters to total parameters (for memory estimation)
        """
        if not self.model_config.moe_enabled:
            return 1.0

        # All expert parameters are stored, but only top_k are used per token
        # For gradient calculation, we need gradients for all experts
        # So parameter storage = 1.0 (all params stored)
        # But we can use this for inference-specific calculations

        return 1.0  # All parameters stored in memory
