"""Multi-node training calculator.

Handles network communication overhead calculation and hybrid
parallelism optimization for multi-node training configurations.
"""

from gpu_mem_calculator.core.models import (
    EngineConfig,
    EngineType,
    HybridParallelismConfig,
    ModelConfig,
    NetworkOverhead,
    NodeConfig,
    ParallelismConfig,
    TrainingConfig,
)


class MultiNodeCalculator:
    """Calculator for multi-node training overhead and optimization.

    This class provides:
    - Network communication overhead estimation
    - Hybrid parallelism strategy optimization
    - Multi-node performance modeling
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        parallelism_config: ParallelismConfig,
        node_config: NodeConfig,
        engine_config: EngineConfig,
    ) -> None:
        """Initialize the multi-node calculator.

        Args:
            model_config: Model architecture configuration
            training_config: Training hyperparameters
            parallelism_config: Parallelism settings
            node_config: Multi-node hardware configuration
            engine_config: Training engine configuration
        """
        self.model_config = model_config
        self.training_config = training_config
        self.parallelism_config = parallelism_config
        self.node_config = node_config
        self.engine_config = engine_config

    def calculate_network_overhead(self) -> NetworkOverhead:
        """Calculate network communication overhead for multi-node training.

        Estimates communication overhead for different collective operations
        based on model size, parallelism strategy, and interconnect bandwidth.

        Returns:
            NetworkOverhead with detailed breakdown
        """
        if not self.node_config.is_multi_node:
            return NetworkOverhead()

        # Get model size in bytes
        model_params = self.model_config.num_parameters
        dtype_bytes = self._get_dtype_bytes()
        model_size_bytes = int(model_params * dtype_bytes)

        # Calculate communication for each collective operation
        allreduce_gb = self._calculate_allreduce_overhead(model_size_bytes)
        allgather_gb = self._calculate_allgather_overhead(model_size_bytes)
        reducescatter_gb = self._calculate_reducescatter_overhead(model_size_bytes)
        point_to_point_gb = self._calculate_pipeline_overhead(model_size_bytes)

        total_overhead_gb = allreduce_gb + allgather_gb + reducescatter_gb + point_to_point_gb

        # Estimate time overhead per step
        overhead_ms = self._estimate_communication_time_ms(total_overhead_gb)

        return NetworkOverhead(
            allreduce_gb=allreduce_gb,
            allgather_gb=allgather_gb,
            reducescatter_gb=reducescatter_gb,
            point_to_point_gb=point_to_point_gb,
            total_overhead_gb=total_overhead_gb,
            estimated_overhead_ms_per_step=overhead_ms,
        )

    def optimize_hybrid_parallelism(
        self,
        hybrid_config: HybridParallelismConfig,
    ) -> ParallelismConfig:
        """Optimize hybrid parallelism strategy for multi-node training.

        Analyzes the hardware configuration and model characteristics
        to recommend optimal parallelism degrees.

        Args:
            hybrid_config: Hybrid parallelism configuration and preferences

        Returns:
            Optimized ParallelismConfig
        """
        if not hybrid_config.auto_optimize:
            return self.parallelism_config

        num_nodes = self.node_config.num_nodes
        gpus_per_node = self.node_config.gpus_per_node or 1
        total_gpus = num_nodes * gpus_per_node

        seq_len = self.model_config.max_seq_len

        # Determine optimal parallelism strategy
        if seq_len >= hybrid_config.sequence_parallel_threshold:
            # Enable sequence parallel for long sequences
            enable_sp = True
        else:
            enable_sp = hybrid_config.enable_sequence_parallel

        # Calculate parallelism degrees
        if hybrid_config.prefer_pipeline_parallel and num_nodes > 1:
            # Prefer pipeline parallel across nodes
            pp_size = int(min(num_nodes, 8))  # Limit pipeline stages
            tp_size = int(min(gpus_per_node, 8))  # Tensor parallel within node
            dp_size = int(total_gpus // (pp_size * tp_size))
        else:
            # Default: maximize data parallel
            tp_size = 1
            pp_size = 1
            dp_size = int(total_gpus)

        # Ensure all values are at least 1
        tp_size = max(1, tp_size)
        pp_size = max(1, pp_size)
        dp_size = max(1, dp_size)

        return ParallelismConfig(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            data_parallel_size=dp_size,
            sequence_parallel=enable_sp,
        )

    def _calculate_allreduce_overhead(self, model_size_bytes: int) -> float:
        """Calculate AllReduce communication overhead.

        AllReduce is used for gradient averaging in data parallel training.
        Algorithm: Ring AllReduce with O(2 * model_size) communication.

        Args:
            model_size_bytes: Model size in bytes

        Returns:
            Communication volume in GB
        """
        # Ring AllReduce: each GPU sends/receives 2 * model_size / num_gpus
        # But we need the total across the network

        # For gradient averaging: 2 * model_size (send + receive)
        allreduce_bytes = 2 * model_size_bytes

        # Adjust for collective operation efficiency
        # In multi-node, cross-node traffic is the bottleneck
        if self.node_config.is_multi_node:
            # Cross-node traffic estimation: divide by num_nodes to account for
            # hierarchical AllReduce (intra-node aggregation via NVLink first,
            # then inter-node communication via InfiniBand/Ethernet).
            # This represents the portion of data that must traverse the
            # slower inter-node interconnect.
            num_nodes = max(1, self.node_config.num_nodes)  # Defensive guard
            allreduce_bytes = int(allreduce_bytes / num_nodes)

        return allreduce_bytes / (1024**3)

    def _calculate_allgather_overhead(self, model_size_bytes: int) -> float:
        """Calculate AllGather communication overhead.

        AllGather is used in ZeRO-3 and tensor parallel for parameter gathering.

        Args:
            model_size_bytes: Model size in bytes

        Returns:
            Communication volume in GB
        """
        # AllGather: (num_gpus - 1) * model_size / num_gpus per GPU
        # But for ZeRO-3, we gather all parameters
        is_zero3 = (
            self.engine_config.type == EngineType.DEEPSPEED and self.engine_config.zero_stage == 3
        )

        if is_zero3:
            # ZeRO-3 gathers all parameters during forward pass
            allgather_bytes = model_size_bytes
        else:
            # Standard allgather for tensor parallel
            allgather_bytes = int(model_size_bytes / self.parallelism_config.tensor_parallel_size)

        # Adjust for multi-node: similar to AllReduce, account for hierarchical
        # communication where intra-node AllGather uses NVLink and only
        # inter-node portion uses the slower interconnect
        if self.node_config.is_multi_node:
            num_nodes = max(1, self.node_config.num_nodes)  # Defensive guard
            allgather_bytes = int(allgather_bytes / num_nodes)

        return allgather_bytes / (1024**3)

    def _calculate_reducescatter_overhead(self, model_size_bytes: int) -> float:
        """Calculate ReduceScatter communication overhead.

        ReduceScatter is used in ZeRO-2 and gradient sharding.

        Args:
            model_size_bytes: Model size in bytes

        Returns:
            Communication volume in GB
        """
        is_zero2 = (
            self.engine_config.type == EngineType.DEEPSPEED and self.engine_config.zero_stage == 2
        )

        if is_zero2:
            # ZeRO-2 scatters gradients
            reducescatter_bytes = model_size_bytes
        else:
            # Standard reducescatter
            reducescatter_bytes = int(model_size_bytes / self.parallelism_config.data_parallel_size)

        # Adjust for multi-node: similar to AllReduce, account for hierarchical
        # communication where intra-node ReduceScatter uses NVLink and only
        # inter-node portion uses the slower interconnect
        if self.node_config.is_multi_node:
            num_nodes = max(1, self.node_config.num_nodes)  # Defensive guard
            reducescatter_bytes = int(reducescatter_bytes / num_nodes)

        return reducescatter_bytes / (1024**3)

    def _calculate_pipeline_overhead(self, model_size_bytes: int) -> float:
        """Calculate pipeline parallel communication overhead.

        Point-to-point communication between pipeline stages.

        Args:
            model_size_bytes: Model size in bytes

        Returns:
            Communication volume in GB
        """
        if self.parallelism_config.pipeline_parallel_size <= 1:
            return 0.0

        # Pipeline parallel sends activations between stages
        # Approximate as layer activations
        hidden_size = self.model_config.hidden_size
        seq_len = self.model_config.max_seq_len
        batch_size = self.training_config.batch_size

        # Activation size per layer
        activation_bytes = batch_size * seq_len * hidden_size * 2  # FP16/BF16

        # Pipeline parallel communication:
        # - Forward pass sends activations between stages
        # - Backward pass sends gradients between stages (same size as activations)
        # - With microbatching, communication happens multiple times per step
        pp_size = max(1, self.parallelism_config.pipeline_parallel_size)
        num_stages = pp_size

        # Default microbatch count estimate (could be configurable)
        # Typical values: 4-16 microbatches for efficient pipeline utilization
        num_microbatches = 4

        # Communication per step: activations × microbatches × stages × 2 (forward + backward)
        pipeline_comm_bytes = activation_bytes * num_microbatches * num_stages * 2

        # Adjust for multi-node: only cross-node pipeline stages contribute to
        # inter-node communication. Estimate based on stage distribution.
        if self.node_config.is_multi_node:
            num_nodes = max(1, self.node_config.num_nodes)  # Defensive guard
            # Assume stages are distributed across nodes, so ~1/num_nodes of
            # pipeline boundaries are cross-node
            pipeline_comm_bytes = int(pipeline_comm_bytes / num_nodes)

        return pipeline_comm_bytes / (1024**3)

    def _estimate_communication_time_ms(self, total_gb: float) -> float:
        """Estimate communication time per training step in milliseconds.

        Args:
            total_gb: Total communication volume in GB

        Returns:
            Estimated time in milliseconds
        """
        if total_gb == 0:
            return 0.0

        # Get bandwidth and convert to GB/s
        # bandwidth_gbps is in gigabits per second (Gbps)
        # Divide by 8 to convert to gigabytes per second (GB/s)
        bandwidth_gbps = self.node_config.get_interconnect_bandwidth_gbps()
        bandwidth_gb_per_sec = bandwidth_gbps / 8.0  # Convert Gbps to GB/s

        # Basic time = size / bandwidth
        time_seconds = total_gb / bandwidth_gb_per_sec

        # Add latency overhead for collective operations
        # Typical latency: 10-50 microseconds per hop
        num_nodes = self.node_config.num_nodes
        latency_overhead = num_nodes * 0.00005  # 50 microseconds per node

        # Network efficiency factor (not 100% efficient)
        efficiency = 0.85

        total_time_seconds = (time_seconds / efficiency) + latency_overhead

        return total_time_seconds * 1000  # Convert to ms

    def _get_dtype_bytes(self) -> float:
        """Get bytes per element based on dtype."""
        dtype_map = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "int8": 1,
            "int4": 0.5,
        }
        return dtype_map.get(self.training_config.dtype.value, 2)

    def _calculate_model_size_gb(self) -> float:
        """Calculate model size in GB."""
        dtype_bytes = self._get_dtype_bytes()
        model_size_bytes = self.model_config.num_parameters * dtype_bytes
        return model_size_bytes / (1024**3)
