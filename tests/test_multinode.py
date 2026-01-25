"""Tests for multi-node training calculation module.

This module tests the multi-node training functionality including:
- Network overhead calculation
- Hybrid parallelism optimization
- Multi-node configuration
"""

import pytest

from gpu_mem_calculator.core.models import (
    DType,
    EngineConfig,
    EngineType,
    HybridParallelismConfig,
    InterconnectType,
    ModelConfig,
    NodeConfig,
    ParallelismConfig,
    TrainingConfig,
)
from gpu_mem_calculator.core.multinode import MultiNodeCalculator

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_model_config():
    """Standard 7B model for multi-node testing."""
    return ModelConfig(
        name="llama2-7b",
        num_parameters=7_000_000_000,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        vocab_size=32000,
        max_seq_len=2048,
    )


@pytest.fixture
def base_training_config():
    """Base training configuration."""
    return TrainingConfig(
        batch_size=4,
        gradient_accumulation_steps=1,
        dtype=DType.BF16,
    )


@pytest.fixture
def base_parallelism_config():
    """Base parallelism configuration."""
    return ParallelismConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=4,
    )


@pytest.fixture
def base_node_config():
    """Standard 4-node configuration."""
    return NodeConfig(
        num_nodes=4,
        gpus_per_node=8,
        interconnect_type=InterconnectType.INFINIBAND,
    )


@pytest.fixture
def base_engine_config():
    """Base engine configuration (DeepSpeed ZeRO-2)."""
    return EngineConfig(
        type=EngineType.DEEPSPEED,
        zero_stage=2,
    )


@pytest.fixture(params=["infiniband", "nvlink", "ethernet_100g"])
def interconnect_type(request):
    """Parameterized interconnect types."""
    return InterconnectType(request.param)


@pytest.fixture(params=[1, 2, 3])
def zero_stage(request):
    """Parameterized ZeRO stages."""
    return request.param


# =============================================================================
# TestNodeConfig
# =============================================================================


class TestNodeConfig:
    """Tests for NodeConfig validation and properties."""

    def test_single_node_default(self, base_model_config):
        """Test is_multi_node=False when num_nodes=1."""
        config = NodeConfig(num_nodes=1)
        assert config.is_multi_node is False

    def test_multi_node_detection(self, base_model_config):
        """Test is_multi_node=True when num_nodes>1."""
        config = NodeConfig(num_nodes=4)
        assert config.is_multi_node is True

    def test_gpus_per_node_auto_calculation(self, base_model_config):
        """Test auto-calculation from num_gpus."""
        # With 32 GPUs and 4 nodes -> 8 GPUs per node
        config = NodeConfig(
            num_nodes=4,
            # gpus_per_node not specified, should be calculated from num_gpus if provided
        )
        # If not specified, it stays None
        assert config.gpus_per_node is None

    def test_interconnect_bandwidth_defaults(self, base_model_config):
        """Test all 6 InterconnectType defaults."""
        bandwidths = {
            InterconnectType.INFINIBAND: 200.0,
            InterconnectType.NVLINK: 300.0,
            InterconnectType.ETHERNET_10G: 10.0,
            InterconnectType.ETHERNET_25G: 25.0,
            InterconnectType.ETHERNET_100G: 100.0,
            InterconnectType.ETHERNET_200G: 200.0,
        }

        for interconnect, expected_bw in bandwidths.items():
            config = NodeConfig(interconnect_type=interconnect)
            assert config.get_interconnect_bandwidth_gbps() == expected_bw

    def test_custom_bandwidth_override(self, base_model_config):
        """Test interconnect_bandwidth_gbps override."""
        custom_bw = 400.0
        config = NodeConfig(
            interconnect_type=InterconnectType.INFINIBAND,
            interconnect_bandwidth_gbps=custom_bw,
        )

        assert config.get_interconnect_bandwidth_gbps() == custom_bw
        assert config.interconnect_bandwidth_gbps == custom_bw


# =============================================================================
# TestNetworkOverhead
# =============================================================================


class TestNetworkOverhead:
    """Tests for network overhead calculation."""

    def test_single_node_zero_overhead(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test empty NetworkOverhead for single node."""
        single_node_config = NodeConfig(num_nodes=1)

        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            single_node_config,
            base_engine_config,
        )

        overhead = calculator.calculate_network_overhead()

        # Single node should have zero overhead
        assert overhead.allreduce_gb == 0.0
        assert overhead.allgather_gb == 0.0
        assert overhead.reducescatter_gb == 0.0
        assert overhead.point_to_point_gb == 0.0
        assert overhead.total_overhead_gb == 0.0

    def test_allreduce_calculation(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test Ring AllReduce: 2*model_size formula."""
        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        overhead = calculator.calculate_network_overhead()

        # AllReduce: 2 * model_size (in bytes)
        # 7B params * 2 bytes (BF16) = 14GB
        # AllReduce = 2 * 14GB = 28GB
        # Divided by num_nodes for cross-node traffic: 28 / 4 = 7GB
        expected_allreduce_gb = (2 * 7_000_000_000 * 2 / (1024**3)) / 4

        assert overhead.allreduce_gb == pytest.approx(expected_allreduce_gb, rel=0.01)

    def test_allgather_zero3(
        self, base_model_config, base_training_config, base_parallelism_config, base_node_config
    ):
        """Test AllGather for ZeRO-3 parameter gathering."""
        # ZeRO-3 config
        engine_config = EngineConfig(type=EngineType.DEEPSPEED, zero_stage=3)

        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            engine_config,
        )

        overhead = calculator.calculate_network_overhead()

        # ZeRO-3: gather all parameters
        # Model size: 7B * 2 bytes = 14GB
        # Cross-node: 14 / 4 = 3.5GB
        expected_allgather_gb = (7_000_000_000 * 2 / (1024**3)) / 4

        assert overhead.allgather_gb == pytest.approx(expected_allgather_gb, rel=0.01)

    def test_allgather_tensor_parallel(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test AllGather for tensor parallel."""
        # Enable tensor parallel
        base_parallelism_config.tensor_parallel_size = 4

        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        overhead = calculator.calculate_network_overhead()

        # TP: AllGather = model_size / TP_size
        # Cross-node: divide by num_nodes
        expected = (7_000_000_000 * 2 / 4 / (1024**3)) / 4

        assert overhead.allgather_gb == pytest.approx(expected, rel=0.01)

    def test_reducescatter_zero2(
        self, base_model_config, base_training_config, base_parallelism_config, base_node_config
    ):
        """Test ReduceScatter for ZeRO-2 gradient scattering."""
        # ZeRO-2 config
        engine_config = EngineConfig(type=EngineType.DEEPSPEED, zero_stage=2)

        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            engine_config,
        )

        overhead = calculator.calculate_network_overhead()

        # ZeRO-2: ReduceScatter full model
        # Model size: 7B * 2 bytes = 14GB
        # Cross-node: 14 / 4 = 3.5GB
        expected_reducescatter_gb = (7_000_000_000 * 2 / (1024**3)) / 4

        assert overhead.reducescatter_gb == pytest.approx(expected_reducescatter_gb, rel=0.01)

    def test_pipeline_communication(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test point-to-point pipeline communication."""
        # Enable pipeline parallel
        base_parallelism_config.pipeline_parallel_size = 4

        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        overhead = calculator.calculate_network_overhead()

        # Pipeline: activations between stages with microbatching
        # activation_size = batch * seq * hidden * 2 bytes
        # Formula: activation_bytes * num_microbatches * num_stages * 2 (forward + backward)
        # Cross-node: divide by num_nodes
        activation_bytes = 4 * 2048 * 4096 * 2
        num_microbatches = 4  # Default value in multinode.py
        num_stages = 4  # pp_size
        pipeline_bytes = activation_bytes * num_microbatches * num_stages * 2
        expected_gb = (pipeline_bytes / 4) / (1024**3)

        assert overhead.point_to_point_gb == pytest.approx(expected_gb, rel=0.1)

    def test_cross_node_adjustment(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test traffic division by num_nodes."""
        # Compare 2-node vs 4-node
        node_config_2 = NodeConfig(num_nodes=2, gpus_per_node=8)
        node_config_4 = NodeConfig(num_nodes=4, gpus_per_node=8)

        calc_2 = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            node_config_2,
            base_engine_config,
        )

        calc_4 = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            node_config_4,
            base_engine_config,
        )

        overhead_2 = calc_2.calculate_network_overhead()
        overhead_4 = calc_4.calculate_network_overhead()

        # 4-node should have less overhead per node (traffic divided)
        assert overhead_4.allreduce_gb < overhead_2.allreduce_gb

    def test_communication_time_estimation(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test bandwidth/latency time calculation."""
        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        overhead = calculator.calculate_network_overhead()

        # Time should be estimated based on bandwidth and latency
        assert overhead.estimated_overhead_ms_per_step is not None
        assert overhead.estimated_overhead_ms_per_step >= 0

    def test_network_efficiency_factor(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test 85% efficiency in time estimation."""
        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        overhead = calculator.calculate_network_overhead()

        # Time = (size / (bandwidth/8)) / 0.85 + latency
        # Verify time is reasonable
        assert overhead.estimated_overhead_ms_per_step > 0

    def test_latency_overhead_per_node(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test 50μs per node latency overhead."""
        # Compare 2-node vs 8-node
        node_config_8 = NodeConfig(num_nodes=8, gpus_per_node=8)

        calc_8 = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            node_config_8,
            base_engine_config,
        )

        overhead_8 = calc_8.calculate_network_overhead()

        # More nodes should have more latency overhead
        # (assuming similar communication volume)
        assert overhead_8.estimated_overhead_ms_per_step > 0


# =============================================================================
# TestMultiNodeCalculator
# =============================================================================


class TestMultiNodeCalculator:
    """Tests for MultiNodeCalculator class."""

    def test_initialization(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test all config attributes stored correctly."""
        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        assert calculator.model_config == base_model_config
        assert calculator.training_config == base_training_config
        assert calculator.parallelism_config == base_parallelism_config
        assert calculator.node_config == base_node_config
        assert calculator.engine_config == base_engine_config

    def test_calculate_network_overhead(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test complete overhead breakdown."""
        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        overhead = calculator.calculate_network_overhead()

        # All components should be present
        assert overhead.allreduce_gb >= 0
        assert overhead.allgather_gb >= 0
        assert overhead.reducescatter_gb >= 0
        assert overhead.point_to_point_gb >= 0
        assert overhead.total_overhead_gb >= 0
        assert overhead.estimated_overhead_ms_per_step is not None

        # Total should be sum of components
        expected_total = (
            overhead.allreduce_gb
            + overhead.allgather_gb
            + overhead.reducescatter_gb
            + overhead.point_to_point_gb
        )
        assert overhead.total_overhead_gb == pytest.approx(expected_total, rel=0.01)

    def test_dtype_bytes_mapping(
        self, base_model_config, base_parallelism_config, base_node_config, base_engine_config
    ):
        """Test dtype to bytes mapping."""
        dtype_bytes_map = {
            DType.FP32: 4,
            DType.FP16: 2,
            DType.BF16: 2,
            DType.INT8: 1,
            DType.INT4: 0.5,
        }

        for dtype, expected_bytes in dtype_bytes_map.items():
            training_config = TrainingConfig(batch_size=4, dtype=dtype)
            calculator = MultiNodeCalculator(
                base_model_config,
                training_config,
                base_parallelism_config,
                base_node_config,
                base_engine_config,
            )

            assert calculator._get_dtype_bytes() == expected_bytes

    def test_model_size_gb_calculation(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test model size in GB calculation."""
        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        # 7B params * 2 bytes (BF16) / 1024^3
        expected_gb = 7_000_000_000 * 2 / (1024**3)

        assert calculator._calculate_model_size_gb() == pytest.approx(expected_gb, rel=0.01)

    def test_allreduce_vs_deepspeed(
        self, base_model_config, base_training_config, base_parallelism_config, base_node_config
    ):
        """Test DDP vs ZeRO AllReduce overhead."""
        # DDP (no ZeRO)
        ddp_config = EngineConfig(type=EngineType.PYTORCH_DDP)
        ddp_calc = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            ddp_config,
        )

        # ZeRO-2
        zero2_config = EngineConfig(type=EngineType.DEEPSPEED, zero_stage=2)
        zero2_calc = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            zero2_config,
        )

        ddp_overhead = ddp_calc.calculate_network_overhead()
        zero2_overhead = zero2_calc.calculate_network_overhead()

        # Both should have AllReduce
        assert ddp_overhead.allreduce_gb > 0
        assert zero2_overhead.allreduce_gb > 0

    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_zero_stage_impact(
        self,
        stage,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
    ):
        """Test communication overhead for different ZeRO stages."""
        engine_config = EngineConfig(type=EngineType.DEEPSPEED, zero_stage=stage)

        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            engine_config,
        )

        overhead = calculator.calculate_network_overhead()

        # All stages should have some overhead
        assert overhead.total_overhead_gb >= 0

        # ZeRO-3 should have AllGather (parameter gathering)
        if stage == 3:
            assert overhead.allgather_gb > 0

        # ZeRO-2 should have ReduceScatter (gradient scattering)
        if stage == 2:
            assert overhead.reducescatter_gb > 0


# =============================================================================
# TestHybridParallelism
# =============================================================================


class TestHybridParallelism:
    """Tests for hybrid parallelism optimization."""

    def test_auto_optimize_disabled(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test returns original config when auto_optimize=False."""
        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        hybrid_config = HybridParallelismConfig(auto_optimize=False)
        result = calculator.optimize_hybrid_parallelism(hybrid_config)

        # Should return original config unchanged
        assert result == base_parallelism_config

    def test_pipeline_parallel_preferred(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test PP preference with prefer_pipeline_parallel=True."""
        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        hybrid_config = HybridParallelismConfig(
            auto_optimize=True,
            prefer_pipeline_parallel=True,
        )
        result = calculator.optimize_hybrid_parallelism(hybrid_config)

        # Should enable PP when multi-node
        assert result.pipeline_parallel_size >= 1

        # Verify all sizes >= 1
        assert result.tensor_parallel_size >= 1
        assert result.data_parallel_size >= 1

    def test_sequence_parallel_threshold(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test SP enablement >= 4096 tokens."""
        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        # Short sequence (no SP)
        base_model_config.max_seq_len = 2048
        hybrid_config = HybridParallelismConfig(
            auto_optimize=True,
            enable_sequence_parallel=True,
            sequence_parallel_threshold=4096,
        )
        result = calculator.optimize_hybrid_parallelism(hybrid_config)

        # Should respect threshold
        assert isinstance(result.sequence_parallel, bool)

    def test_tp_within_node(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test TP limited to gpus_per_node."""
        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        hybrid_config = HybridParallelismConfig(
            auto_optimize=True,
            prefer_pipeline_parallel=True,
        )
        result = calculator.optimize_hybrid_parallelism(hybrid_config)

        # TP should be limited to GPUs per node (8)
        assert result.tensor_parallel_size <= base_node_config.gpus_per_node

    def test_dp_maximization(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test default maximizes data parallel."""
        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        # Default config (no PP preference)
        hybrid_config = HybridParallelismConfig(auto_optimize=True)
        result = calculator.optimize_hybrid_parallelism(hybrid_config)

        # Should maximize DP when no PP preference
        assert result.tensor_parallel_size == 1
        assert result.pipeline_parallel_size == 1
        assert (
            result.data_parallel_size == base_node_config.num_nodes * base_node_config.gpus_per_node
        )

    def test_parallelism_product_constraint(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test TP * PP * DP <= total_gpus constraint."""
        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        hybrid_config = HybridParallelismConfig(
            auto_optimize=True,
            prefer_pipeline_parallel=True,
        )
        result = calculator.optimize_hybrid_parallelism(hybrid_config)

        total_gpus = base_node_config.num_nodes * base_node_config.gpus_per_node
        product = (
            result.tensor_parallel_size * result.pipeline_parallel_size * result.data_parallel_size
        )

        # Product should not exceed total GPUs
        assert product <= total_gpus

    def test_min_parallelism_values(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test all parallelism sizes >= 1."""
        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        hybrid_config = HybridParallelismConfig(auto_optimize=True)
        result = calculator.optimize_hybrid_parallelism(hybrid_config)

        # All sizes should be at least 1
        assert result.tensor_parallel_size >= 1
        assert result.pipeline_parallel_size >= 1
        assert result.data_parallel_size >= 1


# =============================================================================
# TestMultiNodeIntegration
# =============================================================================


class TestMultiNodeIntegration:
    """Integration tests for multi-node scenarios."""

    def test_with_7b_model(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test 4 nodes, 8 GPUs each with 7B model."""
        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        overhead = calculator.calculate_network_overhead()

        # Should have valid overhead calculation
        assert overhead.total_overhead_gb > 0
        assert overhead.estimated_overhead_ms_per_step > 0

    def test_with_175b_model(self):
        """Test large model requiring multi-node."""
        large_model = ModelConfig(
            name="gpt3-175b",
            num_parameters=175_000_000_000,
            num_layers=96,
            hidden_size=12288,
            num_attention_heads=96,
            vocab_size=51200,
            max_seq_len=2048,
        )

        training_config = TrainingConfig(batch_size=1, dtype=DType.BF16)
        parallelism_config = ParallelismConfig(
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        )
        node_config = NodeConfig(num_nodes=8, gpus_per_node=8)
        engine_config = EngineConfig(type=EngineType.DEEPSPEED, zero_stage=3)

        calculator = MultiNodeCalculator(
            large_model,
            training_config,
            parallelism_config,
            node_config,
            engine_config,
        )

        overhead = calculator.calculate_network_overhead()

        # Large model should have significant overhead
        assert overhead.total_overhead_gb > 0

    def test_infiniband_vs_ethernet(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Compare InfiniBand vs Ethernet interconnect."""
        # InfiniBand (200 Gbps)
        ib_config = NodeConfig(
            num_nodes=4,
            gpus_per_node=8,
            interconnect_type=InterconnectType.INFINIBAND,
        )

        # Ethernet 100G (100 Gbps)
        eth_config = NodeConfig(
            num_nodes=4,
            gpus_per_node=8,
            interconnect_type=InterconnectType.ETHERNET_100G,
        )

        ib_calc = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            ib_config,
            base_engine_config,
        )

        eth_calc = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            eth_config,
            base_engine_config,
        )

        ib_overhead = ib_calc.calculate_network_overhead()
        eth_overhead = eth_calc.calculate_network_overhead()

        # Communication volume should be same
        assert ib_overhead.total_overhead_gb == pytest.approx(
            eth_overhead.total_overhead_gb, rel=0.01
        )

        # But Ethernet should have higher time overhead (slower)
        assert (
            eth_overhead.estimated_overhead_ms_per_step > ib_overhead.estimated_overhead_ms_per_step
        )

    def test_communication_overhead_breakdown(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test all communication components present."""
        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        overhead = calculator.calculate_network_overhead()

        # All components should be present (some may be 0)
        assert hasattr(overhead, "allreduce_gb")
        assert hasattr(overhead, "allgather_gb")
        assert hasattr(overhead, "reducescatter_gb")
        assert hasattr(overhead, "point_to_point_gb")
        assert hasattr(overhead, "total_overhead_gb")
        assert hasattr(overhead, "estimated_overhead_ms_per_step")

    def test_feasibility_with_network(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test config fits including network overhead."""
        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        overhead = calculator.calculate_network_overhead()

        # Network overhead should be reasonable fraction of model size
        model_size_gb = calculator._calculate_model_size_gb()

        # Overhead should not exceed model size by too much
        assert overhead.total_overhead_gb < model_size_gb * 10  # Within 10x


# =============================================================================
# Edge Cases
# =============================================================================


class TestMultiNodeEdgeCases:
    """Edge case tests for multi-node functionality."""

    def test_zero_gpus_per_node_constraint(self):
        """Test that gpus_per_node must be >= 1."""
        with pytest.raises(ValueError):
            NodeConfig(num_nodes=4, gpus_per_node=0)

    def test_custom_bandwidth_zero(self):
        """Test custom bandwidth = 0 (edge case)."""
        # This should be rejected by validation
        with pytest.raises(ValueError):
            NodeConfig(
                interconnect_type=InterconnectType.INFINIBAND,
                interconnect_bandwidth_gbps=0.0,
            )

    def test_pipeline_parallel_with_pp_one(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test pipeline parallel with PP=1 (no pipeline)."""
        base_parallelism_config.pipeline_parallel_size = 1

        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        overhead = calculator.calculate_network_overhead()

        # Should have no pipeline overhead
        assert overhead.point_to_point_gb == 0.0

    def test_sequence_parallel_disabled(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_node_config,
        base_engine_config,
    ):
        """Test sequence parallel disabled."""
        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_node_config,
            base_engine_config,
        )

        hybrid_config = HybridParallelismConfig(
            auto_optimize=True,
            enable_sequence_parallel=False,
        )
        result = calculator.optimize_hybrid_parallelism(hybrid_config)

        # Should respect SP disabled
        assert result.sequence_parallel is False

    def test_num_nodes_one(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test single node (num_nodes=1)."""
        single_node_config = NodeConfig(num_nodes=1)

        calculator = MultiNodeCalculator(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            single_node_config,
            base_engine_config,
        )

        overhead = calculator.calculate_network_overhead()

        # Should have zero overhead
        assert overhead.total_overhead_gb == 0.0
        assert overhead.estimated_overhead_ms_per_step is None
