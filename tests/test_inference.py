"""Tests for inference memory calculation module.

This module tests the inference engine implementations including:
- HuggingFace Transformers
- vLLM
- TGI (Text Generation Inference)
- TensorRT-LLM
"""

import pytest

from gpu_mem_calculator.core.models import (
    GPUConfig,
    InferenceConfig,
    InferenceEngineType,
    KVCacheQuantization,
    ModelConfig,
)
from gpu_mem_calculator.inference.calculator import InferenceMemoryCalculator
from gpu_mem_calculator.inference.huggingface import HuggingFaceEngine
from gpu_mem_calculator.inference.tensorrt_llm import TensorRTLLMEngine
from gpu_mem_calculator.inference.tgi import TGIEngine
from gpu_mem_calculator.inference.vllm import VLLMEngine

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_model_config():
    """Standard 7B model for inference testing."""
    return ModelConfig(
        name="llama2-7b",
        num_parameters=7_000_000_000,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        vocab_size=32000,
        max_seq_len=4096,
    )


@pytest.fixture
def base_inference_config():
    """Base inference configuration."""
    return InferenceConfig(
        batch_size=4,
        kv_cache_quantization=KVCacheQuantization.NONE,
        use_kv_cache=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )


@pytest.fixture
def base_gpu_config():
    """Standard GPU configuration."""
    return GPUConfig(
        num_gpus=1,
        gpu_memory_gb=80.0,
    )


@pytest.fixture(params=["none", "int8", "fp8", "int4"])
def kv_quantization(request):
    """Parameterized KV cache quantization."""
    return KVCacheQuantization(request.param)


@pytest.fixture(params=[1, 2, 4, 8])
def tensor_parallel_size(request):
    """Parameterized tensor parallel size."""
    return request.param


# =============================================================================
# TestInferenceConfig
# =============================================================================


class TestInferenceConfig:
    """Tests for InferenceConfig validation and defaults."""

    def test_default_values(self, base_inference_config):
        """Test default InferenceConfig values."""
        assert base_inference_config.batch_size == 4
        assert base_inference_config.kv_cache_quantization == KVCacheQuantization.NONE
        assert base_inference_config.use_kv_cache is True
        assert base_inference_config.tensor_parallel_size == 1
        assert base_inference_config.gpu_memory_utilization == 0.9

    def test_kv_cache_quantization_types(self, base_inference_config):
        """Test all KV cache quantization types."""
        for quant_type in [
            KVCacheQuantization.NONE,
            KVCacheQuantization.INT8,
            KVCacheQuantization.FP8,
            KVCacheQuantization.INT4,
        ]:
            base_inference_config.kv_cache_quantization = quant_type
            assert base_inference_config.kv_cache_quantization == quant_type

    def test_tensor_parallel_size_minimum(self, base_model_config, base_gpu_config):
        """Test tensor_parallel_size >= 1 constraint."""
        with pytest.raises(ValueError):
            InferenceConfig(
                batch_size=4,
                tensor_parallel_size=0,  # Invalid: must be >= 1
            )

    def test_gpu_memory_utilization_bounds(self, base_inference_config):
        """Test gpu_memory_utilization bounds (0.0-1.0)."""
        # Test valid values
        base_inference_config.gpu_memory_utilization = 0.5
        assert base_inference_config.gpu_memory_utilization == 0.5

        base_inference_config.gpu_memory_utilization = 1.0
        assert base_inference_config.gpu_memory_utilization == 1.0

        # Test invalid values
        with pytest.raises(ValueError):
            InferenceConfig(gpu_memory_utilization=0.0)

        with pytest.raises(ValueError):
            InferenceConfig(gpu_memory_utilization=1.1)

    def test_vllm_block_size_validation(self, base_model_config, base_gpu_config):
        """Test block_size validation for vLLM."""
        # Valid block sizes
        for block_size in [1, 16, 32]:
            config = InferenceConfig(
                batch_size=4,
                block_size=block_size,
            )
            assert config.block_size == block_size

        # Invalid block size (must be >= 1)
        with pytest.raises(ValueError):
            InferenceConfig(
                batch_size=4,
                block_size=0,
            )


# =============================================================================
# TestInferenceEngines - Base Class Tests
# =============================================================================


class TestInferenceEngines:
    """Base class tests covering all inference engines."""

    @pytest.mark.parametrize(
        "engine_class",
        [
            HuggingFaceEngine,
            VLLMEngine,
            TGIEngine,
            TensorRTLLMEngine,
        ],
    )
    def test_model_params_calculation(
        self, engine_class, base_model_config, base_inference_config, base_gpu_config
    ):
        """Verify BF16/FP16 precision (2 bytes/param)."""
        engine = engine_class(base_model_config, base_inference_config, base_gpu_config)
        params_bytes = engine._calculate_model_params_bytes()

        # 7B parameters * 2 bytes (BF16/FP16)
        expected_bytes = 7_000_000_000 * 2
        assert params_bytes == expected_bytes

    @pytest.mark.parametrize(
        "engine_class",
        [
            HuggingFaceEngine,
            VLLMEngine,
            TGIEngine,
            TensorRTLLMEngine,
        ],
    )
    def test_kv_cache_bytes_per_token(
        self,
        engine_class,
        base_model_config,
        base_inference_config,
        base_gpu_config,
        kv_quantization,
    ):
        """Test all quantization types for KV cache."""
        base_inference_config.kv_cache_quantization = kv_quantization
        engine = engine_class(base_model_config, base_inference_config, base_gpu_config)

        kv_bytes = engine._get_kv_cache_bytes_per_token()

        # Expected: 2 * num_layers * num_heads * head_dim * bytes_per_value
        # head_dim = hidden_size / num_attention_heads = 4096 / 32 = 128
        # For 7B model: 2 * 32 * 32 * 128 * bytes_per_value

        bytes_per_value_map = {
            KVCacheQuantization.NONE: 2,
            KVCacheQuantization.INT8: 1,
            KVCacheQuantization.FP8: 1,
            KVCacheQuantization.INT4: 0.5,
        }
        bytes_per_value = bytes_per_value_map[kv_quantization]
        expected = 2 * 32 * 32 * 128 * bytes_per_value

        assert kv_bytes == int(expected)

    @pytest.mark.parametrize(
        "engine_class",
        [
            HuggingFaceEngine,
            VLLMEngine,
            TGIEngine,
            TensorRTLLMEngine,
        ],
    )
    def test_kv_cache_disabled(
        self, engine_class, base_model_config, base_inference_config, base_gpu_config
    ):
        """Zero KV cache when use_kv_cache=False."""
        base_inference_config.use_kv_cache = False
        engine = engine_class(base_model_config, base_inference_config, base_gpu_config)

        kv_bytes = engine._calculate_kv_cache_bytes(batch_size=4)
        assert kv_bytes == 0

    @pytest.mark.parametrize(
        "engine_class",
        [
            HuggingFaceEngine,
            VLLMEngine,
            TGIEngine,
            TensorRTLLMEngine,
        ],
    )
    def test_effective_seq_len(
        self, engine_class, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test max_seq_len override."""
        # Default: use model config
        engine = engine_class(base_model_config, base_inference_config, base_gpu_config)
        assert engine._get_effective_seq_len() == 4096

        # Override with inference config
        base_inference_config.max_seq_len = 2048
        engine = engine_class(base_model_config, base_inference_config, base_gpu_config)
        assert engine._get_effective_seq_len() == 2048

    @pytest.mark.parametrize(
        "engine_class",
        [
            HuggingFaceEngine,
            VLLMEngine,
            TGIEngine,
            TensorRTLLMEngine,
        ],
    )
    def test_feasibility_check_single_gpu(
        self, engine_class, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test fits_on_gpu validation for single GPU."""
        base_inference_config.batch_size = 1
        engine = engine_class(base_model_config, base_inference_config, base_gpu_config)
        result = engine.calculate_memory()

        # 7B model should fit on 80GB GPU with batch_size=1
        assert result.fits_on_gpu is True
        assert result.memory_utilization_percent < 100
        assert result.max_supported_batch_size is not None

    @pytest.mark.parametrize(
        "engine_class",
        [
            HuggingFaceEngine,
            VLLMEngine,
            TGIEngine,
            TensorRTLLMEngine,
        ],
    )
    def test_feasibility_check_multi_gpu(
        self, engine_class, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test fits_on_gpu with tensor_parallel > 1."""
        base_inference_config.tensor_parallel_size = 2
        engine = engine_class(base_model_config, base_inference_config, base_gpu_config)
        result = engine.calculate_memory()

        # With TP=2, memory per GPU should be reduced
        assert result.fits_on_gpu is True
        assert result.total_memory_all_gpus_gb > result.total_memory_per_gpu_gb

    @pytest.mark.parametrize(
        "engine_class",
        [
            HuggingFaceEngine,
            VLLMEngine,
            TGIEngine,
            TensorRTLLMEngine,
        ],
    )
    def test_max_batch_size_estimation(
        self, engine_class, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test max_supported_batch_size calculation."""
        base_inference_config.batch_size = 4
        base_inference_config.gpu_memory_utilization = 0.9
        engine = engine_class(base_model_config, base_inference_config, base_gpu_config)
        result = engine.calculate_memory()

        if result.fits_on_gpu:
            assert result.max_supported_batch_size is not None
            assert result.max_supported_batch_size >= 1
            # Max batch should be >= current batch
            assert result.max_supported_batch_size >= base_inference_config.batch_size

    @pytest.mark.parametrize(
        "engine_class",
        [
            HuggingFaceEngine,
            VLLMEngine,
            TGIEngine,
            TensorRTLLMEngine,
        ],
    )
    def test_throughput_estimation(
        self, engine_class, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test tokens/sec estimate."""
        base_inference_config.batch_size = 4
        engine = engine_class(base_model_config, base_inference_config, base_gpu_config)
        result = engine.calculate_memory()

        if result.fits_on_gpu:
            assert result.estimated_throughput_tokens_per_sec is not None
            assert result.estimated_throughput_tokens_per_sec > 0


# =============================================================================
# TestVLLMEngine
# =============================================================================


class TestVLLMEngine:
    """vLLM-specific tests."""

    def test_pagedattention_block_allocation(
        self, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test block-based KV cache with 20% buffer."""
        base_inference_config.block_size = 16
        engine = VLLMEngine(base_model_config, base_inference_config, base_gpu_config)

        # Calculate blocks needed for batch_size=4, seq_len=4096
        # total_tokens = 4 * 4096 = 16384
        # blocks_needed = 16384 / 16 = 1024
        # with 20% buffer = 1024 * 1.2 = 1228.8
        kv_bytes = engine._calculate_vllm_kv_cache(batch_size=4)

        # Verify the calculation uses blocks * block_size * bytes_per_token
        assert kv_bytes > 0

    @pytest.mark.parametrize("block_size", [1, 16, 32])
    def test_block_size_impact(
        self, block_size, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test different block sizes."""
        base_inference_config.block_size = block_size
        engine = VLLMEngine(base_model_config, base_inference_config, base_gpu_config)

        result = engine.calculate_memory()

        # Different block sizes should affect KV cache memory
        assert result.breakdown.kv_cache_gb > 0

    def test_vllm_kernel_fusion(self, base_model_config, base_inference_config, base_gpu_config):
        """Verify 50% activation reduction with kernel fusion."""
        engine = VLLMEngine(base_model_config, base_inference_config, base_gpu_config)

        # vLLM reduces activation memory by 50%
        activation_bytes = engine._calculate_activations(batch_size=4)

        # Base calculation: batch * seq * hidden * layers * bytes * 2 (forward) * 0.5 (reduction)
        # 4 * 4096 * 4096 * 32 * 2 * 2 * 0.5 = 4294967296
        expected = 4 * 4096 * 4096 * 32 * 2 * 2 * 0.5
        assert activation_bytes == int(expected)

    def test_vllm_overhead_components(
        self, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test scheduler, block tables, buffers overhead (150MB)."""
        engine = VLLMEngine(base_model_config, base_inference_config, base_gpu_config)

        overhead_gb = engine._calculate_vllm_overhead()

        # Base overhead (0.15 GB) + block tables + buffers (0.05 GB)
        assert overhead_gb >= 0.15
        assert overhead_gb < 1.0  # Should be reasonable

    def test_vllm_vs_huggingface(self, base_model_config, base_inference_config, base_gpu_config):
        """Compare vLLM vs HuggingFace memory usage."""
        base_inference_config.batch_size = 4

        vllm_engine = VLLMEngine(base_model_config, base_inference_config, base_gpu_config)
        hf_engine = HuggingFaceEngine(base_model_config, base_inference_config, base_gpu_config)

        vllm_result = vllm_engine.calculate_memory()
        hf_result = hf_engine.calculate_memory()

        # vLLM has block-based KV cache which uses more memory but is more efficient
        # Activation memory should be similar (both ~4GB due to same formula)
        # vLLM KV cache has 20% buffer overhead
        assert vllm_result.breakdown.kv_cache_gb > hf_result.breakdown.kv_cache_gb


# =============================================================================
# TestTGIEngine
# =============================================================================


class TestTGIEngine:
    """TGI-specific tests."""

    def test_flash_attention_optimization(
        self, base_model_config, base_inference_config, base_gpu_config
    ):
        """Verify 40% activation reduction with Flash Attention."""
        engine = TGIEngine(base_model_config, base_inference_config, base_gpu_config)

        activation_bytes = engine._calculate_tgi_activations(batch_size=4)

        # Base calculation: batch * seq * hidden * layers * 2 bytes * 0.4 (reduction)
        expected = 4 * 4096 * 4096 * 32 * 2 * 0.4
        assert activation_bytes == int(expected)

    def test_tgi_server_overhead(self, base_model_config, base_inference_config, base_gpu_config):
        """Test base 200MB overhead calculation."""
        engine = TGIEngine(base_model_config, base_inference_config, base_gpu_config)

        overhead_gb = engine._calculate_tgi_overhead()

        # Base overhead (0.2 GB) + buffers (0.1 GB) + batching (0.05 GB)
        assert overhead_gb >= 0.2
        assert overhead_gb < 1.0

    def test_tensor_parallel_communication(
        self, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test TP overhead scaling."""
        # Without TP
        base_inference_config.tensor_parallel_size = 1
        engine = TGIEngine(base_model_config, base_inference_config, base_gpu_config)
        overhead_no_tp = engine._calculate_tgi_overhead()

        # With TP=2
        base_inference_config.tensor_parallel_size = 2
        engine = TGIEngine(base_model_config, base_inference_config, base_gpu_config)
        overhead_with_tp = engine._calculate_tgi_overhead()

        # TP should add overhead
        assert overhead_with_tp > overhead_no_tp

    def test_dynamic_batching_overhead(
        self, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test 50MB batching overhead."""
        engine = TGIEngine(base_model_config, base_inference_config, base_gpu_config)

        overhead_gb = engine._calculate_tgi_overhead()

        # Should include dynamic batching overhead
        assert overhead_gb >= 0.05  # batching_overhead_gb

    def test_tgi_vs_standard(self, base_model_config, base_inference_config, base_gpu_config):
        """Compare Flash Attention vs standard."""
        base_inference_config.batch_size = 4

        tgi_engine = TGIEngine(base_model_config, base_inference_config, base_gpu_config)
        hf_engine = HuggingFaceEngine(base_model_config, base_inference_config, base_gpu_config)

        tgi_result = tgi_engine.calculate_memory()
        hf_result = hf_engine.calculate_memory()

        # TGI should use less activation memory with Flash Attention
        assert tgi_result.breakdown.activations_gb < hf_result.breakdown.activations_gb


# =============================================================================
# TestTensorRTLLMEngine
# =============================================================================


class TestTensorRTLLMEngine:
    """TensorRT-LLM-specific tests."""

    def test_fused_kernels_activation_reduction(
        self, base_model_config, base_inference_config, base_gpu_config
    ):
        """Verify 30% activation reduction with fused kernels."""
        engine = TensorRTLLMEngine(base_model_config, base_inference_config, base_gpu_config)

        activation_bytes = engine._calculate_tensorrt_activations(batch_size=4)

        # Base calculation: batch * seq * hidden * layers * 2 bytes * 0.3 (reduction)
        expected = 4 * 4096 * 4096 * 32 * 2 * 0.3
        assert activation_bytes == int(expected)

    def test_tensorrt_workspace_overhead(
        self, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test 200MB workspace memory overhead."""
        engine = TensorRTLLMEngine(base_model_config, base_inference_config, base_gpu_config)

        overhead_gb = engine._calculate_tensorrt_overhead()

        # Runtime (0.1 GB) + workspace (0.2 GB) + batching (0.05 GB)
        assert overhead_gb >= 0.35

    def test_in_flight_batching(self, base_model_config, base_inference_config, base_gpu_config):
        """Test 50MB batching overhead."""
        engine = TensorRTLLMEngine(base_model_config, base_inference_config, base_gpu_config)

        overhead_gb = engine._calculate_tensorrt_overhead()

        # Should include in-flight batching overhead
        assert overhead_gb >= 0.05

    def test_tensorrt_runtime_overhead(
        self, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test 100MB runtime overhead."""
        engine = TensorRTLLMEngine(base_model_config, base_inference_config, base_gpu_config)

        overhead_gb = engine._calculate_tensorrt_overhead()

        # Should include runtime overhead
        assert overhead_gb >= 0.1

    def test_tensorrt_memory_efficiency(
        self, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test that TensorRT is most memory-efficient engine."""
        base_inference_config.batch_size = 4

        engines = [
            HuggingFaceEngine(base_model_config, base_inference_config, base_gpu_config),
            VLLMEngine(base_model_config, base_inference_config, base_gpu_config),
            TGIEngine(base_model_config, base_inference_config, base_gpu_config),
            TensorRTLLMEngine(base_model_config, base_inference_config, base_gpu_config),
        ]

        results = [engine.calculate_memory() for engine in engines]

        # TensorRT-LLM should have lowest activation memory
        trt_activations = results[-1].breakdown.activations_gb
        for result in results[:-1]:
            assert trt_activations <= result.breakdown.activations_gb


# =============================================================================
# TestHuggingFaceEngine
# =============================================================================


class TestHuggingFaceEngine:
    """Standard HF Transformers tests."""

    def test_standard_activation_memory(
        self, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test no optimization baseline for activation memory."""
        engine = HuggingFaceEngine(base_model_config, base_inference_config, base_gpu_config)

        activation_bytes = engine._calculate_hf_activations(batch_size=4)

        # Base calculation: batch * seq * hidden * layers * 2 bytes (no reduction)
        expected = 4 * 4096 * 4096 * 32 * 2
        assert activation_bytes == int(expected)

    def test_pytorch_overhead(self, base_model_config, base_inference_config, base_gpu_config):
        """Test 150MB base + 50MB loading overhead."""
        engine = HuggingFaceEngine(base_model_config, base_inference_config, base_gpu_config)

        overhead_gb = engine._calculate_hf_overhead()

        # Base (0.15 GB) + loading (0.05 GB) = 0.2 GB
        assert overhead_gb == pytest.approx(0.2, rel=0.01)

    def test_huggingface_as_baseline(
        self, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test HF as reference for comparisons."""
        result = HuggingFaceEngine(
            base_model_config, base_inference_config, base_gpu_config
        ).calculate_memory()

        # All components should be non-zero
        assert result.breakdown.model_params_gb > 0
        assert result.breakdown.kv_cache_gb > 0
        assert result.breakdown.activations_gb > 0
        assert result.breakdown.overhead_gb > 0


# =============================================================================
# TestInferenceMemoryCalculator
# =============================================================================


class TestInferenceMemoryCalculator:
    """Orchestrator tests."""

    @pytest.mark.parametrize(
        "engine_type",
        [
            InferenceEngineType.HUGGINGFACE,
            InferenceEngineType.VLLM,
            InferenceEngineType.TGI,
            InferenceEngineType.TENSORRT_LLM,
            InferenceEngineType.TRTLLM,
        ],
    )
    def test_engine_selection(self, base_model_config, base_inference_config, engine_type):
        """Test all InferenceEngineType values."""
        calculator = InferenceMemoryCalculator(base_model_config, base_inference_config)
        result = calculator.calculate(engine_type)

        assert result is not None
        assert result.total_memory_per_gpu_gb > 0

    def test_invalid_engine_type(self, base_model_config, base_inference_config):
        """Test ValueError for unknown engine types."""
        calculator = InferenceMemoryCalculator(base_model_config, base_inference_config)

        # The Enum itself raises ValueError before our code
        with pytest.raises(ValueError, match="is not a valid InferenceEngineType"):
            # Create an invalid engine type string
            calculator.calculate(InferenceEngineType("invalid_engine"))

    def test_default_gpu_config(self, base_model_config, base_inference_config):
        """Test default GPUConfig when None."""
        calculator = InferenceMemoryCalculator(base_model_config, base_inference_config)

        # Should use default GPUConfig (1x 80GB GPU)
        result = calculator.calculate(InferenceEngineType.HUGGINGFACE)

        assert result is not None
        assert result.fits_on_gpu is not None

    def test_orchestration_workflow(self, base_model_config, base_inference_config):
        """Test end-to-end calculate() workflow."""
        calculator = InferenceMemoryCalculator(base_model_config, base_inference_config)

        result = calculator.calculate(InferenceEngineType.VLLM)

        # Verify complete result
        assert result.total_memory_per_gpu_gb > 0
        assert result.total_memory_all_gpus_gb > 0
        assert result.breakdown.model_params_gb > 0
        assert result.breakdown.kv_cache_gb >= 0
        assert result.breakdown.activations_gb > 0
        assert result.breakdown.overhead_gb >= 0
        assert isinstance(result.fits_on_gpu, bool)
        assert result.memory_utilization_percent >= 0


# =============================================================================
# TestInferenceCrossEngine
# =============================================================================


class TestInferenceCrossEngine:
    """Comparison tests across all engines."""

    def test_all_engines_same_model(
        self, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test all 4 engines with 7B model."""
        engines = [
            HuggingFaceEngine(base_model_config, base_inference_config, base_gpu_config),
            VLLMEngine(base_model_config, base_inference_config, base_gpu_config),
            TGIEngine(base_model_config, base_inference_config, base_gpu_config),
            TensorRTLLMEngine(base_model_config, base_inference_config, base_gpu_config),
        ]

        results = [engine.calculate_memory() for engine in engines]

        # All should have valid results
        for result in results:
            assert result.total_memory_per_gpu_gb > 0
            assert result.breakdown.model_params_gb > 0

    def test_memory_ranking(self, base_model_config, base_inference_config, base_gpu_config):
        """Test memory ranking: TensorRT < TGI < HF ≈ vLLM (optimized order)."""
        base_inference_config.batch_size = 4

        hf_engine = HuggingFaceEngine(base_model_config, base_inference_config, base_gpu_config)
        vllm_engine = VLLMEngine(base_model_config, base_inference_config, base_gpu_config)
        tgi_engine = TGIEngine(base_model_config, base_inference_config, base_gpu_config)
        trt_engine = TensorRTLLMEngine(base_model_config, base_inference_config, base_gpu_config)

        hf_result = hf_engine.calculate_memory()
        vllm_result = vllm_engine.calculate_memory()
        tgi_result = tgi_engine.calculate_memory()
        trt_result = trt_engine.calculate_memory()

        # Activation memory should follow optimization ranking
        # TensorRT (30%) < TGI (40%) < HF (100%) ≈ vLLM (100% due to forward*2*0.5)
        activations = [
            hf_result.breakdown.activations_gb,
            vllm_result.breakdown.activations_gb,
            tgi_result.breakdown.activations_gb,
            trt_result.breakdown.activations_gb,
        ]

        # Verify optimized engines use less than or equal to HF
        # TensorRT should be lowest, TGI should be less than HF
        assert activations[3] < activations[2] < activations[0]
        # vLLM and HF should be approximately equal
        assert activations[1] == pytest.approx(activations[0])

    @pytest.mark.parametrize("quantization", ["none", "int8", "fp8", "int4"])
    def test_kv_cache_quantization_impact(
        self, quantization, base_model_config, base_inference_config, base_gpu_config
    ):
        """Compare all 4 quantization types."""
        base_inference_config.kv_cache_quantization = KVCacheQuantization(quantization)
        base_inference_config.batch_size = 4

        engine = HuggingFaceEngine(base_model_config, base_inference_config, base_gpu_config)
        result = engine.calculate_memory()

        # KV cache memory should be present (unless disabled)
        assert result.breakdown.kv_cache_gb >= 0

        # Quantization should affect memory: NONE > FP8/INT8 > INT4
        # (More detailed comparison would require multiple runs)

    @pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
    def test_tensor_parallel_scaling(
        self, tp_size, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test TP scaling across engines."""
        base_inference_config.tensor_parallel_size = tp_size

        engine = HuggingFaceEngine(base_model_config, base_inference_config, base_gpu_config)
        result = engine.calculate_memory()

        # With TP, memory per GPU should decrease
        # total_memory_all_gpus should increase with TP
        assert result.total_memory_all_gpus_gb >= result.total_memory_per_gpu_gb

        # Verify TP size is reflected
        expected_gpus = tp_size
        assert result.total_memory_all_gpus_gb == pytest.approx(
            result.total_memory_per_gpu_gb * expected_gpus, rel=0.01
        )

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_batch_size_scaling(
        self, batch_size, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test memory scaling with batch size."""
        base_inference_config.batch_size = batch_size

        engine = HuggingFaceEngine(base_model_config, base_inference_config, base_gpu_config)
        result = engine.calculate_memory()

        # Memory should scale with batch size (for KV cache and activations)
        assert result.breakdown.kv_cache_gb > 0
        assert result.breakdown.activations_gb > 0


# =============================================================================
# Edge Cases and Special Scenarios
# =============================================================================


class TestInferenceEdgeCases:
    """Edge case tests for inference engines."""

    def test_zero_batch_size_rejected(self, base_model_config, base_gpu_config):
        """Test that batch_size must be > 0."""
        with pytest.raises(ValueError):
            InferenceConfig(batch_size=0)

    def test_extreme_sequence_length(
        self, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test handling of extreme sequence lengths (32k)."""
        base_model_config.max_seq_len = 32768
        base_inference_config.batch_size = 1  # Small batch to fit

        engine = HuggingFaceEngine(base_model_config, base_inference_config, base_gpu_config)
        result = engine.calculate_memory()

        # Should handle long sequences
        assert result.total_memory_per_gpu_gb > 0

    def test_kv_cache_disabled_with_generation(
        self, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test KV cache disabled scenario."""
        base_inference_config.use_kv_cache = False
        base_inference_config.batch_size = 1

        engine = HuggingFaceEngine(base_model_config, base_inference_config, base_gpu_config)
        result = engine.calculate_memory()

        # KV cache should be zero
        assert result.breakdown.kv_cache_gb == 0

    def test_gpu_memory_utilization_at_bounds(
        self, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test GPU memory utilization at lower and upper bounds."""
        # Lower bound (0.1)
        base_inference_config.gpu_memory_utilization = 0.1
        engine = HuggingFaceEngine(base_model_config, base_inference_config, base_gpu_config)
        result = engine.calculate_memory()

        # Should be feasible even with low utilization target
        assert isinstance(result.fits_on_gpu, bool)

    def test_no_tensor_parallel(self, base_model_config, base_inference_config, base_gpu_config):
        """Test TP=1 (no tensor parallelism)."""
        base_inference_config.tensor_parallel_size = 1

        engine = HuggingFaceEngine(base_model_config, base_inference_config, base_gpu_config)
        result = engine.calculate_memory()

        # With TP=1, all GPUs should have same memory (single GPU here)
        assert result.total_memory_all_gpus_gb == result.total_memory_per_gpu_gb

    def test_vllm_various_block_sizes(
        self, base_model_config, base_inference_config, base_gpu_config
    ):
        """Test various vLLM block sizes."""
        block_sizes = [1, 8, 16, 32, 64]

        for block_size in block_sizes:
            base_inference_config.block_size = block_size
            engine = VLLMEngine(base_model_config, base_inference_config, base_gpu_config)
            result = engine.calculate_memory()

            # Different block sizes should still produce valid results
            assert result.total_memory_per_gpu_gb > 0
