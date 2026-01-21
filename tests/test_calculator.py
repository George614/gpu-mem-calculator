"""Tests for GPU Memory Calculator."""

import pytest
from gpu_mem_calculator.core.models import (
    DType,
    EngineConfig,
    EngineType,
    GPUConfig,
    ModelConfig,
    OptimizerType,
    ParallelismConfig,
    TrainingConfig,
)
from gpu_mem_calculator.core.calculator import GPUMemoryCalculator
from gpu_mem_calculator.utils.precision import gb_from_params, get_precision_from_dtype


class TestPrecision:
    """Test precision utilities."""

    def test_get_precision(self):
        """Test getting precision from dtype string."""
        fp16 = get_precision_from_dtype("fp16")
        assert fp16.bytes_per_param == 2.0

        bf16 = get_precision_from_dtype("bf16")
        assert bf16.bytes_per_param == 2.0

        fp32 = get_precision_from_dtype("fp32")
        assert fp32.bytes_per_param == 4.0

    def test_gb_from_params(self):
        """Test converting parameters to GB."""
        # 1B parameters in FP16 = 2GB
        result = gb_from_params(1_000_000_000, "fp16")
        assert result == pytest.approx(2.0, rel=0.01)


class TestModelConfig:
    """Test ModelConfig validation."""

    def test_basic_config(self):
        """Test creating basic model config."""
        config = ModelConfig(
            name="test-model",
            num_parameters=1_000_000_000,
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            vocab_size=32000,
            max_seq_len=2048,
        )
        assert config.name == "test-model"
        assert config.num_parameters == 1_000_000_000

    def test_largest_layer_auto_calculation(self):
        """Test automatic largest layer calculation."""
        config = ModelConfig(
            name="test-model",
            num_parameters=1_000_000_000,
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
        )
        # Should auto-calculate based on hidden_size
        assert config.largest_layer_params is not None
        assert config.largest_layer_params > 0


class TestTrainingConfig:
    """Test TrainingConfig validation."""

    def test_basic_config(self):
        """Test creating basic training config."""
        config = TrainingConfig(
            batch_size=4,
            gradient_accumulation_steps=2,
            dtype=DType.BF16,
            optimizer=OptimizerType.ADAMW,
        )
        assert config.batch_size == 4
        assert config.effective_batch_size == 8  # 4 * 2 * 1 (default dp)

    def test_activation_checkpointing_bounds(self):
        """Test activation checkpointing level validation."""
        with pytest.raises(ValueError):
            TrainingConfig(activation_checkpointing=5)  # Too high

        with pytest.raises(ValueError):
            TrainingConfig(activation_checkpointing=-1)  # Too low


class TestCalculator:
    """Test main calculator."""

    def test_pytorch_ddp_calculation(self):
        """Test PyTorch DDP memory calculation."""
        model_config = ModelConfig(
            name="test-1b",
            num_parameters=1_000_000_000,
            num_layers=24,
            hidden_size=2048,
            num_attention_heads=16,
            vocab_size=32000,
            max_seq_len=2048,
        )

        training_config = TrainingConfig(
            batch_size=4,
            gradient_accumulation_steps=1,
            dtype=DType.BF16,
            optimizer=OptimizerType.ADAMW,
        )

        parallelism_config = ParallelismConfig(data_parallel_size=4)

        engine_config = EngineConfig(type=EngineType.PYTORCH_DDP)

        gpu_config = GPUConfig(num_gpus=4, gpu_memory_gb=40)

        calculator = GPUMemoryCalculator(
            model_config=model_config,
            training_config=training_config,
            parallelism_config=parallelism_config,
            engine_config=engine_config,
            gpu_config=gpu_config,
        )

        result = calculator.calculate()

        # Should have some memory requirement
        assert result.total_memory_per_gpu_gb > 0
        assert result.breakdown.model_params_gb > 0
        assert result.breakdown.gradients_gb > 0
        assert result.breakdown.optimizer_states_gb > 0

    def test_deepspeed_zero3_vs_ddp(self):
        """Test that ZeRO-3 uses less memory than DDP."""
        model_config = ModelConfig(
            name="test-7b",
            num_parameters=7_000_000_000,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=32000,
            max_seq_len=2048,
        )

        training_config = TrainingConfig(
            batch_size=4,
            dtype=DType.BF16,
        )

        parallelism_config = ParallelismConfig(data_parallel_size=8)

        gpu_config = GPUConfig(num_gpus=8, gpu_memory_gb=80)

        # PyTorch DDP
        ddp_engine = EngineConfig(type=EngineType.PYTORCH_DDP)
        ddp_calc = GPUMemoryCalculator(
            model_config=model_config,
            training_config=training_config,
            parallelism_config=parallelism_config,
            engine_config=ddp_engine,
            gpu_config=gpu_config,
        )
        ddp_result = ddp_calc.calculate()

        # DeepSpeed ZeRO-3
        zero_engine = EngineConfig(type=EngineType.DEEPSPEED, zero_stage=3)
        zero_calc = GPUMemoryCalculator(
            model_config=model_config,
            training_config=training_config,
            parallelism_config=parallelism_config,
            engine_config=zero_engine,
            gpu_config=gpu_config,
        )
        zero_result = zero_calc.calculate()

        # ZeRO-3 should use significantly less memory per GPU
        assert zero_result.total_memory_per_gpu_gb < ddp_result.total_memory_per_gpu_gb
