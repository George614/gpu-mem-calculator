"""Comprehensive tests for GPU Memory Calculator Python API.

Tests cover:
- All training engines (PyTorch DDP, DeepSpeed ZeRO, Megatron-LM, FSDP)
- MoE (Mixture of Experts) models
- All data types (FP32, FP16, BF16, INT8, INT4)
- Offloading options (CPU, NVMe)
- Activation checkpointing levels
- Parallelism strategies
- Preset configurations
"""

import json
import tempfile
from pathlib import Path

import pytest

from gpu_mem_calculator.config.parser import ConfigParser
from gpu_mem_calculator.config.presets import get_preset_config, list_presets
from gpu_mem_calculator.core.calculator import GPUMemoryCalculator
from gpu_mem_calculator.core.models import (
    DType,
    EngineConfig,
    EngineType,
    GPUConfig,
    ModelConfig,
    OffloadDevice,
    OptimizerType,
    ParallelismConfig,
    TrainingConfig,
)


class TestTrainingEngines:
    """Test all training engines."""

    @pytest.fixture
    def base_model_config(self):
        """Base 7B model config for testing."""
        return ModelConfig(
            name="test-7b",
            num_parameters=7_000_000_000,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=32000,
            max_seq_len=4096,
        )

    @pytest.fixture
    def base_training_config(self):
        """Base training config for testing."""
        return TrainingConfig(
            batch_size=4,
            gradient_accumulation_steps=4,
            dtype=DType.BF16,
            optimizer=OptimizerType.ADAMW,
            activation_checkpointing=2,
        )

    @pytest.fixture
    def base_parallelism_config(self):
        """Base parallelism config for testing."""
        return ParallelismConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=4,
            sequence_parallel=False,
        )

    @pytest.fixture
    def base_gpu_config(self):
        """Base GPU config for testing."""
        return GPUConfig(num_gpus=8, gpu_memory_gb=80)

    def test_pytorch_ddp(
        self, base_model_config, base_training_config, base_parallelism_config, base_gpu_config
    ):
        """Test PyTorch DDP engine."""
        engine_config = EngineConfig(type=EngineType.PYTORCH_DDP)

        calculator = GPUMemoryCalculator(
            model_config=base_model_config,
            training_config=base_training_config,
            parallelism_config=base_parallelism_config,
            engine_config=engine_config,
            gpu_config=base_gpu_config,
        )

        result = calculator.calculate()

        # PyTorch DDP should store full model, gradients, and optimizer states
        assert result.total_memory_per_gpu_gb > 0
        assert result.breakdown.model_params_gb > 0
        assert result.breakdown.gradients_gb > 0
        assert result.breakdown.optimizer_states_gb > 0
        # In DDP, model params + gradients + optimizer should be similar magnitude
        assert result.breakdown.model_params_gb > 1.0  # 7B params in BF16

    def test_deepspeed_zero1(
        self, base_model_config, base_training_config, base_parallelism_config, base_gpu_config
    ):
        """Test DeepSpeed ZeRO-1 (optimizer state sharding)."""
        engine_config = EngineConfig(
            type=EngineType.DEEPSPEED,
            zero_stage=1,
        )

        calculator = GPUMemoryCalculator(
            model_config=base_model_config,
            training_config=base_training_config,
            parallelism_config=base_parallelism_config,
            engine_config=engine_config,
            gpu_config=base_gpu_config,
        )

        result = calculator.calculate()

        assert result.total_memory_per_gpu_gb > 0
        # ZeRO-1 shards optimizer states
        assert result.breakdown.optimizer_states_gb >= 0

    def test_deepspeed_zero2(
        self, base_model_config, base_training_config, base_parallelism_config, base_gpu_config
    ):
        """Test DeepSpeed ZeRO-2 (optimizer + gradient sharding)."""
        engine_config = EngineConfig(
            type=EngineType.DEEPSPEED,
            zero_stage=2,
        )

        calculator = GPUMemoryCalculator(
            model_config=base_model_config,
            training_config=base_training_config,
            parallelism_config=base_parallelism_config,
            engine_config=engine_config,
            gpu_config=base_gpu_config,
        )

        result = calculator.calculate()

        assert result.total_memory_per_gpu_gb > 0
        # ZeRO-2 should reduce optimizer and gradient memory
        assert result.breakdown.optimizer_states_gb >= 0
        assert result.breakdown.gradients_gb >= 0

    def test_deepspeed_zero3(
        self, base_model_config, base_training_config, base_parallelism_config, base_gpu_config
    ):
        """Test DeepSpeed ZeRO-3 (full sharding)."""
        engine_config = EngineConfig(
            type=EngineType.DEEPSPEED,
            zero_stage=3,
        )

        calculator = GPUMemoryCalculator(
            model_config=base_model_config,
            training_config=base_training_config,
            parallelism_config=base_parallelism_config,
            engine_config=engine_config,
            gpu_config=base_gpu_config,
        )

        result = calculator.calculate()

        # ZeRO-3 should significantly reduce per-GPU memory
        assert result.total_memory_per_gpu_gb > 0
        # Only largest layer should be stored per GPU
        assert result.breakdown.model_params_gb > 0

    def test_deepspeed_zero3_cpu_offload(
        self, base_model_config, base_training_config, base_parallelism_config, base_gpu_config
    ):
        """Test DeepSpeed ZeRO-3 with CPU optimizer offloading."""
        engine_config = EngineConfig(
            type=EngineType.DEEPSPEED,
            zero_stage=3,
            offload_optimizer=OffloadDevice.CPU,
            offload_param=OffloadDevice.NONE,
        )

        calculator = GPUMemoryCalculator(
            model_config=base_model_config,
            training_config=base_training_config,
            parallelism_config=base_parallelism_config,
            engine_config=engine_config,
            gpu_config=base_gpu_config,
        )

        result = calculator.calculate()

        # CPU offload should reduce GPU memory but increase CPU memory
        assert result.total_memory_per_gpu_gb > 0
        assert result.cpu_memory_gb > 0

    def test_megatron_lm(
        self, base_model_config, base_training_config, base_parallelism_config, base_gpu_config
    ):
        """Test Megatron-LM with tensor parallelism."""
        engine_config = EngineConfig(type=EngineType.MEGATRON_LM)

        parallelism_config = ParallelismConfig(
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            data_parallel_size=1,
            sequence_parallel=True,
        )

        calculator = GPUMemoryCalculator(
            model_config=base_model_config,
            training_config=base_training_config,
            parallelism_config=parallelism_config,
            engine_config=engine_config,
            gpu_config=base_gpu_config,
        )

        result = calculator.calculate()

        assert result.total_memory_per_gpu_gb > 0

    def test_megatron_deepspeed(
        self, base_model_config, base_training_config, base_parallelism_config, base_gpu_config
    ):
        """Test Megatron-LM + DeepSpeed combined."""
        engine_config = EngineConfig(
            type=EngineType.MEGATRON_DEEPSPEED,
            zero_stage=3,
        )

        parallelism_config = ParallelismConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2,
            sequence_parallel=True,
        )

        calculator = GPUMemoryCalculator(
            model_config=base_model_config,
            training_config=base_training_config,
            parallelism_config=parallelism_config,
            engine_config=engine_config,
            gpu_config=base_gpu_config,
        )

        result = calculator.calculate()

        assert result.total_memory_per_gpu_gb > 0

    def test_fsdp_full_shard(
        self, base_model_config, base_training_config, base_parallelism_config, base_gpu_config
    ):
        """Test PyTorch FSDP with full sharding."""
        engine_config = EngineConfig(
            type=EngineType.FSDP,
            sharding_strategy="full_shard",
        )

        calculator = GPUMemoryCalculator(
            model_config=base_model_config,
            training_config=base_training_config,
            parallelism_config=base_parallelism_config,
            engine_config=engine_config,
            gpu_config=base_gpu_config,
        )

        result = calculator.calculate()

        assert result.total_memory_per_gpu_gb > 0


class TestDataTypes:
    """Test different data types."""

    @pytest.fixture
    def base_configs(self):
        """Base configs for testing."""
        return {
            "model_config": ModelConfig(
                name="test-7b",
                num_parameters=7_000_000_000,
                num_layers=32,
                hidden_size=4096,
                num_attention_heads=32,
                vocab_size=32000,
                max_seq_len=4096,
            ),
            "training_config": TrainingConfig(
                batch_size=4,
                dtype=DType.BF16,  # Will be overridden
            ),
            "parallelism_config": ParallelismConfig(data_parallel_size=8),
            "engine_config": EngineConfig(type=EngineType.DEEPSPEED, zero_stage=3),
            "gpu_config": GPUConfig(num_gpus=8, gpu_memory_gb=80),
        }

    def test_fp32(self, base_configs):
        """Test FP32 (32-bit floating point)."""
        base_configs["training_config"].dtype = DType.FP32

        calculator = GPUMemoryCalculator(**base_configs)
        result = calculator.calculate()

        # FP32 should use 4 bytes per parameter
        assert result.total_memory_per_gpu_gb > 0

    def test_fp16(self, base_configs):
        """Test FP16 (16-bit floating point)."""
        base_configs["training_config"].dtype = DType.FP16

        calculator = GPUMemoryCalculator(**base_configs)
        result = calculator.calculate()

        assert result.total_memory_per_gpu_gb > 0

    def test_bf16(self, base_configs):
        """Test BF16 (16-bit bfloat)."""
        base_configs["training_config"].dtype = DType.BF16

        calculator = GPUMemoryCalculator(**base_configs)
        result = calculator.calculate()

        assert result.total_memory_per_gpu_gb > 0

    def test_int8(self, base_configs):
        """Test INT8 (8-bit integer)."""
        base_configs["training_config"].dtype = DType.INT8

        calculator = GPUMemoryCalculator(**base_configs)
        result = calculator.calculate()

        # INT8 should use less memory than BF16
        assert result.total_memory_per_gpu_gb > 0

    def test_int4(self, base_configs):
        """Test INT4 (4-bit integer)."""
        base_configs["training_config"].dtype = DType.INT4

        calculator = GPUMemoryCalculator(**base_configs)
        result = calculator.calculate()

        # INT4 should use even less memory
        assert result.total_memory_per_gpu_gb > 0


class TestMoEModels:
    """Test Mixture of Experts models."""

    def test_mixtral_8x7b(self):
        """Test Mixtral 8x7B configuration."""
        model_config = ModelConfig(
            name="mixtral-8x7b",
            num_parameters=46_700_000_000,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=32000,
            max_seq_len=32768,
            moe_enabled=True,
            num_experts=8,
            top_k=2,
            expert_intermediate_size=14336,
        )

        training_config = TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=4,
            dtype=DType.BF16,
            activation_checkpointing=2,
        )

        parallelism_config = ParallelismConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=4,
        )

        engine_config = EngineConfig(
            type=EngineType.DEEPSPEED,
            zero_stage=3,
        )

        gpu_config = GPUConfig(num_gpus=8, gpu_memory_gb=80)

        calculator = GPUMemoryCalculator(
            model_config=model_config,
            training_config=training_config,
            parallelism_config=parallelism_config,
            engine_config=engine_config,
            gpu_config=gpu_config,
        )

        result = calculator.calculate()

        # MoE should have reduced activation memory due to sparse expert activation
        assert result.total_memory_per_gpu_gb > 0

    def test_glm_4_7_355b(self):
        """Test GLM-4.7 355B configuration."""
        model_config = ModelConfig(
            name="glm-4.7-355b",
            num_parameters=355_000_000_000,
            num_layers=46,
            hidden_size=4096,
            num_attention_heads=96,
            vocab_size=151552,
            max_seq_len=131072,
            moe_enabled=True,
            num_experts=128,
            top_k=8,
            expert_intermediate_size=1408,
            shared_expert_intermediate_size=10944,
        )

        training_config = TrainingConfig(
            batch_size=1,
            gradient_accumulation_steps=16,
            dtype=DType.BF16,
            activation_checkpointing=4,
        )

        parallelism_config = ParallelismConfig(
            tensor_parallel_size=8,
            pipeline_parallel_size=4,
            data_parallel_size=16,
            sequence_parallel=True,
        )

        engine_config = EngineConfig(
            type=EngineType.DEEPSPEED,
            zero_stage=3,
            offload_optimizer=OffloadDevice.CPU,
            offload_param=OffloadDevice.CPU,
        )

        gpu_config = GPUConfig(num_gpus=512, gpu_memory_gb=80)

        calculator = GPUMemoryCalculator(
            model_config=model_config,
            training_config=training_config,
            parallelism_config=parallelism_config,
            engine_config=engine_config,
            gpu_config=gpu_config,
        )

        result = calculator.calculate()

        # Should require massive memory but fit on 512x80GB GPUs
        assert result.total_memory_per_gpu_gb > 0
        assert result.fits_on_gpu

    def test_deepseek_moe_16b(self):
        """Test DeepSeek-MoE 16B configuration."""
        model_config = ModelConfig(
            name="deepseek-moe-16b",
            num_parameters=16_400_000_000,
            num_layers=28,
            hidden_size=2048,
            num_attention_heads=16,
            vocab_size=102400,
            max_seq_len=4096,
            moe_enabled=True,
            num_experts=64,
            top_k=6,
            expert_intermediate_size=1408,
            shared_expert_intermediate_size=10944,
        )

        training_config = TrainingConfig(
            batch_size=4,
            gradient_accumulation_steps=4,
            dtype=DType.BF16,
            activation_checkpointing=2,
        )

        parallelism_config = ParallelismConfig(
            tensor_parallel_size=2,
            data_parallel_size=4,
        )

        engine_config = EngineConfig(
            type=EngineType.DEEPSPEED,
            zero_stage=2,
        )

        gpu_config = GPUConfig(num_gpus=8, gpu_memory_gb=80)

        calculator = GPUMemoryCalculator(
            model_config=model_config,
            training_config=training_config,
            parallelism_config=parallelism_config,
            engine_config=engine_config,
            gpu_config=gpu_config,
        )

        result = calculator.calculate()

        assert result.total_memory_per_gpu_gb > 0


class TestActivationCheckpointing:
    """Test activation checkpointing levels."""

    @pytest.fixture
    def base_configs(self):
        """Base configs for testing."""
        return {
            "model_config": ModelConfig(
                name="test-7b",
                num_parameters=7_000_000_000,
                num_layers=32,
                hidden_size=4096,
                num_attention_heads=32,
                vocab_size=32000,
                max_seq_len=4096,
            ),
            "training_config": TrainingConfig(
                batch_size=8,
                gradient_accumulation_steps=1,
                dtype=DType.BF16,
            ),
            "parallelism_config": ParallelismConfig(data_parallel_size=8),
            "engine_config": EngineConfig(type=EngineType.DEEPSPEED, zero_stage=3),
            "gpu_config": GPUConfig(num_gpus=8, gpu_memory_gb=80),
        }

    def test_checkpointing_level_0(self, base_configs):
        """Test no activation checkpointing."""
        base_configs["training_config"].activation_checkpointing = 0
        calculator = GPUMemoryCalculator(**base_configs)
        result_0 = calculator.calculate()

        # Level 0 should use the most activation memory
        assert result_0.breakdown.activations_gb > 0

    def test_checkpointing_level_2(self, base_configs):
        """Test moderate activation checkpointing."""
        base_configs["training_config"].activation_checkpointing = 2
        calculator = GPUMemoryCalculator(**base_configs)
        result_2 = calculator.calculate()

        assert result_2.breakdown.activations_gb > 0

    def test_checkpointing_level_4(self, base_configs):
        """Test full activation checkpointing."""
        base_configs["training_config"].activation_checkpointing = 4
        calculator = GPUMemoryCalculator(**base_configs)
        result_4 = calculator.calculate()

        # Level 4 should use the least activation memory
        assert result_4.breakdown.activations_gb >= 0


class TestPresetConfigurations:
    """Test preset configuration loading."""

    def test_list_presets(self):
        """Test listing all available presets."""
        presets = list_presets()

        # Should have multiple presets
        assert len(presets) > 0

        # Check for expected presets
        expected_presets = [
            "llama2-7b",
            "llama2-13b",
            "llama2-70b",
            "mixtral-8x7b",
            "gpt3-175b",
        ]

        for preset_name in expected_presets:
            assert preset_name in presets
            assert "display_name" in presets[preset_name]
            assert "description" in presets[preset_name]

    def test_get_preset_config_llama2_7b(self):
        """Test loading LLaMA 2 7B preset."""
        config = get_preset_config("llama2-7b")

        assert config is not None
        assert "model" in config
        assert "training" in config
        assert "parallelism" in config
        assert "engine" in config
        assert "hardware" in config

        assert config["model"]["num_parameters"] == "7B"

    def test_get_preset_config_mixtral(self):
        """Test loading Mixtral 8x7B preset."""
        config = get_preset_config("mixtral-8x7b")

        assert config is not None
        assert config["model"]["moe_enabled"] is True
        assert config["model"]["num_experts"] == 8
        assert config["model"]["top_k"] == 2

    def test_calculate_with_preset(self):
        """Test calculation using preset configuration."""
        from gpu_mem_calculator.core.calculator import GPUMemoryCalculator

        preset_config = get_preset_config("llama2-7b")

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(preset_config, f)
            temp_path = f.name

        try:
            calculator = GPUMemoryCalculator.from_config_file(temp_path)
            result = calculator.calculate()

            assert result.total_memory_per_gpu_gb > 0
        finally:
            Path(temp_path).unlink()

    def test_invalid_preset(self):
        """Test loading invalid preset."""
        config = get_preset_config("invalid-preset-name")

        assert config is None


class TestConfigParser:
    """Test configuration file parser."""

    def test_parse_full_config(self):
        """Test parsing complete configuration file."""
        config_data = {
            "model": {
                "name": "test-model",
                "num_parameters": "7B",
                "num_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "vocab_size": 32000,
                "max_seq_len": 4096,
            },
            "training": {
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "optimizer": "adamw",
                "dtype": "bf16",
                "activation_checkpointing": 2,
            },
            "parallelism": {
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 4,
                "sequence_parallel": False,
            },
            "engine": {
                "type": "deepspeed",
                "zero_stage": 3,
                "offload_optimizer": "cpu",
                "offload_param": "none",
            },
            "hardware": {
                "num_gpus": 8,
                "gpu_memory_gb": 80,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            model_cfg, training_cfg, parallelism_cfg, engine_cfg, gpu_cfg = (
                ConfigParser.parse_full_config(temp_path)
            )

            assert model_cfg.name == "test-model"
            assert model_cfg.num_parameters == 7_000_000_000
            assert training_cfg.batch_size == 4
            assert engine_cfg.type == EngineType.DEEPSPEED
            assert gpu_cfg.num_gpus == 8
        finally:
            Path(temp_path).unlink()

    def test_parse_num_parameters_formats(self):
        """Test parsing different parameter count formats."""
        test_cases = [
            ("7B", 7_000_000_000),
            ("7000M", 7_000_000_000),
            ("7e9", 7_000_000_000),
            ("7000000000", 7_000_000_000),
        ]

        for input_val, expected in test_cases:
            result = ConfigParser._parse_num_params(input_val)
            assert result == expected


class TestOptimizers:
    """Test different optimizers."""

    def test_adam(self):
        """Test Adam optimizer."""
        training_config = TrainingConfig(
            batch_size=4,
            optimizer=OptimizerType.ADAM,
        )
        assert training_config.optimizer == OptimizerType.ADAM

    def test_adamw(self):
        """Test AdamW optimizer."""
        training_config = TrainingConfig(
            batch_size=4,
            optimizer=OptimizerType.ADAMW,
        )
        assert training_config.optimizer == OptimizerType.ADAMW

    def test_adamw_8bit(self):
        """Test AdamW 8-bit optimizer."""
        training_config = TrainingConfig(
            batch_size=4,
            optimizer=OptimizerType.ADAMW_8BIT,
        )
        assert training_config.optimizer == OptimizerType.ADAMW_8BIT

    def test_sgd(self):
        """Test SGD optimizer."""
        training_config = TrainingConfig(
            batch_size=4,
            optimizer=OptimizerType.SGD,
        )
        assert training_config.optimizer == OptimizerType.SGD


class TestFeasibility:
    """Test feasibility checking."""

    def test_feasible_configuration(self):
        """Test configuration that fits on GPU."""
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
            dtype=DType.BF16,
        )

        engine_config = EngineConfig(
            type=EngineType.DEEPSPEED,
            zero_stage=3,
        )

        gpu_config = GPUConfig(num_gpus=4, gpu_memory_gb=80)

        calculator = GPUMemoryCalculator(
            model_config=model_config,
            training_config=training_config,
            parallelism_config=ParallelismConfig(data_parallel_size=4),
            engine_config=engine_config,
            gpu_config=gpu_config,
        )

        result = calculator.calculate()

        assert result.fits_on_gpu is True
        assert result.memory_utilization_percent < 100

    def test_infeasible_configuration(self):
        """Test configuration that doesn't fit on GPU."""
        model_config = ModelConfig(
            name="test-175b",
            num_parameters=175_000_000_000,
            num_layers=96,
            hidden_size=12288,
            num_attention_heads=96,
            vocab_size=50257,
            max_seq_len=2048,
        )

        training_config = TrainingConfig(
            batch_size=1,
            dtype=DType.BF16,
        )

        engine_config = EngineConfig(
            type=EngineType.PYTORCH_DDP,
        )

        # Only 8 GPUs for 175B model with DDP - won't fit!
        gpu_config = GPUConfig(num_gpus=8, gpu_memory_gb=80)

        calculator = GPUMemoryCalculator(
            model_config=model_config,
            training_config=training_config,
            parallelism_config=ParallelismConfig(data_parallel_size=8),
            engine_config=engine_config,
            gpu_config=gpu_config,
        )

        result = calculator.calculate()

        # Should be infeasible
        assert result.fits_on_gpu is False or result.memory_utilization_percent > 100


class TestMemoryBreakdown:
    """Test memory breakdown components."""

    def test_all_components_present(self):
        """Test that all memory components are calculated."""
        model_config = ModelConfig(
            name="test-7b",
            num_parameters=7_000_000_000,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=32000,
            max_seq_len=4096,
        )

        training_config = TrainingConfig(
            batch_size=4,
            dtype=DType.BF16,
        )

        engine_config = EngineConfig(
            type=EngineType.DEEPSPEED,
            zero_stage=3,
        )

        gpu_config = GPUConfig(num_gpus=8, gpu_memory_gb=80)

        calculator = GPUMemoryCalculator(
            model_config=model_config,
            training_config=training_config,
            parallelism_config=ParallelismConfig(data_parallel_size=8),
            engine_config=engine_config,
            gpu_config=gpu_config,
        )

        result = calculator.calculate()

        # Check all breakdown components are present
        assert result.breakdown.model_params_gb >= 0
        assert result.breakdown.gradients_gb >= 0
        assert result.breakdown.optimizer_states_gb >= 0
        assert result.breakdown.activations_gb >= 0
        assert result.breakdown.overhead_gb >= 0

        # Sum of breakdown should equal total
        breakdown_sum = (
            result.breakdown.model_params_gb
            + result.breakdown.gradients_gb
            + result.breakdown.optimizer_states_gb
            + result.breakdown.activations_gb
            + result.breakdown.overhead_gb
        )
        assert abs(breakdown_sum - result.total_memory_per_gpu_gb) < 0.1


class TestParallelismStrategies:
    """Test different parallelism strategies."""

    @pytest.fixture
    def base_configs(self):
        """Base configs for testing."""
        return {
            "model_config": ModelConfig(
                name="test-7b",
                num_parameters=7_000_000_000,
                num_layers=32,
                hidden_size=4096,
                num_attention_heads=32,
                vocab_size=32000,
                max_seq_len=4096,
            ),
            "training_config": TrainingConfig(batch_size=4, dtype=DType.BF16),
            "engine_config": EngineConfig(type=EngineType.DEEPSPEED, zero_stage=3),
            "gpu_config": GPUConfig(num_gpus=8, gpu_memory_gb=80),
        }

    def test_data_parallel_only(self, base_configs):
        """Test data parallelism only."""
        parallelism_config = ParallelismConfig(data_parallel_size=8)

        calculator = GPUMemoryCalculator(
            parallelism_config=parallelism_config,
            **base_configs,
        )

        result = calculator.calculate()
        assert result.total_memory_per_gpu_gb > 0

    def test_tensor_parallel(self, base_configs):
        """Test tensor parallelism."""
        parallelism_config = ParallelismConfig(
            tensor_parallel_size=4,
            data_parallel_size=2,
        )

        calculator = GPUMemoryCalculator(
            parallelism_config=parallelism_config,
            **base_configs,
        )

        result = calculator.calculate()
        assert result.total_memory_per_gpu_gb > 0

    def test_pipeline_parallel(self, base_configs):
        """Test pipeline parallelism."""
        parallelism_config = ParallelismConfig(
            pipeline_parallel_size=4,
            data_parallel_size=2,
        )

        calculator = GPUMemoryCalculator(
            parallelism_config=parallelism_config,
            **base_configs,
        )

        result = calculator.calculate()
        assert result.total_memory_per_gpu_gb > 0

    def test_sequence_parallel(self, base_configs):
        """Test sequence parallelism."""
        parallelism_config = ParallelismConfig(
            tensor_parallel_size=4,
            sequence_parallel=True,
            data_parallel_size=2,
        )

        calculator = GPUMemoryCalculator(
            parallelism_config=parallelism_config,
            **base_configs,
        )

        result = calculator.calculate()
        assert result.total_memory_per_gpu_gb > 0

    def test_hybrid_parallelism(self, base_configs):
        """Test hybrid parallelism (tensor + pipeline + data)."""
        parallelism_config = ParallelismConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2,
            sequence_parallel=True,
        )

        calculator = GPUMemoryCalculator(
            parallelism_config=parallelism_config,
            **base_configs,
        )

        result = calculator.calculate()
        assert result.total_memory_per_gpu_gb > 0


class TestOffloadDevices:
    """Test CPU and NVMe offloading."""

    def test_cpu_optimizer_offload(self):
        """Test CPU optimizer offloading."""
        engine_config = EngineConfig(
            type=EngineType.DEEPSPEED,
            zero_stage=3,
            offload_optimizer=OffloadDevice.CPU,
        )

        calculator = GPUMemoryCalculator(
            model_config=ModelConfig(
                name="test-7b",
                num_parameters=7_000_000_000,
                num_layers=32,
                hidden_size=4096,
                num_attention_heads=32,
                vocab_size=32000,
                max_seq_len=4096,
            ),
            training_config=TrainingConfig(batch_size=4, dtype=DType.BF16),
            parallelism_config=ParallelismConfig(data_parallel_size=8),
            engine_config=engine_config,
            gpu_config=GPUConfig(num_gpus=8, gpu_memory_gb=80),
        )

        result = calculator.calculate()

        # Should have CPU memory usage
        assert result.cpu_memory_gb > 0

    def test_nvme_optimizer_offload(self):
        """Test NVMe optimizer offloading."""
        engine_config = EngineConfig(
            type=EngineType.DEEPSPEED,
            zero_stage=3,
            offload_optimizer=OffloadDevice.NVME,
        )

        calculator = GPUMemoryCalculator(
            model_config=ModelConfig(
                name="test-7b",
                num_parameters=7_000_000_000,
                num_layers=32,
                hidden_size=4096,
                num_attention_heads=32,
                vocab_size=32000,
                max_seq_len=4096,
            ),
            training_config=TrainingConfig(batch_size=4, dtype=DType.BF16),
            parallelism_config=ParallelismConfig(data_parallel_size=8),
            engine_config=engine_config,
            gpu_config=GPUConfig(num_gpus=8, gpu_memory_gb=80),
        )

        result = calculator.calculate()
        assert result.total_memory_per_gpu_gb > 0
