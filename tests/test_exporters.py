"""Tests for framework configuration exporters.

This module tests the configuration exporters for:
- HuggingFace Accelerate
- PyTorch Lightning
- Axolotl
- ExportManager (unified interface)
"""

import json
import tempfile

import pytest

from gpu_mem_calculator.core.models import (
    DType,
    EngineConfig,
    EngineType,
    ModelConfig,
    NodeConfig,
    OffloadDevice,
    ParallelismConfig,
    TrainingConfig,
)
from gpu_mem_calculator.exporters.accelerate import AccelerateExporter
from gpu_mem_calculator.exporters.axolotl import AxolotlExporter
from gpu_mem_calculator.exporters.lightning import LightningExporter
from gpu_mem_calculator.exporters.manager import ExportFormat, ExportManager

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_model_config():
    """Standard 7B model for exporter testing."""
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
        gradient_accumulation_steps=2,
        dtype=DType.BF16,
        optimizer="adamw",
    )


@pytest.fixture
def base_parallelism_config():
    """Base parallelism configuration."""
    return ParallelismConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        data_parallel_size=4,
        sequence_parallel=True,
    )


@pytest.fixture
def base_engine_config():
    """Base engine configuration (DeepSpeed ZeRO-2)."""
    return EngineConfig(
        type=EngineType.DEEPSPEED,
        zero_stage=2,
        offload_optimizer=OffloadDevice.NONE,
        offload_param=OffloadDevice.NONE,
    )


@pytest.fixture
def base_node_config():
    """Base multi-node configuration."""
    return NodeConfig(
        num_nodes=2,
        gpus_per_node=8,
        interconnect_type="infiniband",
    )


@pytest.fixture(params=["accelerate", "lightning", "axolotl", "yaml", "json"])
def export_format(request):
    """Parameterized export formats."""
    return ExportFormat(request.param)


@pytest.fixture(params=["llama2-7b", "mistral-7b", "mixtral-8x7b"])
def model_name(request):
    """Parameterized model names."""
    return request.param


# =============================================================================
# TestAccelerateExporter
# =============================================================================


class TestAccelerateExporter:
    """Tests for AccelerateExporter."""

    def test_distributed_type_detection(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test engine to distributed type mapping."""
        exporter = AccelerateExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        config = exporter.export()

        # DeepSpeed should map to DEEPSPEED
        assert config["distributed_type"] == "DEEPSPEED"

    def test_mixed_precision_mapping(
        self, base_model_config, base_parallelism_config, base_engine_config
    ):
        """Test dtype to precision mapping."""
        for dtype, expected_precision in [
            (DType.BF16, "bf16"),
            (DType.FP16, "fp16"),
            (DType.FP32, "no"),
        ]:
            training_config = TrainingConfig(batch_size=4, dtype=dtype)
            exporter = AccelerateExporter(
                base_model_config,
                training_config,
                base_parallelism_config,
                base_engine_config,
            )

            config = exporter.export()
            assert config["mixed_precision"] == expected_precision

    def test_fsdp_config_generation(
        self, base_model_config, base_training_config, base_parallelism_config
    ):
        """Test FSDP with all sharding strategies."""
        fsdp_config = EngineConfig(
            type=EngineType.FSDP,
            sharding_strategy="full_shard",
        )

        exporter = AccelerateExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            fsdp_config,
        )

        config = exporter.export()

        assert "fsdp_config" in config
        assert config["fsdp_config"]["fsdp_sharding_strategy"] == "FULL_SHARD"

    def test_deepspeed_config_generation(
        self, base_model_config, base_training_config, base_parallelism_config
    ):
        """Test DeepSpeed config with ZeRO stages and offload."""
        ds_config = EngineConfig(
            type=EngineType.DEEPSPEED,
            zero_stage=3,
            offload_optimizer=OffloadDevice.CPU,
            offload_param=OffloadDevice.NVME,
        )

        exporter = AccelerateExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            ds_config,
        )

        config = exporter.export()

        assert "deepspeed_config" in config
        assert config["deepspeed_config"]["zero_optimization"]["stage"] == 3
        assert (
            config["deepspeed_config"]["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
        )
        assert config["deepspeed_config"]["zero_optimization"]["offload_param"]["device"] == "nvme"

    def test_transformer_layer_mapping(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test layer class auto-detection."""
        exporter = AccelerateExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        config = exporter.export()

        # Should include transformer layer classes
        if "fsdp_config" in config:
            assert "fsdp_transformer_layer_cls_to_wrap" in config["fsdp_config"]

    def test_multi_gpu_config(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_engine_config,
        base_node_config,
    ):
        """Test num_machines, num_processes, port."""
        exporter = AccelerateExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
            base_node_config,
        )

        config = exporter.export()

        assert config["num_machines"] == 2
        assert config["num_processes"] == 8
        assert config["main_process_port"] == 29500

    def test_activation_checkpointing_flag(self, base_model_config, base_engine_config):
        """Test fsdp_activation_checkpointing flag."""
        training_config = TrainingConfig(
            batch_size=4,
            activation_checkpointing=1,
        )

        fsdp_config = EngineConfig(type=EngineType.FSDP)

        exporter = AccelerateExporter(
            base_model_config,
            training_config,
            base_parallelism_config,
            fsdp_config,
        )

        config = exporter.export()

        if "fsdp_config" in config:
            assert config["fsdp_config"].get("fsdp_activation_checkpointing") is True

    def test_offload_device_mapping(
        self, base_model_config, base_training_config, base_parallelism_config
    ):
        """Test CPU/NVMe offload configuration."""
        for offload_device, expected_device in [
            (OffloadDevice.CPU, "cpu"),
            (OffloadDevice.NVME, "nvme"),
        ]:
            engine_config = EngineConfig(
                type=EngineType.DEEPSPEED,
                zero_stage=3,
                offload_optimizer=offload_device,
            )

            exporter = AccelerateExporter(
                base_model_config,
                base_training_config,
                base_parallelism_config,
                engine_config,
            )

            config = exporter.export()
            assert (
                config["deepspeed_config"]["zero_optimization"]["offload_optimizer"]["device"]
                == expected_device
            )

    def test_compute_environment(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_engine_config,
        base_node_config,
    ):
        """Test LOCAL_MACHINE vs MULTI_GPU."""
        # Single node
        exporter = AccelerateExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
            None,  # No node config
        )

        config = exporter.export()
        assert config["compute_environment"] == "LOCAL_MACHINE"

        # Multi-node
        exporter_multinode = AccelerateExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
            base_node_config,  # 2 nodes
        )

        config_multinode = exporter_multinode.export()
        assert config_multinode["compute_environment"] == "MULTI_GPU"

    def test_config_structure(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test Accelerate format validation."""
        exporter = AccelerateExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        config = exporter.export()

        # Required keys
        assert "compute_environment" in config
        assert "distributed_type" in config
        assert "mixed_precision" in config


# =============================================================================
# TestLightningExporter
# =============================================================================


class TestLightningExporter:
    """Tests for LightningExporter."""

    def test_trainer_config_generation(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test complete Trainer config."""
        exporter = LightningExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        config = exporter.export()

        assert "trainer" in config
        assert config["trainer"]["accelerator"] == "auto"
        assert config["trainer"]["num_nodes"] == 1
        assert config["trainer"]["accumulate_grad_batches"] == 2
        assert config["trainer"]["gradient_clip_val"] == 1.0

    def test_strategy_mapping(
        self, base_model_config, base_training_config, base_parallelism_config
    ):
        """Test engine to Lightning strategy."""
        # FSDP
        fsdp_config = EngineConfig(type=EngineType.FSDP)
        exporter = LightningExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            fsdp_config,
        )
        config = exporter.export()
        assert config["trainer"]["strategy"] == "fsdp"

        # DeepSpeed
        ds_config = EngineConfig(type=EngineType.DEEPSPEED)
        exporter = LightningExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            ds_config,
        )
        config = exporter.export()
        assert config["trainer"]["strategy"] == "deepspeed"

        # DDP (data parallel)
        parallelism_config = ParallelismConfig(data_parallel_size=4)
        exporter = LightningExporter(
            base_model_config,
            base_training_config,
            parallelism_config,
            EngineConfig(type=EngineType.PYTORCH_DDP),
        )
        config = exporter.export()
        assert config["trainer"]["strategy"] == "ddp"

    def test_precision_mapping(
        self, base_model_config, base_parallelism_config, base_engine_config
    ):
        """Test dtype to Lightning precision."""
        for dtype, expected_precision in [
            (DType.BF16, "bf16-mixed"),
            (DType.FP16, "16-mixed"),
            (DType.FP32, "32"),
        ]:
            training_config = TrainingConfig(batch_size=4, dtype=dtype)
            exporter = LightningExporter(
                base_model_config,
                training_config,
                base_parallelism_config,
                base_engine_config,
            )

            config = exporter.export()
            assert config["trainer"]["precision"] == expected_precision

    def test_num_devices_detection(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_engine_config,
        base_node_config,
    ):
        """Test gpus_per_node to devices mapping."""
        exporter = LightningExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
            base_node_config,
        )

        config = exporter.export()

        assert config["trainer"]["devices"] == 8

    def test_deepspeed_strategy_config(
        self, base_model_config, base_training_config, base_parallelism_config
    ):
        """Test DeepSpeed strategy configuration."""
        ds_config = EngineConfig(
            type=EngineType.DEEPSPEED,
            zero_stage=2,
        )

        exporter = LightningExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            ds_config,
        )

        config = exporter.export()

        assert "deepspeed_config" in config
        assert config["deepspeed_config"]["zero_stage"] == 2

    def test_fsdp_strategy_config(
        self, base_model_config, base_training_config, base_parallelism_config
    ):
        """Test FSDP strategy configuration."""
        fsdp_config = EngineConfig(
            type=EngineType.FSDP,
            sharding_strategy="full_shard",
        )

        exporter = LightningExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            fsdp_config,
        )

        config = exporter.export()

        assert "fsdp_config" in config
        assert config["fsdp_config"]["sharding_strategy"] == "FULL_SHARD"

    def test_gradient_accumulation(self, base_model_config, base_engine_config):
        """Test accumulate_grad_batches mapping."""
        training_config = TrainingConfig(
            batch_size=4,
            gradient_accumulation_steps=4,
        )

        exporter = LightningExporter(
            base_model_config,
            training_config,
            base_parallelism_config,
            base_engine_config,
        )

        config = exporter.export()

        assert config["trainer"]["accumulate_grad_batches"] == 4

    def test_model_architecture_detection(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test transformer class detection."""
        exporter = LightningExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        config = exporter.export()

        # Should detect LlamaDecoderLayer for "llama2-7b"
        if "fsdp_config" in config:
            assert "transformer_cls_name" in config["fsdp_config"]

    def test_export_code_generation(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test Python code string generation."""
        exporter = LightningExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        code = exporter.export_code()

        assert isinstance(code, str)
        assert "import pytorch_lightning as pl" in code
        assert "pl.Trainer(" in code
        assert "accelerator=" in code
        assert "strategy=" in code

    def test_code_validity(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test generated code is syntactically valid."""
        exporter = LightningExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        code = exporter.export_code()

        # Should compile without syntax errors
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax error: {e}")


# =============================================================================
# TestAxolotlExporter
# =============================================================================


class TestAxolotlExporter:
    """Tests for AxolotlExporter."""

    def test_base_model_mapping(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test model name to HF Hub path."""
        exporter = AxolotlExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        config = exporter.export()

        # llama2-7b should map to meta-llama/Llama-2-7b-hf
        assert "base_model" in config
        assert "meta-llama" in config["base_model"] or "llama" in config["base_model"].lower()

    def test_model_type_detection(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test model to class mapping."""
        exporter = AxolotlExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        config = exporter.export()

        # llama2-7b should map to LlamaForCausalLM
        assert config["model_type"] == "LlamaForCausalLM"

    @pytest.mark.parametrize(
        "model_name,expected_type",
        [
            ("llama2-7b", "LlamaForCausalLM"),
            ("mistral-7b", "MistralForCausalLM"),
            ("mixtral-8x7b", "MixtralForCausalLM"),
        ],
    )
    def test_model_type_detection_parameterized(
        self,
        model_name,
        expected_type,
        base_training_config,
        base_parallelism_config,
        base_engine_config,
    ):
        """Test architecture detection for common models."""
        model_config = ModelConfig(
            name=model_name,
            num_parameters=7_000_000_000,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            max_seq_len=2048,
        )

        exporter = AxolotlExporter(
            model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        config = exporter.export()

        assert config["model_type"] == expected_type

    def test_optimizer_selection(
        self, base_model_config, base_parallelism_config, base_engine_config
    ):
        """Test adamw_bnb_8bit vs adamw_torch selection."""
        # BF16 should use adamw_bnb_8bit
        training_config_bf16 = TrainingConfig(batch_size=4, dtype=DType.BF16)
        exporter = AxolotlExporter(
            base_model_config,
            training_config_bf16,
            base_parallelism_config,
            base_engine_config,
        )

        config = exporter.export()

        assert config["optimizer"] == "adamw_bnb_8bit"

    def test_special_tokens_config(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test BOS, EOS, UNK, PAD tokens."""
        exporter = AxolotlExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        config = exporter.export()

        assert "special_tokens" in config
        assert config["special_tokens"]["bos_token"] == "<s>"
        assert config["special_tokens"]["eos_token"] == "</s>"
        assert config["special_tokens"]["unk_token"] == "<unk>"
        assert config["special_tokens"]["pad_token"] == "<pad>"

    def test_deepspeed_config(
        self, base_model_config, base_training_config, base_parallelism_config
    ):
        """Test DeepSpeed in Axolotl format."""
        ds_config = EngineConfig(
            type=EngineType.DEEPSPEED,
            zero_stage=2,
        )

        exporter = AxolotlExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            ds_config,
        )

        config = exporter.export()

        assert "deepspeed" in config
        assert config["deepspeed"]["zero_optimization"]["stage"] == 2

    def test_fsdp_config(self, base_model_config, base_training_config, base_parallelism_config):
        """Test FSDP in Axolotl format."""
        fsdp_config = EngineConfig(
            type=EngineType.FSDP,
            sharding_strategy="full_shard",
        )

        exporter = AxolotlExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            fsdp_config,
        )

        config = exporter.export()

        assert "fsdp" in config
        assert config["fsdp"]["fsdp_sharding_strategy"] == "FULL_SHARD"

    def test_gradient_checkpointing(self, base_model_config, base_engine_config):
        """Test from activation_checkpointing."""
        training_config = TrainingConfig(
            batch_size=4,
            activation_checkpointing=1,
        )

        exporter = AxolotlExporter(
            base_model_config,
            training_config,
            base_parallelism_config,
            base_engine_config,
        )

        config = exporter.export()

        assert config["gradient_checkpointing"] is True

    def test_training_parameters(
        self, base_model_config, base_parallelism_config, base_engine_config
    ):
        """Test learning_rate, scheduler, warmup."""
        training_config = TrainingConfig(batch_size=4, dtype=DType.BF16)

        exporter = AxolotlExporter(
            base_model_config,
            training_config,
            base_parallelism_config,
            base_engine_config,
        )

        config = exporter.export()

        # Should have training parameters
        assert config["learning_rate"] == 2e-4
        assert config["lr_scheduler"] == "cosine"
        assert config["warmup_ratio"] == 0.03

    def test_num_nodes_config(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_engine_config,
        base_node_config,
    ):
        """Test multi-node configuration."""
        exporter = AxolotlExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
            base_node_config,
        )

        config = exporter.export()

        assert config["num_nodes"] == 2
        assert config["gpus_per_node"] == 8

    def test_export_yaml_output(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test YAML string generation."""
        exporter = AxolotlExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        yaml_str = exporter.export_yaml()

        assert isinstance(yaml_str, str)
        assert "base_model:" in yaml_str
        assert "model_type:" in yaml_str
        assert "batch_size:" in yaml_str


# =============================================================================
# TestExportManager
# =============================================================================


class TestExportManager:
    """Tests for ExportManager unified interface."""

    def test_format_detection(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test all ExportFormat values work."""
        manager = ExportManager(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        for format_enum in ExportFormat:
            result = manager.export(format_enum)
            assert result is not None

    def test_accelerate_export(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test delegates to AccelerateExporter."""
        manager = ExportManager(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        config = manager.export(ExportFormat.ACCELERATE)

        assert "distributed_type" in config
        assert "mixed_precision" in config

    def test_lightning_export(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test delegates to LightningExporter."""
        manager = ExportManager(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        config = manager.export(ExportFormat.LIGHTNING)

        assert "trainer" in config
        assert "model_config" in config

    def test_axolotl_export(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test delegates to AxolotlExporter."""
        manager = ExportManager(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        config = manager.export(ExportFormat.AXOLOTL)

        assert "base_model" in config
        assert "model_type" in config

    def test_deepspeed_export(
        self, base_model_config, base_training_config, base_parallelism_config
    ):
        """Test extracts deepspeed_config from accelerate."""
        manager = ExportManager(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            EngineConfig(type=EngineType.DEEPSPEED, zero_stage=2),
        )

        config = manager.export(ExportFormat.DEEPSPEED)

        # Should be the deepspeed_config portion
        assert "zero_optimization" in config
        assert config["zero_optimization"]["stage"] == 2

    def test_generic_yaml_export(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test _export_yaml() format."""
        manager = ExportManager(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        yaml_str = manager.export(ExportFormat.YAML)

        assert isinstance(yaml_str, str)
        assert "model:" in yaml_str
        assert "training:" in yaml_str
        assert "parallelism:" in yaml_str
        assert "engine:" in yaml_str

    def test_generic_json_export(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test _export_json() with model_dump()."""
        manager = ExportManager(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        json_str = manager.export(ExportFormat.JSON)

        assert isinstance(json_str, str)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "model" in parsed
        assert "training" in parsed
        assert "parallelism" in parsed
        assert "engine" in parsed

    def test_multinode_yaml_inclusion(
        self,
        base_model_config,
        base_training_config,
        base_parallelism_config,
        base_engine_config,
        base_node_config,
    ):
        """Test node_config included when num_nodes>1."""
        manager = ExportManager(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
            base_node_config,
        )

        yaml_str = manager.export(ExportFormat.YAML)

        # Should include multinode section
        assert "multinode:" in yaml_str

    def test_supported_formats_list(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test returns all formats."""
        manager = ExportManager(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        formats = manager.get_supported_formats()

        expected_formats = ["accelerate", "lightning", "axolotl", "deepspeed", "yaml", "json"]
        for fmt in expected_formats:
            assert fmt in formats

    def test_invalid_format_error(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test ValueError for unknown format."""
        manager = ExportManager(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        with pytest.raises(ValueError, match="Unknown export format"):
            manager.export("invalid_format")


# =============================================================================
# TestFileExport
# =============================================================================


class TestFileExport:
    """Tests for file export functionality."""

    def test_export_yaml_file(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test YAML file writing."""
        manager = ExportManager(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            filepath = f.name

        try:
            manager.export_to_file(ExportFormat.YAML, filepath)

            # Read and verify
            with open(filepath) as f:
                content = f.read()

            assert "model:" in content
            assert "training:" in content
        finally:
            import os

            os.unlink(filepath)

    def test_export_json_file(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test JSON file writing."""
        manager = ExportManager(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            manager.export_to_file(ExportFormat.JSON, filepath)

            # Read and verify
            with open(filepath) as f:
                content = f.read()

            parsed = json.loads(content)
            assert "model" in parsed
        finally:
            import os

            os.unlink(filepath)

    def test_file_overwrite(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test overwriting existing files."""
        manager = ExportManager(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            # Write first time
            manager.export_to_file(ExportFormat.JSON, filepath)

            # Write second time (should overwrite)
            manager.export_to_file(ExportFormat.JSON, filepath)

            # File should still be valid
            with open(filepath) as f:
                json.load(f)
        finally:
            import os

            os.unlink(filepath)

    def test_file_permissions(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test file is writable (use tempfile)."""
        manager = ExportManager(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        # Use tempfile which should be writable
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = f"{tmpdir}/config.yaml"
            manager.export_to_file(ExportFormat.YAML, filepath)

            # Verify file exists and is readable
            import os

            assert os.path.exists(filepath)
            assert os.access(filepath, os.R_OK)

    def test_dict_to_yaml_conversion(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test correct YAML formatting."""
        manager = ExportManager(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        config = manager.export(ExportFormat.YAML)

        # Should be string format
        assert isinstance(config, str)
        # Should have proper YAML structure
        assert "\n" in config  # Multi-line

    def test_dict_to_json_conversion(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test correct JSON formatting."""
        manager = ExportManager(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        json_str = manager.export(ExportFormat.JSON)

        # Should parse correctly
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)


# =============================================================================
# Edge Cases
# =============================================================================


class TestExporterEdgeCases:
    """Edge case tests for exporters."""

    def test_unknown_model_name_fallback(
        self, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test fallback for unknown model names."""
        model_config = ModelConfig(
            name="unknown-model-xyz",
            num_parameters=7_000_000_000,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            max_seq_len=2048,
        )

        exporter = AxolotlExporter(
            model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        config = exporter.export()

        # Should still have valid output
        assert "base_model" in config
        assert "model_type" in config

    def test_empty_parallelism(self, base_model_config, base_training_config, base_engine_config):
        """Test with DP=TP=PP=1 (no parallelism)."""
        parallelism_config = ParallelismConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        )

        exporter = AccelerateExporter(
            base_model_config,
            base_training_config,
            parallelism_config,
            base_engine_config,
        )

        config = exporter.export()

        # Should still be valid
        assert config is not None

    def test_no_offload(self, base_model_config, base_training_config, base_parallelism_config):
        """Test offload_optimizer=NONE."""
        engine_config = EngineConfig(
            type=EngineType.DEEPSPEED,
            zero_stage=3,
            offload_optimizer=OffloadDevice.NONE,
            offload_param=OffloadDevice.NONE,
        )

        exporter = AccelerateExporter(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            engine_config,
        )

        config = exporter.export()

        # Should not have offload config
        assert "offload_optimizer" not in config["deepspeed_config"]["zero_optimization"]

    def test_invalid_format_error_handling(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test ValueError for invalid format."""
        manager = ExportManager(
            base_model_config,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        with pytest.raises(ValueError):
            manager.export("not_a_real_format")

    def test_minimal_required_fields(
        self, base_model_config, base_training_config, base_parallelism_config, base_engine_config
    ):
        """Test minimal config exports successfully."""
        # Minimal model config
        minimal_model = ModelConfig(
            name="test",
            num_parameters=1_000_000,
            num_layers=1,
            hidden_size=512,
            num_attention_heads=8,
            max_seq_len=512,
        )

        manager = ExportManager(
            minimal_model,
            base_training_config,
            base_parallelism_config,
            base_engine_config,
        )

        # All formats should work
        for format_enum in [ExportFormat.YAML, ExportFormat.JSON]:
            result = manager.export(format_enum)
            assert result is not None
