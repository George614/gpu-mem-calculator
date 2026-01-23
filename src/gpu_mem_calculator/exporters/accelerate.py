"""HuggingFace Accelerate configuration exporter.

Generates configuration files for HuggingFace Accelerate distributed training.
"""

from gpu_mem_calculator.core.models import (
    DType,
    EngineConfig,
    EngineType,
    ModelConfig,
    NodeConfig,
    ParallelismConfig,
    TrainingConfig,
)


class AccelerateExporter:
    """Export configuration to HuggingFace Accelerate format.

    Accelerate uses a YAML configuration file to configure distributed
    training strategies including FSDP, DeepSpeed, and multi-GPU setups.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        parallelism_config: ParallelismConfig,
        engine_config: EngineConfig,
        node_config: NodeConfig | None = None,
    ) -> None:
        """Initialize the Accelerate exporter.

        Args:
            model_config: Model architecture configuration
            training_config: Training hyperparameters
            parallelism_config: Parallelism settings
            engine_config: Training engine configuration
            node_config: Multi-node configuration (optional)
        """
        self.model_config = model_config
        self.training_config = training_config
        self.parallelism_config = parallelism_config
        self.engine_config = engine_config
        self.node_config = node_config

    def export(self) -> dict:
        """Export configuration to Accelerate format.

        Returns:
            Dictionary compatible with Accelerate config file format
        """
        config: dict = {
            "compute_environment": (
                "LOCAL_MACHINE"
                if not self.node_config or self.node_config.num_nodes == 1
                else "MULTI_GPU"
            ),
            "distributed_type": self._get_distributed_type(),
            "mixed_precision": self._get_mixed_precision(),
            "downcast_bf16": self._get_downcast_bf16(),
        }

        # Add multi-GPU configuration
        if self.node_config and self.node_config.num_nodes > 1:
            config["num_machines"] = self.node_config.num_nodes
            config["num_processes"] = self.node_config.gpus_per_node or 1
            config["main_process_port"] = 29500
            config["main_training_function"] = "main"

        # Add FSDP configuration if using FSDP
        if self.engine_config.type == EngineType.FSDP:
            config["fsdp_config"] = self._get_fsdp_config()

        # Add DeepSpeed configuration if using DeepSpeed
        if self.engine_config.type == EngineType.DEEPSPEED:
            config["deepspeed_config"] = self._get_deepspeed_config()

        return config

    def _get_distributed_type(self) -> str:
        """Get Accelerate distributed type."""
        if self.engine_config.type == EngineType.FSDP:
            return "FSDP"
        elif self.engine_config.type == EngineType.DEEPSPEED:
            return "DEEPSPEED"
        elif self.parallelism_config.tensor_parallel_size > 1:
            return "MEGATRON_LM"
        elif self.parallelism_config.data_parallel_size > 1:
            return "MULTI_GPU"
        else:
            return "NO"

    def _get_mixed_precision(self) -> str:
        """Get mixed precision setting."""
        dtype_map = {
            DType.BF16: "bf16",
            DType.FP16: "fp16",
            DType.FP32: "no",
        }
        return dtype_map.get(self.training_config.dtype, "no")

    def _get_downcast_bf16(self) -> str:
        """Get downcast BF16 setting."""
        return "no" if self.training_config.dtype == DType.BF16 else "no"

    def _get_fsdp_config(self) -> dict:
        """Get FSDP-specific configuration."""
        sharding_strategy_map = {
            "no_shard": "NO_SHARD",
            "shard_grad_op": "SHARD_GRAD_OP",
            "full_shard": "FULL_SHARD",
        }

        config = {
            "fsdp_sharding_strategy": sharding_strategy_map.get(
                self.engine_config.sharding_strategy, "FULL_SHARD"
            ),
            "fsdp_offload_params": False,
            "fsdp_origin_params": True,
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_transformer_layer_cls_to_wrap": self._get_transformer_layer_cls(),
            "fsdp_backward_prefetch": "BACKWARD_PRE",
            "fsdp_forward_prefetch": False,
            "fsdp_use_orig_params": True,
            "fsdp_cpu_ram_efficient_loading": True,
        }

        # Add activation checkpointing if enabled
        if self.training_config.activation_checkpointing > 0:
            config["fsdp_activation_checkpointing"] = True

        return config

    def _get_deepspeed_config(self) -> dict:
        """Get DeepSpeed-specific configuration."""
        zero_opt: dict = {
            "stage": self.engine_config.zero_stage or 2,
        }

        config: dict = {
            "train_batch_size": self.training_config.batch_size,
            "train_micro_batch_size_per_gpu": self.training_config.batch_size,
            "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
            "zero_optimization": zero_opt,
            "bf16": {"enabled": self.training_config.dtype == DType.BF16},
            "fp16": {"enabled": self.training_config.dtype == DType.FP16},
            "gradient_clipping": 1.0,
            "prescale_gradients": False,
            "steps_per_print": 100,
        }

        # Add offload configuration if specified
        if self.engine_config.offload_optimizer != "none":
            config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu" if self.engine_config.offload_optimizer == "cpu" else "nvme",
                "pin_memory": True,
            }

        if self.engine_config.offload_param != "none":
            config["zero_optimization"]["offload_param"] = {
                "device": "cpu" if self.engine_config.offload_param == "cpu" else "nvme",
                "pin_memory": True,
            }

        return config

    def _get_transformer_layer_cls(self) -> list[str]:
        """Get transformer layer class names for FSDP auto-wrapping.

        Returns a list of common transformer layer class names based on model architecture.
        """
        # Common transformer layer class names
        common_layers = [
            "BertLayer",
            "GPTJBlock",
            "GPT2Block",
            "BloomBlock",
            "LlamaDecoderLayer",
            "MistralDecoderLayer",
            "MixtralDecoderLayer",
            "Qwen2DecoderLayer",
            "GemmaDecoderLayer",
        ]

        # Could be customized based on model_config.name
        return common_layers
