"""PyTorch Lightning configuration exporter.

Generates configuration and trainer setup for PyTorch Lightning training.
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


class LightningExporter:
    """Export configuration to PyTorch Lightning format.

    Lightning uses a Trainer class with various strategies for distributed
    training including DDP, FSDP, and DeepSpeed.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        parallelism_config: ParallelismConfig,
        engine_config: EngineConfig,
        node_config: NodeConfig | None = None,
    ) -> None:
        """Initialize the Lightning exporter.

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
        """Export configuration to Lightning Trainer format.

        Returns:
            Dictionary with Trainer configuration
        """
        config = {
            "trainer": {
                "accelerator": "auto",
                "devices": self._get_num_devices(),
                "num_nodes": self._get_num_nodes(),
                "strategy": self._get_strategy(),
                "precision": self._get_precision(),
                "max_epochs": 1,  # Placeholder
                "accumulate_grad_batches": self.training_config.gradient_accumulation_steps,
                "gradient_clip_val": 1.0,
                "log_every_n_steps": 50,
            },
            "model_config": {
                "model_name": self.model_config.name,
                "num_parameters": self.model_config.num_parameters,
                "hidden_size": self.model_config.hidden_size,
                "num_layers": self.model_config.num_layers,
                "num_attention_heads": self.model_config.num_attention_heads,
                "max_seq_len": self.model_config.max_seq_len,
            },
        }

        # Add strategy-specific configuration
        if self.engine_config.type == EngineType.DEEPSPEED:
            config["deepspeed_config"] = self._get_deepspeed_config()
        elif self.engine_config.type == EngineType.FSDP:
            config["fsdp_config"] = self._get_fsdp_config()

        return config

    def _get_num_devices(self) -> int | str:
        """Get number of devices."""
        if self.node_config and self.node_config.gpus_per_node:
            return self.node_config.gpus_per_node
        return "auto"

    def _get_num_nodes(self) -> int:
        """Get number of nodes."""
        if self.node_config:
            return self.node_config.num_nodes
        return 1

    def _get_strategy(self) -> str | dict:
        """Get Lightning training strategy."""
        if self.engine_config.type == EngineType.FSDP:
            return "fsdp"
        elif self.engine_config.type == EngineType.DEEPSPEED:
            return "deepspeed"
        elif self.parallelism_config.data_parallel_size > 1:
            return "ddp"
        else:
            return "auto"

    def _get_precision(self) -> str:
        """Get precision setting."""
        dtype_map = {
            DType.BF16: "bf16-mixed",
            DType.FP16: "16-mixed",
            DType.FP32: "32",
        }
        return dtype_map.get(self.training_config.dtype, "32")

    def _get_deepspeed_config(self) -> dict:
        """Get DeepSpeed configuration for Lightning."""
        zero_opt: dict = {
            "stage": self.engine_config.zero_stage or 2,
        }

        config: dict = {
            "zero_stage": self.engine_config.zero_stage or 2,
            "zero_optimization": zero_opt,
            "bf16": {"enabled": self.training_config.dtype == DType.BF16},
            "fp16": {"enabled": self.training_config.dtype == DType.FP16},
            "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": self.training_config.batch_size,
            "train_batch_size": self.training_config.batch_size
            * self.training_config.gradient_accumulation_steps,
        }

        # Add offload configuration
        if self.engine_config.offload_optimizer != "none":
            config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu" if self.engine_config.offload_optimizer == "cpu" else "nvme",
            }

        if self.engine_config.offload_param != "none":
            config["zero_optimization"]["offload_param"] = {
                "device": "cpu" if self.engine_config.offload_param == "cpu" else "nvme",
            }

        return config

    def _get_fsdp_config(self) -> dict:
        """Get FSDP configuration for Lightning."""
        sharding_strategy_map = {
            "no_shard": "NO_SHARD",
            "shard_grad_op": "SHARD_GRAD_OP",
            "full_shard": "FULL_SHARD",
        }

        config = {
            "sharding_strategy": sharding_strategy_map.get(
                self.engine_config.sharding_strategy, "FULL_SHARD"
            ),
            "cpu_ram_efficient_loading": True,
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "transformer_cls_name": self._get_transformer_cls_name(),
            "activation_checkpointing": self.training_config.activation_checkpointing > 0,
        }

        return config

    def _get_transformer_cls_name(self) -> str:
        """Get transformer class name for FSDP wrapping."""
        # Map common model names to their layer classes
        model_layer_map = {
            "llama": "LlamaDecoderLayer",
            "mistral": "MistralDecoderLayer",
            "mixtral": "MixtralDecoderLayer",
            "qwen": "Qwen2DecoderLayer",
            "gemma": "GemmaDecoderLayer",
            "bloom": "BloomBlock",
            "gpt2": "GPT2Block",
            "gptj": "GPTJBlock",
            "bert": "BertLayer",
        }

        # Try to match based on model name
        model_name_lower = self.model_config.name.lower()
        for key, value in model_layer_map.items():
            if key in model_name_lower:
                return value

        return "LlamaDecoderLayer"  # Default

    def export_code(self) -> str:
        """Generate Python code for Lightning Trainer setup.

        Returns:
            String with Python code
        """
        config = self.export()

        code = f"""import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy, FSDPStrategy

# Model configuration
model_config = {config["model_config"]}

# Trainer configuration
trainer = pl.Trainer(
    accelerator="{config["trainer"]["accelerator"]}",
    devices={config["trainer"]["devices"]},
    num_nodes={config["trainer"]["num_nodes"]},
    strategy="{config["trainer"]["strategy"]}",
    precision="{config["trainer"]["precision"]}",
    max_epochs={config["trainer"]["max_epochs"]},
    accumulate_grad_batches={config["trainer"]["accumulate_grad_batches"]},
    gradient_clip_val={config["trainer"]["gradient_clip_val"]},
    log_every_n_steps={config["trainer"]["log_every_n_steps"]},
)

# Training loop
# model = YourModel(model_config)
# trainer.fit(model)
"""
        return code
