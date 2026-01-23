"""Axolotl configuration exporter.

Generates configuration files for Axolotl fine-tuning framework.
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


class AxolotlExporter:
    """Export configuration to Axolotl YAML format.

    Axolotl uses a YAML configuration file for fine-tuning LLMs
    with various backends including DeepSpeed, FSDP, and XLA.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        parallelism_config: ParallelismConfig,
        engine_config: EngineConfig,
        node_config: NodeConfig | None = None,
    ) -> None:
        """Initialize the Axolotl exporter.

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
        """Export configuration to Axolotl YAML format.

        Returns:
            Dictionary compatible with Axolotl config file format
        """
        config = {
            # Base model configuration
            "base_model": self._get_base_model(),
            "model_type": self._get_model_type(),
            # Tokenizer
            "tokenizer_type": "AutoTokenizer",
            # Training configuration
            "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
            "batch_size": self.training_config.batch_size,
            "micro_batch_size": self.training_config.batch_size,
            "num_epochs": 3,
            "learning_rate": 2e-4,
            "optimizer": (
                "adamw_bnb_8bit" if self.training_config.dtype == DType.BF16 else "adamw_torch"
            ),
            "bf16": self.training_config.dtype == DType.BF16,
            "fp16": self.training_config.dtype == DType.FP16,
            "tf32": True,
            "gradient_checkpointing": self.training_config.activation_checkpointing > 0,
        }

        # Add special tokens configuration
        config.update(
            {
                "special_tokens": {
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "unk_token": "<unk>",
                    "pad_token": "<pad>",
                }
            }
        )

        # Add distributed training configuration
        if self.engine_config.type == EngineType.DEEPSPEED:
            config["deepspeed"] = self._get_deepspeed_config()
        elif self.engine_config.type == EngineType.FSDP:
            config["fsdp"] = self._get_fsdp_config()

        # Add multi-GPU configuration
        if self.node_config and self.node_config.num_nodes > 1:
            config["num_nodes"] = self.node_config.num_nodes
            config["gpus_per_node"] = self.node_config.gpus_per_node

        # Add additional training parameters
        config.update(
            {
                "val_set_size": 0.1,  # 10% validation
                "output_dir": "./output",
                "logging_steps": 10,
                "save_steps": 100,
                "eval_steps": 100,
                "save_total_limit": 2,
                "lr_scheduler": "cosine",
                "warmup_ratio": 0.03,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
            }
        )

        return config

    def _get_base_model(self) -> str:
        """Get base model path/name.

        Returns a placeholder or extracts from model_config.name
        """
        # Try to construct a reasonable model path
        name_map = {
            "llama2-7b": "meta-llama/Llama-2-7b-hf",
            "llama2-13b": "meta-llama/Llama-2-13b-hf",
            "llama2-70b": "meta-llama/Llama-2-70b-hf",
            "mistral-7b": "mistralai/Mistral-7B-v0.1",
            "mixtral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
            "gpt3-175b": "gpt3-175b-placeholder",  # Not on HF
        }

        return name_map.get(self.model_config.name.lower(), self.model_config.name)

    def _get_model_type(self) -> str:
        """Get model type for Axolotl."""
        model_type_map = {
            "llama": "LlamaForCausalLM",
            "mistral": "MistralForCausalLM",
            "mixtral": "MixtralForCausalLM",
            "qwen": "Qwen2ForCausalLM",
            "gemma": "GemmaForCausalLM",
            "bloom": "BloomForCausalLM",
            "gpt2": "GPT2LMHeadModel",
            "gptj": "GPTJForCausalLM",
            "bert": "BertForMaskedLM",
        }

        model_name_lower = self.model_config.name.lower()
        for key, value in model_type_map.items():
            if key in model_name_lower:
                return value

        return "LlamaForCausalLM"  # Default

    def _get_deepspeed_config(self) -> dict:
        """Get DeepSpeed configuration for Axolotl."""
        zero_opt: dict = {
            "stage": self.engine_config.zero_stage or 2,
        }

        config: dict = {
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
                "pin_memory": True,
            }

        if self.engine_config.offload_param != "none":
            config["zero_optimization"]["offload_param"] = {
                "device": "cpu" if self.engine_config.offload_param == "cpu" else "nvme",
                "pin_memory": True,
            }

        return config

    def _get_fsdp_config(self) -> dict:
        """Get FSDP configuration for Axolotl."""
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

        return config

    def _get_transformer_layer_cls(self) -> str:
        """Get transformer layer class name."""
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

        model_name_lower = self.model_config.name.lower()
        for key, value in model_layer_map.items():
            if key in model_name_lower:
                return value

        return "LlamaDecoderLayer"  # Default

    def export_yaml(self) -> str:
        """Generate YAML configuration string.

        Returns:
            YAML-formatted configuration string
        """
        import yaml  # type: ignore[import-untyped]

        config = self.export()
        return yaml.dump(config, default_flow_style=False, sort_keys=False)  # type: ignore[no-any-return]
