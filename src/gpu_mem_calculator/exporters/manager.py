"""Export manager for framework configurations.

Provides a unified interface for exporting configurations to various
training framework formats.
"""

from enum import Enum

from gpu_mem_calculator.core.models import (
    EngineConfig,
    ModelConfig,
    NodeConfig,
    ParallelismConfig,
    TrainingConfig,
)
from gpu_mem_calculator.exporters.accelerate import AccelerateExporter
from gpu_mem_calculator.exporters.axolotl import AxolotlExporter
from gpu_mem_calculator.exporters.lightning import LightningExporter


class ExportFormat(str, Enum):
    """Supported export formats."""

    ACCELERATE = "accelerate"
    LIGHTNING = "lightning"
    AXOLOTL = "axolotl"
    DEEPSPEED = "deepspeed"
    YAML = "yaml"
    JSON = "json"


class ExportManager:
    """Unified export manager for all framework configurations.

    This class provides a simple interface to export training
    configurations to various framework formats.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        parallelism_config: ParallelismConfig,
        engine_config: EngineConfig,
        node_config: NodeConfig | None = None,
    ) -> None:
        """Initialize the export manager.

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

        # Initialize exporters
        self.accelerate_exporter = AccelerateExporter(
            model_config=model_config,
            training_config=training_config,
            parallelism_config=parallelism_config,
            engine_config=engine_config,
            node_config=node_config,
        )

        self.lightning_exporter = LightningExporter(
            model_config=model_config,
            training_config=training_config,
            parallelism_config=parallelism_config,
            engine_config=engine_config,
            node_config=node_config,
        )

        self.axolotl_exporter = AxolotlExporter(
            model_config=model_config,
            training_config=training_config,
            parallelism_config=parallelism_config,
            engine_config=engine_config,
            node_config=node_config,
        )

    def export(self, format: ExportFormat | str) -> dict | str:
        """Export configuration to specified format.

        Args:
            format: Export format (accelerate, lightning, axolotl, deepspeed, yaml, json)

        Returns:
            Dictionary or string with exported configuration
        """
        format_str = format.value if isinstance(format, ExportFormat) else format

        match format_str:
            case ExportFormat.ACCELERATE:
                return self.accelerate_exporter.export()
            case ExportFormat.LIGHTNING:
                return self.lightning_exporter.export()
            case ExportFormat.AXOLOTL:
                return self.axolotl_exporter.export()
            case ExportFormat.DEEPSPEED:
                # DeepSpeed config is embedded in accelerate export
                config = self.accelerate_exporter.export()
                return config.get("deepspeed_config", {})  # type: ignore[no-any-return]
            case ExportFormat.YAML:
                return self._export_yaml()
            case ExportFormat.JSON:
                return self._export_json()
            case _:
                raise ValueError(f"Unknown export format: {format}")

    def export_to_file(
        self,
        format: ExportFormat | str,
        filepath: str,
    ) -> None:
        """Export configuration to a file.

        Args:
            format: Export format
            filepath: Path to output file
        """
        config = self.export(format)

        if isinstance(config, dict):
            if format == ExportFormat.YAML or (
                isinstance(format, str) and format.lower() == "yaml"
            ):
                import yaml  # type: ignore[import-untyped]

                with open(filepath, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            else:
                import json

                with open(filepath, "w") as f:
                    json.dump(config, f, indent=2)
        else:
            with open(filepath, "w") as f:
                f.write(config)

    def _export_yaml(self) -> str:
        """Export configuration to generic YAML format.

        Returns:
            YAML-formatted configuration string
        """
        import yaml  # type: ignore[import-untyped]

        config = {
            "model": {
                "name": self.model_config.name,
                "num_parameters": self.model_config.num_parameters,
                "num_layers": self.model_config.num_layers,
                "hidden_size": self.model_config.hidden_size,
                "num_attention_heads": self.model_config.num_attention_heads,
                "vocab_size": self.model_config.vocab_size,
                "max_seq_len": self.model_config.max_seq_len,
                "moe_enabled": self.model_config.moe_enabled,
            },
            "training": {
                "batch_size": self.training_config.batch_size,
                "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
                "optimizer": self.training_config.optimizer.value,
                "dtype": self.training_config.dtype.value,
                "activation_checkpointing": self.training_config.activation_checkpointing,
            },
            "parallelism": {
                "tensor_parallel_size": self.parallelism_config.tensor_parallel_size,
                "pipeline_parallel_size": self.parallelism_config.pipeline_parallel_size,
                "data_parallel_size": self.parallelism_config.data_parallel_size,
                "sequence_parallel": self.parallelism_config.sequence_parallel,
            },
            "engine": {
                "type": self.engine_config.type.value,
                "zero_stage": self.engine_config.zero_stage,
                "offload_optimizer": self.engine_config.offload_optimizer.value,
                "offload_param": self.engine_config.offload_param.value,
            },
        }

        # Add node configuration if multi-node
        if self.node_config and self.node_config.num_nodes > 1:
            config["multinode"] = {
                "num_nodes": self.node_config.num_nodes,
                "gpus_per_node": self.node_config.gpus_per_node,
                "interconnect_type": self.node_config.interconnect_type.value,
            }

        return yaml.dump(config, default_flow_style=False, sort_keys=False)  # type: ignore[no-any-return]

    def _export_json(self) -> str:
        """Export configuration to JSON format.

        Returns:
            JSON-formatted configuration string
        """
        import json

        config = {
            "model": self.model_config.model_dump(),
            "training": self.training_config.model_dump(),
            "parallelism": self.parallelism_config.model_dump(),
            "engine": self.engine_config.model_dump(),
        }

        # Add node configuration if multi-node
        if self.node_config:
            config["multinode"] = self.node_config.model_dump()

        return json.dumps(config, indent=2)

    def get_supported_formats(self) -> list[str]:
        """Get list of supported export formats.

        Returns:
            List of format names
        """
        return [f.value for f in ExportFormat]
