"""Configuration file parser and utilities."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

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


class ConfigParseError(Exception):
    """Error parsing configuration file."""

    def __init__(self, message: str, errors: list[Any] | None = None):
        super().__init__(message)
        self.errors = errors or []


class ConfigParser:
    """Parse and validate configuration files."""

    @staticmethod
    def _convert_dtype(value: str) -> DType:
        """Convert string dtype to DType enum."""
        dtype_map = {
            "float32": DType.FP32,
            "fp32": DType.FP32,
            "float16": DType.FP16,
            "fp16": DType.FP16,
            "bfloat16": DType.BF16,
            "bf16": DType.BF16,
            "int8": DType.INT8,
            "int4": DType.INT4,
        }
        return dtype_map.get(value.lower(), DType.BF16)

    @staticmethod
    def _convert_optimizer(value: str) -> OptimizerType:
        """Convert string optimizer to OptimizerType enum."""
        opt_map = {
            "adam": OptimizerType.ADAM,
            "adamw": OptimizerType.ADAMW,
            "sgd": OptimizerType.SGD,
            "adamw_8bit": OptimizerType.ADAMW_8BIT,
            "adamw-8bit": OptimizerType.ADAMW_8BIT,
        }
        return opt_map.get(value.lower(), OptimizerType.ADAMW)

    @staticmethod
    def _convert_engine(value: str) -> EngineType:
        """Convert string engine to EngineType enum."""
        engine_map = {
            "pytorch": EngineType.PYTORCH_DDP,
            "pytorch_ddp": EngineType.PYTORCH_DDP,
            "ddp": EngineType.PYTORCH_DDP,
            "deepspeed": EngineType.DEEPSPEED,
            "megatron": EngineType.MEGATRON_LM,
            "megatron_lm": EngineType.MEGATRON_LM,
            "megatron-lm": EngineType.MEGATRON_LM,
            "fsdp": EngineType.FSDP,
            "megatron_deepspeed": EngineType.MEGATRON_DEEPSPEED,
        }
        return engine_map.get(value.lower(), EngineType.PYTORCH_DDP)

    @staticmethod
    def _convert_offload(value: str) -> OffloadDevice:
        """Convert string offload to OffloadDevice enum."""
        offload_map = {
            "none": OffloadDevice.NONE,
            "cpu": OffloadDevice.CPU,
            "nvme": OffloadDevice.NVME,
        }
        return offload_map.get(value.lower(), OffloadDevice.NONE)

    @staticmethod
    def _parse_num_params(value: str | int | float) -> int:
        """Parse number of parameters from various formats.

        Supports:
        - Raw integer: 7000000000
        - Billions: "7B", "7b", "7e9"
        - Millions: "7000M", "7000m", "7000e6"
        """
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)

        if isinstance(value, str):
            value = value.strip().upper()

            # Handle billions suffix
            if value.endswith("B"):
                return int(float(value[:-1]) * 1_000_000_000)

            # Handle millions suffix
            if value.endswith("M"):
                return int(float(value[:-1]) * 1_000_000)

            # Handle scientific notation
            if "E" in value:
                return int(float(value))

            # Try direct conversion
            return int(value)

        raise ValueError(f"Cannot parse parameter count: {value}")

    @classmethod
    def parse_model_config(cls, data: dict[str, Any]) -> ModelConfig:
        """Parse model configuration from dict.

        Args:
            data: Dictionary with model configuration

        Returns:
            ModelConfig object

        Raises:
            ConfigParseError: If validation fails
        """
        try:
            # Convert parameter count if it's a string
            if "num_parameters" in data and isinstance(data["num_parameters"], str):
                data["num_parameters"] = cls._parse_num_params(data["num_parameters"])

            if "largest_layer_params" in data and isinstance(data["largest_layer_params"], str):
                data["largest_layer_params"] = cls._parse_num_params(data["largest_layer_params"])

            return ModelConfig(**data)
        except ValidationError as e:
            raise ConfigParseError("Invalid model configuration", e.errors()) from e

    @classmethod
    def parse_training_config(cls, data: dict[str, Any]) -> TrainingConfig:
        """Parse training configuration from dict.

        Args:
            data: Dictionary with training configuration

        Returns:
            TrainingConfig object

        Raises:
            ConfigParseError: If validation fails
        """
        try:
            # Convert dtype
            if "dtype" in data and isinstance(data["dtype"], str):
                data["dtype"] = cls._convert_dtype(data["dtype"])

            # Convert optimizer
            if "optimizer" in data and isinstance(data["optimizer"], str):
                data["optimizer"] = cls._convert_optimizer(data["optimizer"])

            return TrainingConfig(**data)
        except ValidationError as e:
            raise ConfigParseError("Invalid training configuration", e.errors()) from e

    @classmethod
    def parse_parallelism_config(cls, data: dict[str, Any]) -> ParallelismConfig:
        """Parse parallelism configuration from dict.

        Args:
            data: Dictionary with parallelism configuration

        Returns:
            ParallelismConfig object

        Raises:
            ConfigParseError: If validation fails
        """
        try:
            return ParallelismConfig(**data)
        except ValidationError as e:
            raise ConfigParseError("Invalid parallelism configuration", e.errors()) from e

    @classmethod
    def parse_engine_config(cls, data: dict[str, Any]) -> EngineConfig:
        """Parse engine configuration from dict.

        Args:
            data: Dictionary with engine configuration

        Returns:
            EngineConfig object

        Raises:
            ConfigParseError: If validation fails
        """
        try:
            # Convert engine type
            if "type" in data and isinstance(data["type"], str):
                data["type"] = cls._convert_engine(data["type"])

            # Convert offload options
            if "offload_optimizer" in data and isinstance(data["offload_optimizer"], str):
                data["offload_optimizer"] = cls._convert_offload(data["offload_optimizer"])

            if "offload_param" in data and isinstance(data["offload_param"], str):
                data["offload_param"] = cls._convert_offload(data["offload_param"])

            return EngineConfig(**data)
        except ValidationError as e:
            raise ConfigParseError("Invalid engine configuration", e.errors()) from e

    @classmethod
    def parse_gpu_config(cls, data: dict[str, Any]) -> GPUConfig:
        """Parse GPU configuration from dict.

        Args:
            data: Dictionary with GPU configuration

        Returns:
            GPUConfig object

        Raises:
            ConfigParseError: If validation fails
        """
        try:
            return GPUConfig(**data)
        except ValidationError as e:
            raise ConfigParseError("Invalid GPU configuration", e.errors()) from e

    @classmethod
    def parse_file(cls, config_path: str | Path) -> dict[str, Any]:
        """Parse configuration from JSON file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary with parsed configuration

        Raises:
            ConfigParseError: If file cannot be read or parsed
        """
        path = Path(config_path)
        if not path.exists():
            raise ConfigParseError(f"Configuration file not found: {config_path}")

        try:
            with path.open("r") as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            raise ConfigParseError(f"Invalid JSON in configuration file: {e}") from e
        except Exception as e:
            raise ConfigParseError(f"Error reading configuration file: {e}") from e

    @classmethod
    def parse_full_config(
        cls,
        config_path: str | Path,
    ) -> tuple[ModelConfig, TrainingConfig, ParallelismConfig, EngineConfig, GPUConfig]:
        """Parse complete configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Tuple of (ModelConfig, TrainingConfig, ParallelismConfig, EngineConfig, GPUConfig)

        Raises:
            ConfigParseError: If validation fails
        """
        data = cls.parse_file(config_path)

        try:
            model_config = cls.parse_model_config(data.get("model", {}))
            training_config = cls.parse_training_config(data.get("training", {}))
            parallelism_config = cls.parse_parallelism_config(data.get("parallelism", {}))
            engine_config = cls.parse_engine_config(data.get("engine", {}))
            gpu_config = cls.parse_gpu_config(data.get("hardware", {}))

            return (
                model_config,
                training_config,
                parallelism_config,
                engine_config,
                gpu_config,
            )
        except ConfigParseError:
            raise
        except Exception as e:
            raise ConfigParseError(f"Unexpected error parsing configuration: {e}") from e


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary with configuration data
    """
    return ConfigParser.parse_file(config_path)


def save_config(data: dict[str, Any], output_path: str | Path) -> None:
    """Save configuration to JSON file.

    Args:
        data: Configuration dictionary to save
        output_path: Path to save configuration file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(data, f, indent=2)
