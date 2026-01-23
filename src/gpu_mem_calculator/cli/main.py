"""CLI interface for GPU Memory Calculator."""

import json
import sys
from pathlib import Path
from typing import Literal

import click


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """GPU Memory Calculator for LLM Training.

    Calculate GPU memory requirements for training Large Language Models
    with various training engines (PyTorch DDP, DeepSpeed, Megatron-LM, FSDP).
    """
    pass


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to JSON configuration file",
)
@click.option(
    "--preset",
    "-p",
    type=str,
    help="Name of a preset model configuration",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (default: stdout)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
    help="Output format (default: table)",
)
def calculate(
    config: str | None,
    preset: str | None,
    output: str | None,
    format: Literal["json", "yaml", "table"],
) -> None:
    """Calculate GPU memory requirements from config file or preset.

    Examples:
        gpu-mem-calc calculate --config configs/llama2_7b.json
        gpu-mem-calc calculate --preset llama2-7b
        gpu-mem-calc calculate -p mixtral-8x7b --format json
    """
    if not config and not preset:
        click.echo("Error: Either --config or --preset is required", err=True)
        sys.exit(1)

    if config and preset:
        click.echo("Error: Cannot use both --config and --preset", err=True)
        sys.exit(1)

    try:
        import tempfile

        from gpu_mem_calculator.core.calculator import GPUMemoryCalculator

        if preset:
            # Load preset configuration
            from gpu_mem_calculator.config.presets import get_preset_config

            preset_config = get_preset_config(preset)
            if preset_config is None:
                click.echo(
                    f"Error: Preset '{preset}' not found. "
                    "Use 'gpu-mem-calc presets' to list available presets.",
                    err=True,
                )
                sys.exit(1)

            # Write preset to temp file for from_config_file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(preset_config, f, indent=2)
                temp_path = f.name

            calculator = GPUMemoryCalculator.from_config_file(temp_path)
            Path(temp_path).unlink()  # Clean up temp file
        else:
            calculator = GPUMemoryCalculator.from_config_file(config)

        result = calculator.calculate()

        # Format output
        if format == "json":
            output_text = json.dumps(result.model_dump(mode="json"), indent=2)
        elif format == "yaml":
            try:
                import yaml

                output_text = yaml.dump(result.model_dump(mode="json"), default_flow_style=False)
            except ImportError:
                click.echo(
                    "Error: YAML format requires PyYAML. Install with: pip install pyyaml",
                    err=True,
                )
                sys.exit(1)
        else:  # table
            output_text = _format_result_as_table(result, calculator)

        # Write output
        if output:
            Path(output).write_text(output_text)
            click.echo(f"Results written to {output}")
        else:
            click.echo(output_text)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument(
    "params",
    type=float,
    required=True,
)
@click.option(
    "--gpus",
    "-g",
    type=int,
    default=1,
    help="Number of GPUs (default: 1)",
)
@click.option(
    "--gpu-mem",
    "-m",
    type=float,
    default=80.0,
    help="GPU memory in GB (default: 80.0)",
)
@click.option(
    "--engine",
    "-e",
    type=click.Choice(["pytorch", "deepspeed", "megatron", "fsdp"]),
    default="pytorch",
    help="Training engine (default: pytorch)",
)
@click.option(
    "--dtype",
    "-d",
    type=click.Choice(["fp32", "fp16", "bf16"]),
    default="bf16",
    help="Data type (default: bf16)",
)
def quick(
    params: float,
    gpus: int,
    gpu_mem: float,
    engine: str,
    dtype: str,
) -> None:
    """Quick calculation from model size (in billions of parameters).

    Example:
        gpu-mem-calc quick 7 --gpus 8 --engine deepspeed
    """
    try:
        from gpu_mem_calculator.core.calculator import GPUMemoryCalculator
        from gpu_mem_calculator.core.models import (
            DType,
            EngineConfig,
            EngineType,
            GPUConfig,
            ModelConfig,
            ParallelismConfig,
            TrainingConfig,
        )

        # Map engine string to EngineType
        engine_map = {
            "pytorch": EngineType.PYTORCH_DDP,
            "deepspeed": EngineType.DEEPSPEED,
            "megatron": EngineType.MEGATRON_LM,
            "fsdp": EngineType.FSDP,
        }

        # Map dtype string to DType
        dtype_map = {
            "fp32": DType.FP32,
            "fp16": DType.FP16,
            "bf16": DType.BF16,
        }

        # Create a minimal config for quick calculation
        # Estimate model architecture from parameter count
        # Rough approximation based on typical transformer models
        num_params = int(params * 1e9)

        # Estimate hidden size and layers from param count
        # These are rough approximations
        if params <= 1:
            hidden_size, num_layers = 768, 12
        elif params <= 7:
            hidden_size, num_layers = 4096, 32
        elif params <= 13:
            hidden_size, num_layers = 5120, 40
        elif params <= 30:
            hidden_size, num_layers = 6656, 60
        elif params <= 65:
            hidden_size, num_layers = 8192, 80
        else:
            hidden_size, num_layers = 12288, 96

        model_config = ModelConfig(
            name="quick-estimate",
            num_parameters=num_params,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=hidden_size // 128,
            vocab_size=32000,
            max_seq_len=2048,
        )

        training_config = TrainingConfig(
            batch_size=1,
            gradient_accumulation_steps=1,
            dtype=dtype_map[dtype],
        )

        parallelism_config = ParallelismConfig(data_parallel_size=gpus)

        engine_config = EngineConfig(
            type=engine_map[engine],
            zero_stage=2 if engine == "deepspeed" else None,
        )

        gpu_config = GPUConfig(num_gpus=gpus, gpu_memory_gb=gpu_mem)

        calculator = GPUMemoryCalculator(
            model_config=model_config,
            training_config=training_config,
            parallelism_config=parallelism_config,
            engine_config=engine_config,
            gpu_config=gpu_config,
        )

        result = calculator.calculate()

        # Display results
        click.echo(_format_result_as_table(result, calculator))

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True),
)
def validate(config_path: str) -> None:
    """Validate a configuration file.

    Example:
        gpu-mem-calc validate configs/my_config.json
    """
    try:
        from gpu_mem_calculator.config import ConfigParser

        ConfigParser.parse_full_config(config_path)
        click.echo(f"✓ Configuration file '{config_path}' is valid")

    except Exception as e:
        click.echo(f"✗ Validation failed: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--format",
    "-f",
    type=click.Choice(["list", "json", "table"]),
    default="list",
    help="Output format (default: list)",
)
def presets(format: str) -> None:
    """List available model preset configurations.

    Examples:
        gpu-mem-calc presets
        gpu-mem-calc presets --format table
        gpu-mem-calc presets -f json
    """
    try:
        from gpu_mem_calculator.config.presets import list_presets

        all_presets = list_presets()

        if not all_presets:
            click.echo("No presets found.")
            return

        if format == "json":
            click.echo(json.dumps(all_presets, indent=2))
        elif format == "table":
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(
                title="Available Model Presets",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Preset Name", style="cyan", width=25)
            table.add_column("Display Name", style="green", width=30)
            table.add_column("Description", style="yellow")

            for name, info in sorted(all_presets.items()):
                table.add_row(name, info["display_name"], info["description"])

            console.print(table)
        else:  # list format
            click.echo("Available model presets:\n")
            for name, info in sorted(all_presets.items()):
                click.echo(f"  {name:25} - {info['display_name']}")
                if info.get("description"):
                    click.echo(f"{'':27}{info['description']}")
                click.echo()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _format_result_as_table(result, calculator) -> str:
    """Format result as ASCII table."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Main results table
    table = Table(
        title="GPU Memory Calculation Results",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="green")

    # Memory results
    table.add_row("Memory per GPU", f"{result.total_memory_per_gpu_gb:.2f} GB")
    table.add_row("Total GPU Memory", f"{result.total_memory_all_gpus_gb:.2f} GB")
    table.add_row("CPU Memory", f"{result.cpu_memory_gb:.2f} GB")
    table.add_row("", "")  # Spacer

    # Breakdown
    table.add_row("Model Parameters", f"{result.breakdown.model_params_gb:.2f} GB")
    table.add_row("Gradients", f"{result.breakdown.gradients_gb:.2f} GB")
    table.add_row("Optimizer States", f"{result.breakdown.optimizer_states_gb:.2f} GB")
    table.add_row("Activations", f"{result.breakdown.activations_gb:.2f} GB")
    table.add_row("Overhead", f"{result.breakdown.overhead_gb:.2f} GB")
    table.add_row("", "")  # Spacer

    # Feasibility
    status = "✓ Fits" if result.fits_on_gpu else "✗ OOM"
    table.add_row("Status", status)
    table.add_row("Memory Utilization", f"{result.memory_utilization_percent:.1f}%")
    if result.recommended_batch_size:
        table.add_row("Recommended Batch Size", str(result.recommended_batch_size))

    # Capture table output
    from io import StringIO

    buffer = StringIO()
    console.file = buffer
    console.print(table)
    return buffer.getvalue()


if __name__ == "__main__":
    main()
