"""FastAPI backend for GPU Memory Calculator web application."""

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from gpu_mem_calculator.config.presets import load_presets, get_preset_config
from gpu_mem_calculator.core.calculator import GPUMemoryCalculator
from gpu_mem_calculator.core.models import (
    DType,
    EngineConfig,
    EngineType,
    GPUConfig,
    MemoryResult,
    ModelConfig,
    OffloadDevice,
    OptimizerType,
    ParallelismConfig,
    TrainingConfig,
)
from gpu_mem_calculator.utils.precision import gb_from_bytes

# Create FastAPI app
app = FastAPI(
    title="GPU Memory Calculator",
    description="Calculate GPU memory requirements for LLM training",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates and static files
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Mount static files
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Request/Response models
class CalculateRequest(BaseModel):
    """Request model for memory calculation."""

    model: dict[str, Any] = Field(description="Model configuration")
    training: dict[str, Any] = Field(description="Training configuration")
    parallelism: dict[str, Any] | None = Field(default=None, description="Parallelism configuration")
    engine: dict[str, Any] | None = Field(default=None, description="Engine configuration")
    hardware: dict[str, Any] | None = Field(default=None, description="Hardware configuration")


class PresetInfo(BaseModel):
    """Information about a preset model configuration."""

    name: str
    display_name: str
    description: str
    config: dict[str, Any]


# Load presets at startup using shared preset loader
# The shared loader reads from web/presets/models.json
def _load_presets_from_shared() -> dict[str, PresetInfo]:
    """Load presets using the shared preset loader."""
    all_presets = load_presets()
    return {
        name: PresetInfo(
            name=name,
            display_name=preset.get("display_name", name),
            description=preset.get("description", ""),
            config=preset.get("config", {}),
        )
        for name, preset in all_presets.items()
    }


PRESETS = _load_presets_from_shared()


# API Routes
@app.get("/")
async def index(request: Request) -> Any:
    """Serve the main web page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/engines")
async def list_engines() -> dict[str, str]:
    """List supported training engines."""
    return {
        "pytorch_ddp": "PyTorch DDP (Distributed Data Parallel)",
        "deepspeed": "DeepSpeed ZeRO",
        "megatron_lm": "Megatron-LM",
        "fsdp": "PyTorch FSDP (Fully Sharded Data Parallel)",
        "megatron_deepspeed": "Megatron-LM + DeepSpeed",
    }


@app.get("/api/optimizers")
async def list_optimizers() -> dict[str, str]:
    """List supported optimizers."""
    return {
        "adam": "Adam",
        "adamw": "AdamW",
        "adamw_8bit": "AdamW 8-bit",
        "sgd": "SGD",
    }


@app.get("/api/dtypes")
async def list_dtypes() -> dict[str, str]:
    """List supported data types."""
    return {
        "fp32": "FP32 (32-bit floating point)",
        "fp16": "FP16 (16-bit floating point)",
        "bf16": "BF16 (16-bit bfloat)",
        "int8": "INT8 (8-bit integer)",
        "int4": "INT4 (4-bit integer)",
    }


@app.get("/api/presets")
async def list_presets() -> dict[str, dict[str, str]]:
    """List all preset model configurations."""
    return {
        name: {
            "display_name": preset.display_name,
            "description": preset.description,
        }
        for name, preset in PRESETS.items()
    }


@app.get("/api/preset/{preset_name}")
async def get_preset(preset_name: str) -> dict[str, Any]:
    """Get a specific preset configuration."""
    if preset_name not in PRESETS:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")

    return PRESETS[preset_name].config


@app.post("/api/calculate")
async def calculate_memory(request: CalculateRequest) -> MemoryResult:
    """Calculate GPU memory requirements.

    Args:
        request: Calculation request with model, training, and hardware configs

    Returns:
        MemoryResult with complete memory breakdown
    """
    try:
        # Parse model configuration
        model_data = request.model.copy()
        # Parse num_parameters if it's a string (e.g., "7B", "7000M")
        if "num_parameters" in model_data and isinstance(model_data["num_parameters"], str):
            from gpu_mem_calculator.config.parser import ConfigParser
            model_data["num_parameters"] = ConfigParser._parse_num_params(model_data["num_parameters"])

        model_config = ModelConfig(**model_data)

        # Parse training configuration
        training_config = TrainingConfig(**request.training)

        # Parse optional configurations with defaults
        parallelism_config = (
            ParallelismConfig(**request.parallelism) if request.parallelism else ParallelismConfig()
        )

        engine_config = (
            EngineConfig(**request.engine) if request.engine else EngineConfig()
        )

        gpu_config = (
            GPUConfig(**request.hardware) if request.hardware else GPUConfig()
        )

        # Create calculator and compute
        calculator = GPUMemoryCalculator(
            model_config=model_config,
            training_config=training_config,
            parallelism_config=parallelism_config,
            engine_config=engine_config,
            gpu_config=gpu_config,
        )

        result = calculator.calculate()
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/validate")
async def validate_config(request: CalculateRequest) -> dict[str, Any]:
    """Validate a configuration without calculating memory.

    Args:
        request: Configuration to validate

    Returns:
        Validation result with valid flag and any errors
    """
    try:
        # Try to parse all configurations
        ModelConfig(**request.model)
        TrainingConfig(**request.training)

        if request.parallelism:
            ParallelismConfig(**request.parallelism)

        if request.engine:
            EngineConfig(**request.engine)

        if request.hardware:
            GPUConfig(**request.hardware)

        return {"valid": True, "errors": []}

    except Exception as e:
        return {"valid": False, "errors": [str(e)]}


def main() -> None:
    """Run the development server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
