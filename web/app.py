"""FastAPI backend for GPU Memory Calculator web application."""

import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator, model_validator
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Request model for memory calculation with comprehensive validation."""

    model: dict[str, Any] = Field(description="Model configuration")
    training: dict[str, Any] = Field(description="Training configuration")
    parallelism: dict[str, Any] | None = Field(default=None, description="Parallelism configuration")
    engine: dict[str, Any] | None = Field(default=None, description="Engine configuration")
    hardware: dict[str, Any] | None = Field(default=None, description="Hardware configuration")

    @field_validator('model')
    @classmethod
    def validate_moe_settings(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate MoE-specific constraints."""
        if v.get('moe_enabled'):
            num_experts = v.get('num_experts', 1)
            top_k = v.get('top_k', 1)

            if top_k > num_experts:
                raise ValueError(
                    f"MoE top_k ({top_k}) cannot exceed num_experts ({num_experts})"
                )

            if num_experts < 1 or num_experts > 256:
                raise ValueError(
                    f"num_experts must be between 1 and 256, got {num_experts}"
                )

            if top_k < 1 or top_k > 8:
                raise ValueError(
                    f"top_k must be between 1 and 8, got {top_k}"
                )

        return v

    @model_validator(mode='after')
    def validate_parallelism_consistency(self) -> 'CalculateRequest':
        """Validate parallelism settings consistency."""
        if self.parallelism and self.hardware:
            tensor_pp = self.parallelism.get('tensor_parallel_size', 1)
            pipeline_pp = self.parallelism.get('pipeline_parallel_size', 1)
            data_pp = self.parallelism.get('data_parallel_size', 1)
            num_gpus = self.hardware.get('num_gpus', 1)

            effective_gpus = tensor_pp * pipeline_pp * data_pp

            if effective_gpus != num_gpus:
                raise ValueError(
                    f"Parallelism mismatch: tensor_pp ({tensor_pp}) × "
                    f"pipeline_pp ({pipeline_pp}) × data_pp ({data_pp}) = "
                    f"{effective_gpus} GPUs, but num_gpus is set to {num_gpus}. "
                    f"These must match."
                )

        # Validate sequence parallel requires tensor parallel > 1
        if self.parallelism and self.parallelism.get('sequence_parallel'):
            tensor_pp = self.parallelism.get('tensor_parallel_size', 1)
            if tensor_pp <= 1:
                raise ValueError(
                    f"Sequence parallelism requires tensor_parallel_size > 1, "
                    f"got {tensor_pp}"
                )

        return self

    @model_validator(mode='after')
    def validate_engine_settings(self) -> 'CalculateRequest':
        """Validate engine-specific settings."""
        if not self.engine:
            return self

        engine_type = self.engine.get('type')
        zero_stage = self.engine.get('zero_stage', 0)

        # ZeRO stages only valid for DeepSpeed engines
        if engine_type not in ['deepspeed', 'megatron_deepspeed'] and zero_stage > 0:
            raise ValueError(
                f"ZeRO stages are only supported for DeepSpeed engines, "
                f"got engine_type='{engine_type}' with zero_stage={zero_stage}"
            )

        # Validate ZeRO stage range
        if zero_stage < 0 or zero_stage > 3:
            raise ValueError(
                f"zero_stage must be between 0 and 3, got {zero_stage}"
            )

        return self


class PresetInfo(BaseModel):
    """Information about a preset model configuration."""

    name: str
    display_name: str
    description: str
    config: dict[str, Any]


# Simple in-memory cache for calculation results
# In production, use Redis or similar
_calculation_cache: dict[str, tuple[MemoryResult, float]] = {}  # key -> (result, timestamp)
_CACHE_TTL = 3600  # 1 hour
_MAX_CACHE_SIZE = 1000


def _cache_key_from_request(request: CalculateRequest) -> str:
    """Generate cache key from request."""
    request_dict = request.model_dump()
    # Sort keys for consistent hashing
    request_str = json.dumps(request_dict, sort_keys=True)
    return hashlib.md5(request_str.encode()).hexdigest()


def _get_cached_result(key: str) -> MemoryResult | None:
    """Get cached result if available and not expired."""
    if key in _calculation_cache:
        result, timestamp = _calculation_cache[key]
        import time
        if time.time() - timestamp < _CACHE_TTL:
            return result
        else:
            # Expired, remove from cache
            del _calculation_cache[key]
    return None


def _cache_result(key: str, result: MemoryResult) -> None:
    """Cache calculation result."""
    import time
    # Simple cache eviction if too large
    if len(_calculation_cache) >= _MAX_CACHE_SIZE:
        # Remove oldest entry (first key)
        oldest_key = next(iter(_calculation_cache))
        del _calculation_cache[oldest_key]

    _calculation_cache[key] = (result, time.time())


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
    # Check cache first
    cache_key = _cache_key_from_request(request)
    cached_result = _get_cached_result(cache_key)
    if cached_result is not None:
        logger.info(f"Cache hit for key: {cache_key[:8]}...")
        return cached_result

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

        # Cache the result
        _cache_result(cache_key, result)

        logger.info(
            f"Calculation successful: {model_config.name}, "
            f"{result.total_memory_per_gpu_gb:.2f} GB per GPU"
        )

        return result

    except ValueError as e:
        # User input validation error
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Validation error",
                "message": str(e),
                "type": "validation_error"
            }
        ) from e
    except Exception as e:
        # Unexpected system error
        logger.error(f"Calculation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "An unexpected error occurred during calculation"
            }
        ) from e


@app.post("/api/export/deepspeed")
async def export_deepspeed_config(request: CalculateRequest) -> dict[str, Any]:
    """Export DeepSpeed configuration file.

    Args:
        request: Calculation request with model, training, and hardware configs

    Returns:
        DeepSpeed config JSON and memory result
    """
    try:
        # First calculate memory
        calc_result = await calculate_memory(request)

        # Generate DeepSpeed config
        parallelism = request.parallelism or {}
        training = request.training
        engine = request.engine or {}

        train_batch_size = (
            training.get('batch_size', 1) *
            training.get('gradient_accumulation_steps', 1) *
            parallelism.get('data_parallel_size', 1)
        )

        zero_stage = engine.get('zero_stage', 0)
        offload_optimizer = engine.get('offload_optimizer', 'none')
        offload_param = engine.get('offload_param', 'none')

        deepspeed_config = {
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": training.get('batch_size', 1),
            "gradient_accumulation_steps": training.get('gradient_accumulation_steps', 1),
            "optimizer": {
                "type": training.get('optimizer', 'AdamW'),
                "params": {
                    "lr": 0.0001,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 0.0001,
                    "warmup_num_steps": 2000
                }
            },
            "fp16": {
                "enabled": training.get('dtype') in ['fp16', 'int4', 'int8']
            },
            "bf16": {
                "enabled": training.get('dtype') == 'bf16'
            },
            "zero_optimization": {
                "stage": zero_stage
            },
            "gradient_clipping": training.get('gradient_clipping', 1.0),
            "steps_per_print": 100,
        }

        # Add offload config if ZeRO stage >= 1
        if zero_stage >= 1:
            deepspeed_config["zero_optimization"]["offload_optimizer"] = {
                "device": offload_optimizer
            }
            deepspeed_config["zero_optimization"]["offload_param"] = {
                "device": offload_param
            }

        return {
            "config": deepspeed_config,
            "memory_result": calc_result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DeepSpeed export error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate DeepSpeed config: {str(e)}"
        ) from e


@app.post("/api/optimize/batch-size")
async def optimize_batch_size(request: CalculateRequest) -> dict[str, Any]:
    """Find maximum batch size that fits in GPU memory.

    Uses binary search to find the maximum batch size that doesn't OOM.

    Args:
        request: Calculation request with model, training, and hardware configs

    Returns:
        Maximum batch size that fits and corresponding memory result
    """
    try:
        # Create a mutable copy for testing
        from copy import deepcopy

        min_batch = 1
        max_batch = 512  # Reasonable upper bound
        best_batch = 1

        while min_batch <= max_batch:
            mid = (min_batch + max_batch) // 2

            # Create modified request with test batch size
            test_request = deepcopy(request)
            test_request.training['batch_size'] = mid

            try:
                # Validate and calculate
                CalculateRequest.model_validate(test_request)
                result = await calculate_memory(test_request)

                if result.fits_on_gpu:
                    best_batch = mid
                    min_batch = mid + 1
                else:
                    max_batch = mid - 1
            except (ValueError, HTTPException):
                # Invalid config or doesn't fit
                max_batch = mid - 1

        # Get final result for best batch size
        final_request = deepcopy(request)
        final_request.training['batch_size'] = best_batch
        final_result = await calculate_memory(final_request)

        return {
            "max_batch_size": best_batch,
            "memory_result": final_result
        }

    except Exception as e:
        logger.error(f"Batch size optimization error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to optimize batch size: {str(e)}"
        ) from e


@app.post("/api/validate")
async def validate_config(request: CalculateRequest) -> dict[str, Any]:
    """Validate a configuration without calculating memory.

    Args:
        request: Configuration to validate

    Returns:
        Validation result with valid flag and any errors
    """
    try:
        # Pydantic validation happens automatically when creating CalculateRequest
        # If we get here, the request is valid
        return {"valid": True, "errors": []}

    except ValueError as e:
        # Validation error
        return {"valid": False, "errors": [str(e)]}
    except Exception as e:
        # Unexpected error
        logger.error(f"Validation error: {str(e)}", exc_info=True)
        return {"valid": False, "errors": [str(e)]}


def main() -> None:
    """Run the development server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
