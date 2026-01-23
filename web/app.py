"""FastAPI backend for GPU Memory Calculator web application."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.requests import Request

from gpu_mem_calculator.config.presets import load_presets
from gpu_mem_calculator.core.calculator import GPUMemoryCalculator
from gpu_mem_calculator.core.models import (
    EngineConfig,
    GPUConfig,
    MemoryResult,
    ModelConfig,
    ParallelismConfig,
    TrainingConfig,
)

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
    parallelism: dict[str, Any] | None = Field(
        default=None,
        description="Parallelism configuration",
    )
    engine: dict[str, Any] | None = Field(default=None, description="Engine configuration")
    hardware: dict[str, Any] | None = Field(default=None, description="Hardware configuration")

    @field_validator("model")
    @classmethod
    def validate_moe_settings(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate MoE-specific constraints."""
        if v.get("moe_enabled"):
            num_experts = v.get("num_experts", 1)
            top_k = v.get("top_k", 1)

            if top_k > num_experts:
                raise ValueError(f"MoE top_k ({top_k}) cannot exceed num_experts ({num_experts})")

            if num_experts < 1 or num_experts > 256:
                raise ValueError(f"num_experts must be between 1 and 256, got {num_experts}")

            if top_k < 1 or top_k > 8:
                raise ValueError(f"top_k must be between 1 and 8, got {top_k}")

        return v

    @model_validator(mode="after")
    def validate_parallelism_consistency(self) -> "CalculateRequest":
        """Validate parallelism settings consistency."""
        if self.parallelism and self.hardware:
            tensor_pp = self.parallelism.get("tensor_parallel_size", 1)
            pipeline_pp = self.parallelism.get("pipeline_parallel_size", 1)
            data_pp = self.parallelism.get("data_parallel_size", 1)
            num_gpus = self.hardware.get("num_gpus", 1)

            effective_gpus = tensor_pp * pipeline_pp * data_pp

            if effective_gpus != num_gpus:
                raise ValueError(
                    f"Parallelism mismatch: tensor_pp ({tensor_pp}) × "
                    f"pipeline_pp ({pipeline_pp}) × data_pp ({data_pp}) = "
                    f"{effective_gpus} GPUs, but num_gpus is set to {num_gpus}. "
                    f"These must match."
                )

        # Validate sequence parallel requires tensor parallel > 1
        if self.parallelism and self.parallelism.get("sequence_parallel"):
            tensor_pp = self.parallelism.get("tensor_parallel_size", 1)
            if tensor_pp <= 1:
                raise ValueError(
                    f"Sequence parallelism requires tensor_parallel_size > 1, " f"got {tensor_pp}"
                )

        return self

    @model_validator(mode="after")
    def validate_engine_settings(self) -> "CalculateRequest":
        """Validate engine-specific settings."""
        if not self.engine:
            return self

        engine_type = self.engine.get("type")
        zero_stage = self.engine.get("zero_stage", 0)

        # ZeRO stages only valid for DeepSpeed engines
        if engine_type not in ["deepspeed", "megatron_deepspeed"] and zero_stage > 0:
            raise ValueError(
                f"ZeRO stages are only supported for DeepSpeed engines, "
                f"got engine_type='{engine_type}' with zero_stage={zero_stage}"
            )

        # Validate ZeRO stage range
        if zero_stage < 0 or zero_stage > 3:
            raise ValueError(f"zero_stage must be between 0 and 3, got {zero_stage}")

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
        if "num_parameters" in model_data and isinstance(
            model_data["num_parameters"],
            str,
        ):
            from gpu_mem_calculator.config.parser import ConfigParser

            model_data["num_parameters"] = ConfigParser._parse_num_params(
                model_data["num_parameters"],
            )

        model_config = ModelConfig(**model_data)

        # Parse training configuration
        training_config = TrainingConfig(**request.training)

        # Parse optional configurations with defaults
        parallelism_config = (
            ParallelismConfig(**request.parallelism) if request.parallelism else ParallelismConfig()
        )

        engine_config = EngineConfig(**request.engine) if request.engine else EngineConfig()

        gpu_config = GPUConfig(**request.hardware) if request.hardware else GPUConfig()

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
            detail={"error": "Validation error", "message": str(e), "type": "validation_error"},
        ) from e
    except Exception as e:
        # Unexpected system error
        logger.error(f"Calculation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "An unexpected error occurred during calculation",
            },
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
            training.get("batch_size", 1)
            * training.get("gradient_accumulation_steps", 1)
            * parallelism.get("data_parallel_size", 1)
        )

        zero_stage = engine.get("zero_stage", 0)
        offload_optimizer = engine.get("offload_optimizer", "none")
        offload_param = engine.get("offload_param", "none")

        deepspeed_config = {
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": training.get("batch_size", 1),
            "gradient_accumulation_steps": training.get("gradient_accumulation_steps", 1),
            "optimizer": {
                "type": training.get("optimizer", "AdamW"),
                "params": {"lr": 0.0001, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {"warmup_min_lr": 0, "warmup_max_lr": 0.0001, "warmup_num_steps": 2000},
            },
            "fp16": {"enabled": training.get("dtype") in ["fp16", "int4", "int8"]},
            "bf16": {"enabled": training.get("dtype") == "bf16"},
            "zero_optimization": {"stage": zero_stage},
            "gradient_clipping": training.get("gradient_clipping", 1.0),
            "steps_per_print": 100,
        }

        # Add offload config if ZeRO stage >= 1
        if zero_stage >= 1:
            deepspeed_config["zero_optimization"]["offload_optimizer"] = {
                "device": offload_optimizer
            }
            deepspeed_config["zero_optimization"]["offload_param"] = {"device": offload_param}

        return {"config": deepspeed_config, "memory_result": calc_result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DeepSpeed export error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate DeepSpeed config: {str(e)}"
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
            test_request.training["batch_size"] = mid

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
        final_request.training["batch_size"] = best_batch
        final_result = await calculate_memory(final_request)

        return {"max_batch_size": best_batch, "memory_result": final_result}

    except Exception as e:
        logger.error(f"Batch size optimization error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to optimize batch size: {str(e)}"
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


@app.post("/api/explain-formula")
async def explain_formula(request: CalculateRequest) -> dict[str, Any]:
    """Explain the memory formula used for calculation.

    Returns detailed information about which formula is being used,
    with the user's values plugged in, and links to documentation.

    Args:
        request: Calculation request with model, training, and hardware configs

    Returns:
        Formula explanation with formula type, breakdown, and references
    """
    try:
        # Get configuration details
        engine_type = request.engine.get("type", "pytorch_ddp") if request.engine else "pytorch_ddp"
        num_params = request.model.get("num_parameters", 0)

        # Parse num_parameters if it's a string (e.g., "7B", "7000M")
        if isinstance(num_params, str):
            from gpu_mem_calculator.config.parser import ConfigParser

            num_params = ConfigParser._parse_num_params(num_params)

        optimizer = request.training.get("optimizer", "adamw")
        num_gpus = request.hardware.get("num_gpus", 1) if request.hardware else 1
        batch_size = request.training.get("batch_size", 1)

        # Calculate memory to get the breakdown
        result = await calculate_memory(request)

        # Determine formula description based on engine type
        formula_info = {
            "engine_type": engine_type,
            "engine_name": _get_engine_name(engine_type),
            "formula_components": [],
            "total_memory_gb": round(result.total_memory_per_gpu_gb, 2),
            "breakdown": {
                "model_params_gb": round(result.breakdown.model_params_gb, 2),
                "gradients_gb": round(result.breakdown.gradients_gb, 2),
                "optimizer_states_gb": round(result.breakdown.optimizer_states_gb, 2),
                "activations_gb": round(result.breakdown.activations_gb, 2),
                "overhead_gb": round(result.breakdown.overhead_gb, 2),
            },
            "references": _get_formula_references(engine_type),
        }

        # Add engine-specific formula details
        if engine_type == "pytorch_ddp":
            formula_info["formula_description"] = (
                "PyTorch DDP stores complete copies of model parameters, gradients, "
                "and optimizer states on each GPU."
            )
            formula_info["formula_components"] = [
                {
                    "name": "Model Parameters",
                    "formula": f"{num_params:,} × 2 bytes (FP16/BF16)",
                    "result": f"{result.breakdown.model_params_gb:.2f} GB",
                    "description": "Full model stored on each GPU",
                },
                {
                    "name": "Gradients",
                    "formula": f"{num_params:,} × 2 bytes (FP16)",
                    "result": f"{result.breakdown.gradients_gb:.2f} GB",
                    "description": "Full gradients during backward pass",
                },
                {
                    "name": "Optimizer States",
                    "formula": _get_optimizer_formula(optimizer, num_params)["formula"],
                    "result": f"{result.breakdown.optimizer_states_gb:.2f} GB",
                    "description": _get_optimizer_formula(optimizer, num_params)["description"],
                },
            ]

        elif engine_type in ["deepspeed", "megatron_deepspeed"]:
            zero_stage = request.engine.get("zero_stage", 0) if request.engine else 0
            offload_optimizer = (
                request.engine.get("offload_optimizer", "none") if request.engine else "none"
            )
            offload_param = (
                request.engine.get("offload_param", "none") if request.engine else "none"
            )

            if zero_stage == 0:
                stage_name = "ZeRO-0 (Baseline)"
                formula_info["formula_description"] = (
                    f"{stage_name}: No memory optimization. Same as PyTorch DDP."
                )
            elif zero_stage == 1:
                stage_name = "ZeRO-1"
                formula_info["formula_description"] = (
                    f"{stage_name}: Shards optimizer states across {num_gpus} GPUs. "
                    f"Reduces optimizer memory by {num_gpus}x."
                )
            elif zero_stage == 2:
                stage_name = "ZeRO-2"
                formula_info["formula_description"] = (
                    f"{stage_name}: Shards optimizer states AND gradients across {num_gpus} GPUs. "
                    f"Reduces memory by {num_gpus}x for both components."
                )
            elif zero_stage == 3:
                stage_name = "ZeRO-3"
                formula_info["formula_description"] = (
                    f"{stage_name}: Shards parameters, gradients, AND optimizer states. "
                    f"Only largest layer stored intact. Linear memory reduction with GPU count."
                )

            formula_info["zero_stage"] = zero_stage
            formula_info["offload_optimizer"] = offload_optimizer
            formula_info["offload_param"] = offload_param

            # Add ZeRO-specific components
            if zero_stage == 3:
                # Estimate largest layer (approx 10% of params for typical models)
                largest_params = num_params // 10
                formula_info["formula_components"] = [
                    {
                        "name": "Largest Layer",
                        "formula": f"{largest_params:,} × 4 bytes (FP16 params + grads)",
                        "result": f"{result.breakdown.model_params_gb:.2f} GB",
                        "description": "Gathered during compute, largest layer kept intact",
                    },
                    {
                        "name": "Sharded Parameters",
                        "formula": f"({num_params:,} × 2 bytes) / {num_gpus} GPUs",
                        "result": "Included in model params",
                        "description": "Remaining parameters sharded across GPUs",
                    },
                    {
                        "name": "Sharded Optimizer States",
                        "formula": (
                            (
                                f"({_get_optimizer_formula(optimizer, num_params)['formula']}) "
                                f"/ {num_gpus} GPUs"
                            )
                            if offload_optimizer == "none"
                            else f"Offloaded to {offload_optimizer}"
                        ),
                        "result": f"{result.breakdown.optimizer_states_gb:.2f} GB",
                        "description": (
                            _get_optimizer_formula(optimizer, num_params)["description"]
                            + " (sharded or offloaded)"
                        ),
                    },
                ]
            else:
                # ZeRO-1 or ZeRO-2
                formula_info["formula_components"] = [
                    {
                        "name": "Model Parameters",
                        "formula": f"{num_params:,} × 2 bytes (FP16)",
                        "result": f"{result.breakdown.model_params_gb:.2f} GB",
                        "description": "Full model on each GPU",
                    },
                    {
                        "name": "Gradients",
                        "formula": (
                            f"{num_params:,} × 2 bytes"
                            if zero_stage < 2
                            else f"({num_params:,} × 2 bytes) / {num_gpus} GPUs"
                        ),
                        "result": f"{result.breakdown.gradients_gb:.2f} GB",
                        "description": (
                            "Sharded across GPUs" if zero_stage >= 2 else "Full gradients"
                        ),
                    },
                    {
                        "name": "Optimizer States",
                        "formula": (
                            (
                                f"({_get_optimizer_formula(optimizer, num_params)['formula']}) "
                                f"/ {num_gpus} GPUs"
                            )
                            if offload_optimizer == "none"
                            else f"Offloaded to {offload_optimizer}"
                        ),
                        "result": f"{result.breakdown.optimizer_states_gb:.2f} GB",
                        "description": (
                            _get_optimizer_formula(optimizer, num_params)["description"]
                            + " (sharded or offloaded)"
                        ),
                    },
                ]

        elif engine_type == "fsdp":
            sharding_strategy = (
                request.engine.get("sharding_strategy", "full_shard")
                if request.engine
                else "full_shard"
            )

            if sharding_strategy == "no_shard":
                strategy_name = "No Sharding (like DDP)"
            elif sharding_strategy == "shard_grad_op":
                strategy_name = "Shard Gradients + Optimizer (like ZeRO-2)"
            else:
                strategy_name = "Full Shard (like ZeRO-3)"

            formula_info["sharding_strategy"] = sharding_strategy
            formula_info["strategy_name"] = strategy_name
            formula_info["formula_description"] = f"FSDP with {strategy_name.lower()} strategy."

        elif engine_type == "megatron_lm":
            formula_info["formula_description"] = (
                "Megatron-LM uses tensor and/or pipeline parallelism to "
                "split the model across GPUs, reducing memory per GPU."
            )

            # Add parallelism info
            if request.parallelism:
                tp_size = request.parallelism.get("tensor_parallel_size", 1)
                pp_size = request.parallelism.get("pipeline_parallel_size", 1)
                formula_info["parallelism"] = {
                    "tensor_parallel_size": tp_size,
                    "pipeline_parallel_size": pp_size,
                }

        # Add activation memory explanation
        formula_info["formula_components"].append(
            {
                "name": "Activations",
                "formula": (
                    f"batch_size({batch_size}) × seq_len × hidden_size × "
                    f"layers × ~16 bytes/token/layer"
                ),
                "result": f"{result.breakdown.activations_gb:.2f} GB",
                "description": "Memory from intermediate activations during forward/backward pass",
            }
        )

        return formula_info

    except Exception as e:
        logger.error(f"Formula explanation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate formula explanation: {str(e)}"
        ) from e


def _get_engine_name(engine_type: str) -> str:
    """Get human-readable engine name."""
    names = {
        "pytorch_ddp": "PyTorch DDP (Distributed Data Parallel)",
        "deepspeed": "DeepSpeed ZeRO",
        "megatron_lm": "Megatron-LM",
        "fsdp": "PyTorch FSDP (Fully Sharded Data Parallel)",
        "megatron_deepspeed": "Megatron-LM + DeepSpeed",
    }
    return names.get(engine_type, engine_type)


def _get_optimizer_formula(optimizer: str, num_params: int) -> dict[str, str]:
    """Get optimizer memory formula based on optimizer type.

    Args:
        optimizer: Optimizer type (adam, adamw, sgd, adamw_8bit)
        num_params: Number of model parameters

    Returns:
        Dictionary with 'formula' and 'description' keys
    """
    num_params_formatted = f"{num_params:,}"

    if optimizer in ["adam", "adamw"]:
        return {
            "formula": f"{num_params_formatted} × 12 bytes (Adam/AdamW FP32)",
            "description": "4 bytes FP32 params + 4 bytes momentum + 4 bytes variance",
        }
    elif optimizer == "adamw_8bit":
        return {
            "formula": f"{num_params_formatted} × 2 bytes (AdamW 8-bit)",
            "description": "8-bit quantized optimizer states (2 bytes per parameter)",
        }
    elif optimizer == "sgd":
        return {
            "formula": f"{num_params_formatted} × 4 bytes (SGD)",
            "description": "4 bytes FP32 params (no momentum for SGD)",
        }
    else:
        # Default to AdamW
        return {
            "formula": f"{num_params_formatted} × 12 bytes (Adam/AdamW FP32)",
            "description": "4 bytes FP32 params + 4 bytes momentum + 4 bytes variance",
        }


def _get_formula_references(engine_type: str) -> list[dict[str, str]]:
    """Get authoritative references for the formula."""
    references = [
        {
            "title": "EleutherAI Transformer Math 101",
            "url": "https://blog.eleuther.ai/transformer-math/",
            "description": "Comprehensive transformer memory breakdown with formulas",
        },
        {
            "title": "Microsoft Research ZeRO Blog",
            "url": "https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/",
            "description": "ZeRO optimization techniques and memory formulas",
        },
    ]

    if engine_type in ["deepspeed", "megatron_deepspeed"]:
        references.append(
            {
                "title": "DeepSpeed Memory Documentation",
                "url": "https://deepspeed.readthedocs.io/en/latest/memory.html",
                "description": "Official DeepSpeed memory requirements and formulas",
            }
        )
    elif engine_type == "megatron_lm" or engine_type == "megatron_deepspeed":
        references.append(
            {
                "title": "NVIDIA Megatron-LM",
                "url": "https://github.com/NVIDIA/Megatron-LM",
                "description": "Megatron-LM tensor and pipeline parallelism",
            }
        )
    elif engine_type == "fsdp":
        references.append(
            {
                "title": "PyTorch FSDP Documentation",
                "url": "https://pytorch.org/docs/stable/fsdp.html",
                "description": "PyTorch Fully Sharded Data Parallel documentation",
            }
        )

    return references


def main() -> None:
    """Run the development server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
