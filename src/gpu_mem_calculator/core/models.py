"""Data models for GPU memory calculation."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EngineType(str, Enum):
    """Supported training engine types."""

    PYTORCH_DDP = "pytorch_ddp"
    DEEPSPEED = "deepspeed"
    MEGATRON_LM = "megatron_lm"
    FSDP = "fsdp"
    MEGATRON_DEEPSPEED = "megatron_deepspeed"


class OptimizerType(str, Enum):
    """Supported optimizer types."""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    ADAMW_8BIT = "adamw_8bit"


class DType(str, Enum):
    """Supported data types."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


class OffloadDevice(str, Enum):
    """CPU offload options."""

    NONE = "none"
    CPU = "cpu"
    NVME = "nvme"


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    name: str = Field(default="custom", description="Model name")
    num_parameters: int = Field(gt=0, description="Total number of parameters")
    num_layers: int = Field(gt=0, description="Number of transformer layers")
    hidden_size: int = Field(gt=0, description="Hidden dimension size")
    num_attention_heads: int = Field(gt=0, description="Number of attention heads")
    vocab_size: int = Field(default=32000, gt=0, description="Vocabulary size")
    max_seq_len: int = Field(default=2048, gt=0, description="Maximum sequence length")
    largest_layer_params: int | None = Field(
        default=None,
        gt=0,
        description="Largest layer parameters (auto-calculated if not provided)",
    )

    # MoE (Mixture of Experts) parameters
    moe_enabled: bool = Field(default=False, description="Enable Mixture of Experts")
    num_experts: int = Field(default=8, ge=1, description="Number of experts in MoE")
    top_k: int = Field(default=2, ge=1, description="Number of experts activated per token (top-k)")
    expert_intermediate_size: int | None = Field(
        default=None,
        gt=0,
        description="Expert intermediate layer size (defaults to 4x hidden_size)",
    )
    shared_expert_intermediate_size: int | None = Field(
        default=None,
        gt=0,
        description="Shared expert intermediate size (for models like GLM with shared experts)",
    )

    @field_validator("largest_layer_params")
    @classmethod
    def calculate_largest_layer(cls, v: int | None, info) -> int | None:
        """Calculate largest layer params if not provided."""
        if v is None:
            # Estimate based on whether MoE is enabled
            hidden = info.data.get("hidden_size", 0)
            moe_enabled = info.data.get("moe_enabled", False)
            num_experts = info.data.get("num_experts", 1)

            if hidden:
                if moe_enabled:
                    # For MoE: largest layer includes expert parameters
                    # Typical expert: hidden_size * intermediate_size
                    expert_intermediate = info.data.get("expert_intermediate_size") or hidden * 4
                    return int(hidden * expert_intermediate * 2)  # Rough estimate for MoE layer
                else:
                    # Dense model: attention output + MLP
                    return int(hidden * hidden * 4)  # Rough estimate
        return v

    @property
    def effective_num_experts(self) -> int:
        """Get effective number of experts (returns 1 if MoE disabled)."""
        return self.num_experts if self.moe_enabled else 1

    @property
    def active_experts(self) -> int:
        """Get number of active experts per token (top_k or 1 if dense)."""
        return self.top_k if self.moe_enabled else 1


class TrainingConfig(BaseModel):
    """Training hyperparameters configuration."""

    batch_size: int = Field(default=1, gt=0, description="Batch size per GPU")
    gradient_accumulation_steps: int = Field(default=1, gt=0, description="Gradient accumulation steps")
    optimizer: OptimizerType = Field(default=OptimizerType.ADAMW, description="Optimizer type")
    dtype: DType = Field(default=DType.BF16, description="Data type for training")
    activation_checkpointing: int = Field(
        default=0,
        ge=0,
        le=4,
        description="Activation checkpointing level (0-4)",
    )


class ParallelismConfig(BaseModel):
    """Parallelism configuration."""

    tensor_parallel_size: int = Field(default=1, ge=1, description="Tensor parallelism degree")
    pipeline_parallel_size: int = Field(default=1, ge=1, description="Pipeline parallelism degree")
    data_parallel_size: int = Field(default=1, ge=1, description="Data parallelism degree")
    sequence_parallel: bool = Field(default=False, description="Enable sequence parallelism")

    @property
    def total_parallel_size(self) -> int:
        """Calculate total parallelism degree."""
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.data_parallel_size
        )


class EngineConfig(BaseModel):
    """Training engine specific configuration."""

    type: EngineType = Field(default=EngineType.PYTORCH_DDP, description="Training engine type")
    zero_stage: int | None = Field(
        default=None,
        ge=0,
        le=3,
        description="DeepSpeed ZeRO stage (only for DeepSpeed engine)",
    )
    offload_optimizer: OffloadDevice = Field(
        default=OffloadDevice.NONE,
        description="CPU offload for optimizer states",
    )
    offload_param: OffloadDevice = Field(
        default=OffloadDevice.NONE,
        description="CPU offload for parameters",
    )
    zero_init: bool = Field(
        default=True,
        description="Use ZeRO initialization (only for DeepSpeed ZeRO-3)",
    )
    sharding_strategy: Literal["no_shard", "shard_grad_op", "full_shard"] = Field(
        default="full_shard",
        description="FSDP sharding strategy",
    )


class GPUConfig(BaseModel):
    """Hardware configuration."""

    num_gpus: int = Field(default=1, ge=1, description="Number of GPUs")
    gpu_memory_gb: float = Field(default=80.0, gt=0, description="GPU memory in GB")
    total_gpu_memory_gb: float | None = Field(
        default=None,
        description="Total GPU memory (calculated if not provided)",
    )

    @field_validator("total_gpu_memory_gb")
    @classmethod
    def calculate_total_memory(cls, v: float | None, info) -> float | None:
        """Calculate total GPU memory if not provided."""
        if v is None:
            num_gpus = info.data.get("num_gpus", 1)
            gpu_mem = info.data.get("gpu_memory_gb", 80.0)
            return num_gpus * gpu_mem
        return v


class MemoryBreakdown(BaseModel):
    """Memory calculation result breakdown."""

    model_params_gb: float = Field(ge=0, description="Model parameters memory in GB")
    gradients_gb: float = Field(ge=0, description="Gradients memory in GB")
    optimizer_states_gb: float = Field(ge=0, description="Optimizer states memory in GB")
    activations_gb: float = Field(ge=0, description="Activations memory in GB")
    overhead_gb: float = Field(default=0.0, ge=0, description="Additional overhead in GB")

    @property
    def total_memory_gb(self) -> float:
        """Total memory in GB."""
        return (
            self.model_params_gb
            + self.gradients_gb
            + self.optimizer_states_gb
            + self.activations_gb
            + self.overhead_gb
        )


class MemoryResult(BaseModel):
    """Complete memory calculation result."""

    total_memory_per_gpu_gb: float = Field(ge=0, description="Total memory per GPU in GB")
    total_memory_all_gpus_gb: float = Field(ge=0, description="Total memory across all GPUs in GB")
    cpu_memory_gb: float = Field(default=0.0, ge=0, description="CPU memory required in GB")
    breakdown: MemoryBreakdown = Field(description="Memory breakdown by component")
    fits_on_gpu: bool = Field(description="Whether the config fits on available GPU")
    memory_utilization_percent: float = Field(ge=0, description="Memory utilization percentage")
    recommended_batch_size: int | None = Field(
        default=None,
        description="Recommended batch size if current doesn't fit",
    )
