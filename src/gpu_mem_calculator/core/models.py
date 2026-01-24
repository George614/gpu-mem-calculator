"""Data models for GPU memory calculation."""

from __future__ import annotations

from enum import Enum
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core.core_schema import ValidationInfo as FieldValidationInfo


class EngineType(str, Enum):
    """Supported training engine types."""

    PYTORCH_DDP = "pytorch_ddp"
    DEEPSPEED = "deepspeed"
    MEGATRON_LM = "megatron_lm"
    FSDP = "fsdp"
    MEGATRON_DEEPSPEED = "megatron_deepspeed"


class InferenceEngineType(str, Enum):
    """Supported inference engine types."""

    HUGGINGFACE = "huggingface"
    VLLM = "vllm"
    TGI = "tgi"
    TENSORRT_LLM = "tensorrt_llm"
    TRTLLM = "trtllm"
    SGLANG = "sglang"


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

    @model_validator(mode="after")
    def calculate_largest_layer(self) -> ModelConfig:
        """Calculate largest layer params if not provided."""
        if self.largest_layer_params is not None:
            return self
        # Calculate it
        hidden = self.hidden_size
        moe_enabled = self.moe_enabled

        if hidden and moe_enabled:
            # For MoE: largest layer includes expert parameters
            expert_intermediate = self.expert_intermediate_size or hidden * 4
            self.largest_layer_params = int(hidden * expert_intermediate * 2)
        elif hidden:
            # Dense model: attention output + MLP
            self.largest_layer_params = int(hidden * hidden * 4)
        return self

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
    gradient_accumulation_steps: int = Field(
        default=1,
        gt=0,
        description="Gradient accumulation steps",
    )
    optimizer: OptimizerType = Field(default=OptimizerType.ADAMW, description="Optimizer type")
    dtype: DType = Field(default=DType.BF16, description="Data type for training")
    activation_checkpointing: int = Field(
        default=0,
        ge=0,
        le=4,
        description="Activation checkpointing level (0-4)",
    )

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size with gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


class ParallelismConfig(BaseModel):
    """Parallelism configuration."""

    tensor_parallel_size: int = Field(default=1, ge=1, description="Tensor parallelism degree")
    pipeline_parallel_size: int = Field(default=1, ge=1, description="Pipeline parallelism degree")
    data_parallel_size: int = Field(default=1, ge=1, description="Data parallelism degree")
    sequence_parallel: bool = Field(default=False, description="Enable sequence parallelism")

    @property
    def total_parallel_size(self) -> int:
        """Calculate total parallelism degree."""
        return self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size


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
    def calculate_total_memory(cls, v: float | None, info: FieldValidationInfo) -> float | None:
        """Calculate total GPU memory if not provided."""
        if v is None:
            num_gpus = cast(int, info.data.get("num_gpus", 1))
            gpu_mem = cast(float, info.data.get("gpu_memory_gb", 80.0))
            return num_gpus * gpu_mem
        return v


class InterconnectType(str, Enum):
    """Multi-node interconnect types."""

    INFINIBAND = "infiniband"
    NVLINK = "nvlink"
    ETHERNET_10G = "ethernet_10g"
    ETHERNET_25G = "ethernet_25g"
    ETHERNET_100G = "ethernet_100g"
    ETHERNET_200G = "ethernet_200g"


class NodeConfig(BaseModel):
    """Multi-node configuration."""

    num_nodes: int = Field(default=1, ge=1, description="Number of nodes")
    gpus_per_node: int | None = Field(
        default=None,
        ge=1,
        description="GPUs per node (calculated from num_gpus if not provided)",
    )
    interconnect_type: InterconnectType = Field(
        default=InterconnectType.INFINIBAND,
        description="Interconnect type between nodes",
    )
    interconnect_bandwidth_gbps: float | None = Field(
        default=None,
        gt=0,
        description="Interconnect bandwidth in Gbps (default: auto from type)",
    )

    @field_validator("gpus_per_node")
    @classmethod
    def calculate_gpus_per_node(cls, v: int | None, info: FieldValidationInfo) -> int | None:
        """Calculate GPUs per node if not provided."""
        if v is None:
            num_nodes = cast(int, info.data.get("num_nodes", 1))
            num_gpus = cast(int, info.data.get("num_gpus", 1))
            return max(1, num_gpus // num_nodes)
        return v

    def get_interconnect_bandwidth_gbps(self) -> float:
        """Get interconnect bandwidth in Gbps.

        Returns bandwidth from config or default based on interconnect type.
        """
        if self.interconnect_bandwidth_gbps:
            return self.interconnect_bandwidth_gbps

        # Default bandwidth values for each interconnect type
        bandwidth_defaults = {
            InterconnectType.INFINIBAND: 200.0,  # HDR200 InfiniBand
            InterconnectType.NVLINK: 300.0,  # NVLink/NVSwitch
            InterconnectType.ETHERNET_10G: 10.0,
            InterconnectType.ETHERNET_25G: 25.0,
            InterconnectType.ETHERNET_100G: 100.0,
            InterconnectType.ETHERNET_200G: 200.0,
        }
        return bandwidth_defaults.get(self.interconnect_type, 100.0)

    @property
    def is_multi_node(self) -> bool:
        """Check if this is a multi-node configuration."""
        return self.num_nodes > 1


class NetworkOverhead(BaseModel):
    """Network communication overhead for multi-node training."""

    allreduce_gb: float = Field(default=0.0, ge=0, description="AllReduce communication in GB")
    allgather_gb: float = Field(default=0.0, ge=0, description="AllGather communication in GB")
    reducescatter_gb: float = Field(
        default=0.0, ge=0, description="ReduceScatter communication in GB"
    )
    point_to_point_gb: float = Field(
        default=0.0, ge=0, description="Point-to-point communication in GB"
    )
    total_overhead_gb: float = Field(default=0.0, ge=0, description="Total network overhead in GB")
    estimated_overhead_ms_per_step: float | None = Field(
        default=None,
        description="Estimated communication overhead per training step in milliseconds",
    )


class HybridParallelismConfig(BaseModel):
    """Hybrid parallelism configuration for optimal multi-node scaling."""

    auto_optimize: bool = Field(
        default=False,
        description="Automatically optimize parallelism strategy for given hardware",
    )
    target_gpu_utilization: float = Field(
        default=0.85,
        gt=0.0,
        le=1.0,
        description="Target GPU memory utilization (0.0-1.0)",
    )
    prefer_pipeline_parallel: bool = Field(
        default=False,
        description="Prefer pipeline parallelism over data parallel for multi-node",
    )
    max_pipeline_chunks: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of pipeline chunks (virtual stages)",
    )
    enable_sequence_parallel: bool = Field(
        default=True,
        description="Enable sequence parallelism for long sequences",
    )
    sequence_parallel_threshold: int = Field(
        default=4096,
        ge=1,
        description="Sequence length threshold for enabling sequence parallel",
    )


class MemoryBreakdown(BaseModel):
    """Memory calculation result breakdown."""

    model_config = ConfigDict(protected_namespaces=())

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
    network_overhead: NetworkOverhead | None = Field(
        default=None,
        description="Network communication overhead for multi-node training",
    )
    fits_on_gpu: bool = Field(description="Whether the config fits on available GPU")
    memory_utilization_percent: float = Field(ge=0, description="Memory utilization percentage")
    recommended_batch_size: int | None = Field(
        default=None,
        description="Recommended batch size if current doesn't fit",
    )
    multi_node_info: dict | None = Field(
        default=None,
        description="Additional multi-node configuration info",
    )


class KVCacheQuantization(str, Enum):
    """KV cache quantization options."""

    NONE = "none"
    INT8 = "int8"
    FP8 = "fp8"
    INT4 = "int4"


class InferenceMemoryBreakdown(BaseModel):
    """Memory breakdown for inference workloads."""

    model_config = ConfigDict(protected_namespaces=())

    model_params_gb: float = Field(ge=0, description="Model parameters memory in GB")
    kv_cache_gb: float = Field(ge=0, description="KV cache memory in GB")
    activations_gb: float = Field(ge=0, description="Activation memory in GB")
    overhead_gb: float = Field(default=0.0, ge=0, description="Additional overhead in GB")

    @property
    def total_memory_gb(self) -> float:
        """Total memory in GB."""
        return self.model_params_gb + self.kv_cache_gb + self.activations_gb + self.overhead_gb


class InferenceConfig(BaseModel):
    """Inference-specific configuration."""

    batch_size: int = Field(default=1, gt=0, description="Batch size for inference")
    max_seq_len: int | None = Field(
        default=None,
        gt=0,
        description="Override max sequence length for inference (default: use model config)",
    )
    kv_cache_quantization: KVCacheQuantization = Field(
        default=KVCacheQuantization.NONE,
        description="KV cache quantization type",
    )
    use_kv_cache: bool = Field(default=True, description="Enable KV cache for generation")
    tensor_parallel_size: int = Field(default=1, ge=1, description="Tensor parallelism degree")
    enable_streaming: bool = Field(default=False, description="Enable streaming inference")

    # Common inference options
    gpu_memory_utilization: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="GPU memory utilization target (0.0-1.0)",
    )

    # TGI-specific options
    max_total_tokens: int | None = Field(
        default=None,
        gt=0,
        description="TGI: Maximum total tokens (input + output) - defines memory budget",
    )
    max_input_tokens: int | None = Field(
        default=None,
        gt=0,
        description="TGI: Maximum input tokens",
    )
    max_batch_total_tokens: int | None = Field(
        default=None,
        gt=0,
        description="TGI: Maximum total tokens across all batches",
    )
    tgi_quantize: Literal[
        "none",
        "awq",
        "eetq",
        "exl2",
        "gptq",
        "marlin",
        "bitsandbytes",
        "bitsandbytes-nf4",
        "bitsandbytes-fp4",
        "fp8",
    ] = Field(
        default="none",
        description="TGI: Weight quantization method",
    )
    tgi_dtype: Literal["float16", "bfloat16"] = Field(
        default="bfloat16",
        description="TGI: Data type for inference",
    )
    sharded: bool = Field(default=False, description="TGI: Enable sharded inference")
    num_shard: int | None = Field(
        default=None,
        ge=1,
        description="TGI: Number of shards for sharded inference",
    )

    # vLLM-specific options
    block_size: int | None = Field(
        default=None,
        ge=1,
        description="vLLM: Block size for KV cache management (default: 16)",
    )
    swap_space_gb: float = Field(default=0.0, ge=0.0, description="vLLM: CPU swap space in GB")
    enable_prefix_caching: bool = Field(default=False, description="vLLM: Enable prefix caching")
    enforce_eager: bool = Field(
        default=False,
        description="vLLM: Enable eager mode (disable CUDA graph)",
    )
    max_num_batched_tokens: int | None = Field(
        default=None,
        gt=0,
        description="vLLM: Maximum number of batched tokens",
    )
    max_num_seqs: int | None = Field(
        default=None,
        gt=0,
        description="vLLM: Maximum number of sequences in a batch",
    )
    vllm_quantization: Literal["none", "awq", "gptq", "squeezellm", "fp8"] = Field(
        default="none",
        description="vLLM: Weight quantization method",
    )

    # TensorRT-LLM-specific options
    trt_max_batch_size: int | None = Field(
        default=None,
        gt=0,
        description="TensorRT-LLM: Maximum batch size",
    )
    trt_max_input_len: int | None = Field(
        default=None,
        gt=0,
        description="TensorRT-LLM: Maximum input length",
    )
    trt_max_seq_len: int | None = Field(
        default=None,
        gt=0,
        description="TensorRT-LLM: Maximum sequence length",
    )
    trt_max_beam_width: int | None = Field(
        default=None,
        ge=1,
        description="TensorRT-LLM: Maximum beam width for beam search",
    )

    # SGLang-specific options
    chunk_size: int | None = Field(
        default=None,
        ge=1,
        description="SGLang: Prefill chunk size for long contexts (default: 8192)",
    )
    max_running_requests: int | None = Field(
        default=None,
        ge=1,
        description="SGLang: Maximum number of concurrent requests",
    )
    disable_radix_cache: bool = Field(
        default=False,
        description="SGLang: Disable RadixAttention cache (for debugging)",
    )
    enable_p2p: bool = Field(
        default=False,
        description="SGLang: Enable P2P attention for multi-GPU",
    )
    disable_custom_all_reduce: bool = Field(
        default=False,
        description="SGLang: Disable custom all-reduce kernel",
    )
    attention_backend: Literal["flashinfer", "triton", "torch"] = Field(
        default="flashinfer",
        description="SGLang: Attention backend implementation",
    )
    enable_torch_compile: bool = Field(
        default=False,
        description="SGLang: Enable torch.compile for model optimization",
    )
    radix_cache_max_seq_len: int | None = Field(
        default=None,
        gt=0,
        description="SGLang: Maximum sequence length for RadixCache",
    )
    speculative_algo: Literal["default", "medusa", "eagle"] = Field(
        default="default",
        description="SGLang: Speculative decoding algorithm",
    )
    multi_lora_enabled: bool = Field(default=False, description="SGLang: Enable multi-LoRA serving")


class InferenceMemoryResult(BaseModel):
    """Inference memory calculation result."""

    total_memory_per_gpu_gb: float = Field(ge=0, description="Total memory per GPU in GB")
    total_memory_all_gpus_gb: float = Field(ge=0, description="Total memory across all GPUs in GB")
    breakdown: InferenceMemoryBreakdown = Field(description="Memory breakdown by component")
    fits_on_gpu: bool = Field(description="Whether the config fits on available GPU")
    memory_utilization_percent: float = Field(ge=0, description="Memory utilization percentage")
    max_supported_batch_size: int | None = Field(
        default=None,
        description="Maximum batch size that fits in GPU memory",
    )
    estimated_throughput_tokens_per_sec: float | None = Field(
        default=None,
        description="Estimated throughput in tokens/second",
    )
