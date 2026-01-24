# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU Memory Calculator is a Python application for calculating GPU memory requirements for training and running Large Language Models (LLMs). It supports multiple training engines (PyTorch DDP, DeepSpeed ZeRO, Megatron-LM, FSDP), inference engines (HuggingFace, vLLM, TGI, TensorRT-LLM, SGLang), multi-node training, and can export configurations to training frameworks.

**Technology Stack:**
- Python 3.10+ with Pydantic v2 for data validation
- FastAPI + Jinja2 for web UI
- pytest for testing (239 tests, targeting 95% coverage)
- Development tools: black (100 char line), ruff, mypy (strict mode)

## Common Commands

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_inference.py

# Run with coverage
pytest --cov=gpu_mem_calculator tests/

# Run tests for specific module
pytest tests/test_calculator.py -v

# Run inference tests only
pytest tests/test_inference.py -v
```

### Code Quality
```bash
# Format code
black src/ cli/ web/ tests/

# Check linting
ruff check src/ cli/ web/ tests/

# Type checking
mypy src/

# Run all quality checks
black src/ && ruff check src/ && mypy src/
```

### Web Server
```bash
# Install with web dependencies
pip install -e ".[web]"

# Start web server (port 8000, 5000 may be occupied by macOS ControlCenter)
uvicorn web.app:app --host 0.0.0.0 --port 8000

# Or using python module
python -m uvicorn web.app:app --reload
```

### CLI Usage
```bash
# List available presets
gpu-mem-calc presets

# Calculate with preset
gpu-mem-calc calculate --preset llama2-7b

# Calculate from config file
gpu-mem-calc calculate --config configs/llama2_7b_deepspeed.json

# Quick calculation
gpu-mem-calc quick 7 --gpus 8 --engine deepspeed
```

## Architecture Overview

The codebase follows a modular architecture with clear separation of concerns:

### Core Calculation Engine (`src/gpu_mem_calculator/core/`)

**models.py** - Pydantic models for all configurations:
- `ModelConfig`: Model architecture (layers, hidden_size, num_parameters, MoE settings)
- `TrainingConfig`: Batch size, optimizer, dtype, activation checkpointing
- `ParallelismConfig`: TP, PP, DP sizes
- `EngineConfig`: Training engine type, ZeRO stage, offload settings
- `InferenceConfig`: Inference engine, KV cache quantization, engine-specific params
- `MemoryResult`: Breakdown of memory by component (params, gradients, optimizer, activations)
- `InferenceMemoryResult`: Inference-specific memory breakdown

**calculator.py** - Main orchestrator for training calculations:
- `GPUMemoryCalculator.calculate()` dispatches to appropriate engine
- Engines selected based on `EngineConfig.type` (pytorch_ddp, deepspeed, megatron_lm, fsdp, megatron_deepspeed)
- Returns `MemoryResult` with feasibility analysis and multi-node overhead

**formulas.py** - Memory calculation formulas:
- Component-by-component formulas (parameters, gradients, optimizer states, activations)
- Optimizer-specific calculations (AdamW: 12 bytes/param, AdamW 8-bit: 2 bytes/param, SGD: 4 bytes/param)
- Activation checkpointing multipliers (5 levels)

**multinode.py** - Network overhead calculator:
- `MultiNodeCalculator` calculates AllReduce, AllGather, ReduceScatter, pipeline communication
- Supports different interconnects (InfiniBand, NVLink, Ethernet)
- `HybridParallelismConfig` for automatic TP+PP+DP optimization

### Training Engines (`src/gpu_mem_calculator/engines/`)

All engines inherit from `BaseEngine` which provides:
- Common initialization logic
- `_check_feasibility()`: GPU fitting check with batch size recommendations
- `_create_result()`: MemoryResult construction with multi-node support
- `calculate_moe_activation_multiplier()`: MoE-specific activation memory reduction

**Engine implementations:**
- `pytorch.py`: Baseline DDP (no sharding)
- `deepspeed.py`: ZeRO stages 0-3 with CPU/NVMe offloading
- `megatron.py`: Tensor + pipeline parallelism
- `fsdp.py`: Fully sharded data parallel with sharding strategies
- Each implements `calculate_memory()` returning engine-specific memory breakdown

### Inference Engines (`src/gpu_mem_calculator/inference/`)

**base.py** - `BaseInferenceEngine` provides:
- `_calculate_model_params_bytes()`: Model size calculation (BF16: 2 bytes/param)
- `_get_kv_cache_bytes_per_token()`: KV cache per token based on quantization (NONE: 2, INT8: 1, FP8: 1, INT4: 0.5)
- `_check_feasibility()`: GPU fitting with max batch size estimation

**Engine implementations:**
- `huggingface.py`: Baseline HF Transformers (no optimizations)
- `vllm.py`: PagedAttention with block-based KV cache allocation (20% buffer)
- `tgi.py`: Flash Attention optimization (40% activation reduction)
- `tensorrt_llm.py`: Fused kernels (30% activation reduction)
- `sglang.py`: RadixAttention with tree-based KV cache sharing (~30% memory savings)

**calculator.py** - `InferenceMemoryCalculator`:
- Orchestrates inference engine selection
- `calculate(engine_type)` dispatches to appropriate engine

### Exporters (`src/gpu_mem_calculator/exporters/`)

**manager.py** - `ExportManager`:
- Unified interface for exporting to different frameworks
- `export(format)`: Returns dict with framework-specific config
- `export_to_file()`: Saves config to YAML/JSON files

**Individual exporters:**
- `accelerate.py`: HuggingFace Accelerate config (distributed_type, fsdp_config, deepspeed_config)
- `lightning.py`: PyTorch Lightning Trainer config (strategy, precision, devices)
- `axolotl.py`: Axolotl YAML config (model_type, optimizer, special_tokens)

### Configuration (`src/gpu_mem_calculator/config/`)

**presets.py** - Pre-configured model definitions:
- Dense models: LLaMA 2 (7B, 13B, 70B), GPT-3 (175B)
- MoE models: Mixtral 8x7B, GLM-4 (9B), GLM-4.7 (355B), GLM-4.5 Air (106B), Qwen1.5-MoE-A2.7B, DeepSeek-MoE (16B)
- `load_presets()`: Returns dict of `ModelPreset` objects

**parser.py** - `ConfigParser`:
- Parses JSON config files
- Handles human-readable parameter formats ("7B", "7000M", "7000000000")
- Validates configuration against Pydantic models

### Web Application (`web/`)

**app.py** - FastAPI application:
- Main page serves `index.html` with tabbed interface (Training, Inference, Multi-Node)
- API endpoints: `/api/calculate`, `/api/inference/calculate`, `/api/multinode/calculate`
- `/api/export/*` endpoints for framework config export
- `/api/presets` for loading model presets
- `/api/optimize/batch-size` for finding maximum batch size

**templates/index.html** - Single-page app with:
- Three tabs: Training, Inference, Multi-Node
- Real-time form validation and auto-calculation (1s debounce)
- Engine-specific settings sections (TGI, vLLM, TensorRT-LLM, SGLang)
- Visual memory breakdown with color-coded bar charts
- Accessibility features (ARIA labels, keyboard navigation)

**static/js/app.js** - Frontend logic:
- `GPUMemoryCalculatorApp` class manages UI state
- `calculateTrainingMemory()`, `calculateInferenceMemory()`, `calculateMultiNode()`
- `updateInferenceEngineFields()`: Shows/hides engine-specific settings based on selection
- Form reset, validation, and result display functions

### Utilities (`src/gpu_mem_calculator/utils/`)

**precision.py** - Precision handling utilities:
- `get_precision_bytes(dtype)`: Returns bytes per value (fp32: 4, fp16/bf16: 2, int8: 1, int4: 0.5)

## Key Implementation Patterns

### Adding a New Training Engine

1. Create engine class in `src/gpu_mem_calculator/engines/new_engine.py` inheriting from `BaseEngine`
2. Implement `calculate_memory()` method returning `MemoryResult` with breakdown
3. Add engine type to `EngineType` enum in `models.py`
4. Add engine to import and match statement in `calculator.py`
5. Add comprehensive tests in `tests/test_comprehensive.py`

### Adding a New Inference Engine

1. Create engine class in `src/gpu_mem_calculator/inference/new_engine.py` inheriting from `BaseInferenceEngine`
2. Implement `calculate_memory()` method returning `InferenceMemoryResult`
3. Add engine type to `InferenceEngineType` enum in `models.py`
4. Add engine-specific parameters to `InferenceConfig` in `models.py`
5. Add engine to import, type alias, and match statement in `inference/calculator.py`
6. Update web UI:
   - Add option to inference engine dropdown in `index.html`
   - Add engine-specific settings section
   - Update `updateInferenceEngineFields()` in `app.js`
   - Update `calculateInferenceMemory()` to collect new parameters
   - Update `resetInferenceForm()` to reset new fields
7. Update backend API in `web/app.py` to accept new parameters
8. Add tests in `tests/test_inference.py`

### Adding Model Presets

Edit `src/gpu_mem_calculator/config/presets.py`:
```python
"model-name": ModelPreset(
    name="Human Readable Name",
    num_parameters=7_000_000_000,  # Use underscores for readability
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    vocab_size=32000,
    max_seq_len=4096,
    moe_enabled=False,  # Set True for MoE models
    num_experts=8,      # Required if moe_enabled=True
    top_k=2,            # Required if moe_enabled=True
    description="Model description",
),
```

### MoE Model Support

MoE (Mixture of Experts) models require special handling:
- `ModelConfig.moe_enabled`: Enable MoE mode
- `ModelConfig.num_experts`: Total number of experts
- `ModelConfig.top_k`: Number of active experts per token
- `BaseEngine.calculate_moe_activation_multiplier()`: Returns activation memory reduction ratio
- Activation memory scales with `top_k / num_experts` instead of full parameter count
- All expert parameters stored in memory, but only `top_k` activated per token

### Parameter Format Parsing

The calculator accepts flexible parameter formats:
- "7B" → 7,000,000,000
- "7000M" → 7,000,000,000
- "7000000000" → 7,000,000,000
- Handled by `ConfigParser._parse_num_parameters()` in `config/parser.py`

## Important Constraints

### Type Safety
- All code must pass `mypy src/` with strict type checking enabled
- Use `|` for union types (Python 3.10+), not `typing.Union`
- Always specify return types on functions
- Use Pydantic `Field()` for model validation with descriptions

### Code Style
- Black formatter with 100 character line length
- Ruff for linting (E, F, W, I, N, B, UP rule sets)
- All strings must be double-quoted, not single-quoted

### Testing Requirements
- All new features require comprehensive tests
- Target 95% code coverage across modules
- Tests are organized by module: `test_calculator.py`, `test_inference.py`, `test_exporters.py`, `test_multinode.py`
- Use parameterized tests for testing multiple configurations

### Memory Formulas

**Core Components (verified against authoritative sources):**
- Model params: `num_params × dtype_bytes` (BF16: 2, FP32: 4)
- Gradients: `num_params × dtype_bytes` (same as model)
- Optimizer: `num_params × optimizer_bytes` (AdamW: 12, AdamW 8-bit: 2, SGD: 4)
- Activations: `batch × seq × hidden × layers × bytes_per_value × checkpointing_multiplier`

**ZeRO Stages:**
- ZeRO-0: `16×params + activations`
- ZeRO-1: `2×params + 2×params + (12×params)/gpus + activations`
- ZeRO-2: `2×params + (2×params)/gpus + (12×params)/gpus + activations`
- ZeRO-3: `largest_layer + (16×params)/gpus + activations` (largest_layer ≈ 4×params/10)

**MoE Activation Memory:**
```python
activation_ratio = top_k / num_experts  # Only active experts
router_overhead = 0.1                   # 10% gating overhead
multiplier = min(1.0, activation_ratio + router_overhead)
```

### Web UI State Management

The web UI uses vanilla JavaScript with class-based state management:
- `GPUMemoryCalculatorApp` class encapsulates all UI logic
- Preset models are shared across all tabs (same data source)
- Engine-specific settings sections show/hide based on dropdown selection
- Auto-calculation with 1s debounce to prevent excessive API calls
- Results update in-place without page refresh

### API Response Format

All calculation endpoints return JSON with breakdown:
```json
{
  "total_memory_per_gpu_gb": 122.04,
  "total_memory_all_gpus_gb": 122.04,
  "breakdown": {
    "model_params_gb": 13.04,
    "kv_cache_gb": 76.80,
    "activations_gb": 32.00,
    "overhead_gb": 0.15
  },
  "fits_on_gpu": false,
  "memory_utilization_percent": 152.5
}
```

### Date Convention

All dates in documentation and code should use **2026** as the development year.

## File Organization Notes

- **Source code**: `src/gpu_mem_calculator/` (installed as editable package)
- **Tests**: `tests/` at repository root
- **Web assets**: `web/templates/`, `web/static/`
- **Config examples**: `configs/` (not in source tree)
- **Documentation**: `docs/` (markdown files)
- **Presets**: `src/gpu_mem_calculator/config/presets.py` (single file with all models)

## Common Pitfalls

1. **Forgetting to update calculator.py when adding engines**: Both the import and match statement need updating
2. **Missing web UI parameters**: When adding inference engine parameters, update HTML, JavaScript, AND backend API
3. **MoE activation memory**: Don't forget MoE models have reduced activation memory (use `calculate_moe_activation_multiplier()`)
4. **Type hints**: Always specify return types, mypy runs in CI
5. **Test parameter names**: Inference tests use parameterized fixtures - ensure parameter names match
6. **ZeRO stage validation**: ZeRO stage must be 0-3, offload settings only apply to ZeRO-3
7. **Tensor parallel sharding**: Model parameters sharded across TP GPUs, divide by `tensor_parallel_size`
8. **Formatter confusion**: Use `black src/ cli/ web/ tests/` (include all directories, not just src/)
