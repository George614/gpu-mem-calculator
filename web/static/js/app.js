// GPU Memory Calculator - Main Application Logic

class GPUMemCalculator {
    constructor() {
        this.apiBase = '/api';
        this.autoCalculateEnabled = true;
        this.debounceTimer = null;
        this.debounceDelay = 500; // ms
        this.isApplyingConfig = false; // Flag to prevent auto-calc during preset loads
        this.initEventListeners();
        this.initAutoCalculate();
    }

    initEventListeners() {
        // Preset selection
        document.getElementById('preset-select').addEventListener('change', (e) => {
            if (e.target.value !== 'custom') {
                this.loadPreset(e.target.value);
            }
        });

        // Batch size slider sync
        const batchSizeInput = document.getElementById('batch-size');
        const batchSizeSlider = document.getElementById('batch-size-slider');

        batchSizeSlider.addEventListener('input', (e) => {
            batchSizeInput.value = e.target.value;
        });

        batchSizeInput.addEventListener('input', (e) => {
            batchSizeSlider.value = e.target.value;
        });

        // GPU memory dropdown
        document.getElementById('gpu-model').addEventListener('change', (e) => {
            const customInput = document.getElementById('gpu-mem-custom');
            if (e.target.value === 'custom') {
                customInput.style.display = 'block';
            } else {
                customInput.style.display = 'none';
                customInput.value = e.target.value;
            }
        });

        // Engine type change - update dynamic fields
        document.getElementById('engine-type').addEventListener('change', (e) => {
            this.updateEngineFields(e.target.value);
        });

        // Parallelism change - update effective GPUs
        const parallelismInputs = ['tensor-pp', 'pipeline-pp', 'data-pp'];
        parallelismInputs.forEach(id => {
            document.getElementById(id).addEventListener('input', () => {
                this.updateEffectiveGPUs();
            });
        });

        // MoE checkbox - toggle visibility of MoE fields
        document.getElementById('moe-enabled').addEventListener('change', (e) => {
            this.toggleMoEFields(e.target.checked);
        });

        // MoE field changes - update display
        ['num-experts', 'top-k'].forEach(id => {
            document.getElementById(id).addEventListener('input', () => {
                this.updateMoEDisplay();
            });
        });

        // Calculate button
        document.getElementById('calculate-btn').addEventListener('click', () => {
            this.calculateMemory();
        });

        // Reset button
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetForm();
        });

        // Save config button
        document.getElementById('save-config-btn').addEventListener('click', () => {
            this.saveConfig();
        });

        // Copy JSON button
        document.getElementById('copy-json-btn').addEventListener('click', () => {
            this.copyConfigJSON();
        });

        // Initialize engine fields
        this.updateEngineFields('deepspeed');
        this.updateEffectiveGPUs();
    }

    initAutoCalculate() {
        // List of all input IDs that should trigger auto-calculation
        const autoCalcInputs = [
            // Model settings
            'model-name', 'num-params', 'num-layers', 'hidden-size', 'num-heads',
            'vocab-size', 'seq-len',
            // MoE settings
            'moe-enabled', 'num-experts', 'top-k', 'expert-intermediate-size', 'shared-expert-size',
            // Training settings
            'batch-size', 'batch-size-slider', 'grad-accum', 'optimizer', 'dtype',
            'activation-checkpointing',
            // Parallelism
            'tensor-pp', 'pipeline-pp', 'data-pp', 'seq-parallel',
            // Engine settings
            'engine-type', 'zero-stage', 'offload-optimizer', 'offload-param',
            'zero-init', 'sharding-strategy', 'use-distributed-optimizer',
            'num-micro-batches', 'gradient-clipping', 'weight-decay', 'lr', 'warmup-steps',
            // Hardware
            'num-gpus', 'gpu-model', 'gpu-mem-custom',
        ];

        // Add event listeners to all inputs
        autoCalcInputs.forEach(id => {
            const element = document.getElementById(id);
            if (!element) return;

            // Use 'change' event for selects and checkboxes
            // Use 'input' event for text/number inputs
            const eventType = (element.tagName === 'SELECT' ||
                              element.tagName === 'INPUT' &&
                              (element.type === 'checkbox' || element.type === 'range'))
                              ? 'input' : 'input';

            element.addEventListener(eventType, () => {
                this.scheduleAutoCalculate();
            });
        });
    }

    scheduleAutoCalculate() {
        // Don't auto-calculate if currently applying a config (preset load)
        if (this.isApplyingConfig) return;

        // Don't auto-calculate if disabled
        if (!this.autoCalculateEnabled) return;

        // Clear existing timer
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }

        // Schedule new calculation
        this.debounceTimer = setTimeout(() => {
            this.calculateMemory();
        }, this.debounceDelay);
    }

    setAutoCalculate(enabled) {
        this.autoCalculateEnabled = enabled;
    }

    async loadPreset(presetName) {
        try {
            const response = await fetch(`${this.apiBase}/preset/${presetName}`);
            if (!response.ok) {
                throw new Error(`Failed to load preset: ${presetName}`);
            }

            const config = await response.json();
            this.applyConfig(config);
        } catch (error) {
            this.showError(`Failed to load preset: ${error.message}`);
        }
    }

    applyConfig(config) {
        // Set flag to prevent auto-calculation during config load
        this.isApplyingConfig = true;

        // Apply model configuration
        if (config.model) {
            if (config.model.name) document.getElementById('model-name').value = config.model.name;
            if (config.model.num_parameters) document.getElementById('num-params').value = config.model.num_parameters;
            if (config.model.num_layers) document.getElementById('num-layers').value = config.model.num_layers;
            if (config.model.hidden_size) document.getElementById('hidden-size').value = config.model.hidden_size;
            if (config.model.num_attention_heads) document.getElementById('num-heads').value = config.model.num_attention_heads;
            if (config.model.vocab_size) document.getElementById('vocab-size').value = config.model.vocab_size;
            if (config.model.max_seq_len) document.getElementById('seq-len').value = config.model.max_seq_len;
        }

        // Apply MoE configuration
        if (config.model.moe_enabled !== undefined) {
            document.getElementById('moe-enabled').checked = config.model.moe_enabled;
            this.toggleMoEFields(config.model.moe_enabled);

            if (config.model.moe_enabled) {
                if (config.model.num_experts) {
                    document.getElementById('num-experts').value = config.model.num_experts;
                }
                if (config.model.top_k) {
                    document.getElementById('top-k').value = config.model.top_k;
                }
                if (config.model.expert_intermediate_size) {
                    document.getElementById('expert-intermediate-size').value = config.model.expert_intermediate_size;
                }
                if (config.model.shared_expert_intermediate_size) {
                    document.getElementById('shared-expert-size').value = config.model.shared_expert_intermediate_size;
                }
                this.updateMoEDisplay();
            }
        }

        // Apply training configuration
        if (config.training) {
            if (config.training.batch_size) {
                document.getElementById('batch-size').value = config.training.batch_size;
                document.getElementById('batch-size-slider').value = config.training.batch_size;
            }
            if (config.training.gradient_accumulation_steps) {
                document.getElementById('grad-accum').value = config.training.gradient_accumulation_steps;
            }
            if (config.training.optimizer) document.getElementById('optimizer').value = config.training.optimizer;
            if (config.training.dtype) document.getElementById('dtype').value = config.training.dtype;
            if (config.training.activation_checkpointing !== undefined) {
                document.getElementById('activation-checkpointing').value = config.training.activation_checkpointing;
            }
        }

        // Apply parallelism configuration
        if (config.parallelism) {
            if (config.parallelism.tensor_parallel_size) {
                document.getElementById('tensor-pp').value = config.parallelism.tensor_parallel_size;
            }
            if (config.parallelism.pipeline_parallel_size) {
                document.getElementById('pipeline-pp').value = config.parallelism.pipeline_parallel_size;
            }
            if (config.parallelism.data_parallel_size) {
                document.getElementById('data-pp').value = config.parallelism.data_parallel_size;
            }
            if (config.parallelism.sequence_parallel) {
                document.getElementById('seq-parallel').checked = config.parallelism.sequence_parallel;
            }
        }

        // Apply engine configuration
        if (config.engine) {
            if (config.engine.type) {
                document.getElementById('engine-type').value = config.engine.type;
                this.updateEngineFields(config.engine.type);
            }
            if (config.engine.zero_stage !== undefined) {
                document.getElementById('zero-stage').value = config.engine.zero_stage;
            }
            if (config.engine.offload_optimizer) {
                document.getElementById('offload-optimizer').value = config.engine.offload_optimizer;
            }
            if (config.engine.offload_param) {
                document.getElementById('offload-param').value = config.engine.offload_param;
            }
        }

        // Apply hardware configuration
        if (config.hardware) {
            if (config.hardware.num_gpus) document.getElementById('num-gpus').value = config.hardware.num_gpus;
            if (config.hardware.gpu_memory_gb) {
                document.getElementById('gpu-model').value = config.hardware.gpu_memory_gb;
                document.getElementById('gpu-mem-custom').value = config.hardware.gpu_memory_gb;
            }
        }

        this.updateEffectiveGPUs();

        // Re-enable auto-calculation and trigger calculation
        setTimeout(() => {
            this.isApplyingConfig = false;
            this.calculateMemory();
        }, 100);
    }

    updateEngineFields(engineType) {
        const zeroStageGroup = document.getElementById('zero-stage-group');
        const offloadOptGroup = document.getElementById('offload-opt-group');
        const offloadParamGroup = document.getElementById('offload-param-group');
        const zeroInitGroup = document.getElementById('zero-init-group');
        const shardingStrategyGroup = document.getElementById('sharding-strategy-group');
        const megatronOptions = document.getElementById('megatron-options');

        // Hide all first
        zeroStageGroup.style.display = 'none';
        offloadOptGroup.style.display = 'none';
        offloadParamGroup.style.display = 'none';
        zeroInitGroup.style.display = 'none';
        shardingStrategyGroup.style.display = 'none';
        megatronOptions.style.display = 'none';

        // Show/hide fields based on engine type
        switch (engineType) {
            case 'deepspeed':
            case 'megatron_deepspeed':
                zeroStageGroup.style.display = 'block';
                offloadOptGroup.style.display = 'block';
                offloadParamGroup.style.display = 'block';
                zeroInitGroup.style.display = 'block';
                break;
            case 'pytorch_ddp':
            case 'megatron_lm':
                // No special options
                break;
            case 'fsdp':
                shardingStrategyGroup.style.display = 'block';
                break;
        }

        // Show Megatron options for Megatron engines
        if (engineType === 'megatron_lm' || engineType === 'megatron_deepspeed') {
            megatronOptions.style.display = 'block';
        }
    }

    updateEffectiveGPUs() {
        const tensorPP = parseInt(document.getElementById('tensor-pp').value) || 1;
        const pipelinePP = parseInt(document.getElementById('pipeline-pp').value) || 1;
        const dataPP = parseInt(document.getElementById('data-pp').value) || 1;

        const effectiveGPUs = tensorPP * pipelinePP * dataPP;
        document.getElementById('effective-gpus').textContent = effectiveGPUs;
    }

    toggleMoEFields(enabled) {
        const moeFields = document.getElementById('moe-fields');
        moeFields.style.display = enabled ? 'block' : 'none';
        if (enabled) {
            this.updateMoEDisplay();
        }
    }

    updateMoEDisplay() {
        const numExperts = parseInt(document.getElementById('num-experts').value) || 8;
        const topK = parseInt(document.getElementById('top-k').value) || 2;

        document.getElementById('total-experts-display').textContent = numExperts;
        document.getElementById('active-experts-display').textContent = topK;
    }

    collectFormData() {
        // Get GPU memory value
        let gpuMem = document.getElementById('gpu-model').value;
        if (gpuMem === 'custom') {
            gpuMem = parseFloat(document.getElementById('gpu-mem-custom').value);
        } else {
            gpuMem = parseFloat(gpuMem);
        }

        // Get engine type
        const engineType = document.getElementById('engine-type').value;

        // Get MoE parameters
        const moeEnabled = document.getElementById('moe-enabled').checked;
        const expertIntermediateSize = document.getElementById('expert-intermediate-size').value;
        const sharedExpertSize = document.getElementById('shared-expert-size').value;

        return {
            model: {
                name: document.getElementById('model-name').value,
                num_parameters: document.getElementById('num-params').value,
                num_layers: parseInt(document.getElementById('num-layers').value),
                hidden_size: parseInt(document.getElementById('hidden-size').value),
                num_attention_heads: parseInt(document.getElementById('num-heads').value),
                vocab_size: parseInt(document.getElementById('vocab-size').value),
                max_seq_len: parseInt(document.getElementById('seq-len').value),
                moe_enabled: moeEnabled,
                num_experts: moeEnabled ? parseInt(document.getElementById('num-experts').value) : 1,
                top_k: moeEnabled ? parseInt(document.getElementById('top-k').value) : 1,
                expert_intermediate_size: expertIntermediateSize ? parseInt(expertIntermediateSize) : null,
                shared_expert_intermediate_size: sharedExpertSize ? parseInt(sharedExpertSize) : null,
            },
            training: {
                batch_size: parseInt(document.getElementById('batch-size').value),
                gradient_accumulation_steps: parseInt(document.getElementById('grad-accum').value),
                optimizer: document.getElementById('optimizer').value,
                dtype: document.getElementById('dtype').value,
                activation_checkpointing: parseInt(document.getElementById('activation-checkpointing').value),
            },
            parallelism: {
                tensor_parallel_size: parseInt(document.getElementById('tensor-pp').value),
                pipeline_parallel_size: parseInt(document.getElementById('pipeline-pp').value),
                data_parallel_size: parseInt(document.getElementById('data-pp').value),
                sequence_parallel: document.getElementById('seq-parallel').checked,
            },
            engine: {
                type: engineType,
                zero_stage: parseInt(document.getElementById('zero-stage').value),
                offload_optimizer: document.getElementById('offload-optimizer').value,
                offload_param: document.getElementById('offload-param').value,
                zero_init: document.getElementById('zero-init').checked,
                sharding_strategy: document.getElementById('sharding-strategy')?.value || null,
                use_distributed_optimizer: document.getElementById('use-distributed-optimizer')?.checked || false,
                num_micro_batches: parseInt(document.getElementById('num-micro-batches')?.value || 1),
            },
            hardware: {
                num_gpus: parseInt(document.getElementById('num-gpus').value),
                gpu_memory_gb: gpuMem,
            },
        };
    }

    async calculateMemory() {
        const config = this.collectFormData();
        const calculateBtn = document.getElementById('calculate-btn');

        // Show loading state
        calculateBtn.disabled = true;
        calculateBtn.textContent = 'Calculating...';

        try {
            const response = await fetch(`${this.apiBase}/calculate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Calculation failed');
            }

            const result = await response.json();
            this.displayResults(result);
        } catch (error) {
            this.showError(`Calculation failed: ${error.message}`);
        } finally {
            calculateBtn.disabled = false;
            calculateBtn.textContent = 'Calculate';
        }
    }

    displayResults(result) {
        // Main memory results
        document.getElementById('result-per-gpu').textContent = `${result.total_memory_per_gpu_gb.toFixed(2)} GB`;
        document.getElementById('result-total').textContent = `${result.total_memory_all_gpus_gb.toFixed(2)} GB`;
        document.getElementById('result-cpu').textContent = `${result.cpu_memory_gb.toFixed(2)} GB`;

        // Breakdown
        document.getElementById('breakdown-params').textContent = `${result.breakdown.model_params_gb.toFixed(2)} GB`;
        document.getElementById('breakdown-grads').textContent = `${result.breakdown.gradients_gb.toFixed(2)} GB`;
        document.getElementById('breakdown-optimizer').textContent = `${result.breakdown.optimizer_states_gb.toFixed(2)} GB`;
        document.getElementById('breakdown-activations').textContent = `${result.breakdown.activations_gb.toFixed(2)} GB`;
        document.getElementById('breakdown-overhead').textContent = `${result.breakdown.overhead_gb.toFixed(2)} GB`;

        // Update bar chart
        this.updateBarChart(result.breakdown);

        // Feasibility
        const statusEl = document.getElementById('feasibility-status');
        const utilEl = document.getElementById('feasibility-util');
        const recommendedBatchEl = document.getElementById('recommended-batch-container');
        const recommendedBatchValue = document.getElementById('recommended-batch');

        utilEl.textContent = `${result.memory_utilization_percent.toFixed(1)}%`;

        if (result.fits_on_gpu) {
            statusEl.textContent = '✓ Fits on GPU';
            statusEl.className = 'metric-value status-success';
            recommendedBatchEl.style.display = 'none';
        } else {
            statusEl.textContent = '✗ OOM (Out of Memory)';
            statusEl.className = 'metric-value status-danger';
            if (result.recommended_batch_size) {
                recommendedBatchValue.textContent = result.recommended_batch_size;
                recommendedBatchEl.style.display = 'flex';
            }
        }

        // Color code utilization
        if (result.memory_utilization_percent < 80) {
            utilEl.className = 'metric-value status-success';
        } else if (result.memory_utilization_percent < 95) {
            utilEl.className = 'metric-value status-warning';
        } else {
            utilEl.className = 'metric-value status-danger';
        }
    }

    updateBarChart(breakdown) {
        const total = breakdown.model_params_gb + breakdown.gradients_gb +
                     breakdown.optimizer_states_gb + breakdown.activations_gb;

        const paramsPct = (breakdown.model_params_gb / total) * 100;
        const gradsPct = (breakdown.gradients_gb / total) * 100;
        const optimizerPct = (breakdown.optimizer_states_gb / total) * 100;
        const activationsPct = (breakdown.activations_gb / total) * 100;

        document.getElementById('bar-params').style.width = `${paramsPct}%`;
        document.getElementById('bar-grads').style.width = `${gradsPct}%`;
        document.getElementById('bar-optimizer').style.width = `${optimizerPct}%`;
        document.getElementById('bar-activations').style.width = `${activationsPct}%`;
    }

    resetForm() {
        document.getElementById('preset-select').value = 'custom';
        document.getElementById('model-name').value = 'custom-model';
        document.getElementById('num-params').value = '7B';
        document.getElementById('num-layers').value = '32';
        document.getElementById('hidden-size').value = '4096';
        document.getElementById('num-heads').value = '32';
        document.getElementById('vocab-size').value = '32000';
        document.getElementById('seq-len').value = '4096';

        // Reset MoE fields
        document.getElementById('moe-enabled').checked = false;
        document.getElementById('num-experts').value = '8';
        document.getElementById('top-k').value = '2';
        document.getElementById('expert-intermediate-size').value = '';
        document.getElementById('shared-expert-size').value = '';
        this.toggleMoEFields(false);

        document.getElementById('batch-size').value = '4';
        document.getElementById('batch-size-slider').value = '4';
        document.getElementById('grad-accum').value = '4';
        document.getElementById('optimizer').value = 'adamw';
        document.getElementById('dtype').value = 'bf16';
        document.getElementById('activation-checkpointing').value = '2';
        document.getElementById('tensor-pp').value = '1';
        document.getElementById('pipeline-pp').value = '1';
        document.getElementById('data-pp').value = '8';
        document.getElementById('seq-parallel').checked = false;
        document.getElementById('engine-type').value = 'deepspeed';
        document.getElementById('zero-stage').value = '3';
        document.getElementById('offload-optimizer').value = 'cpu';
        document.getElementById('offload-param').value = 'none';
        document.getElementById('zero-init').checked = true;
        document.getElementById('num-gpus').value = '8';
        document.getElementById('gpu-model').value = '80';

        this.updateEngineFields('deepspeed');
        this.updateEffectiveGPUs();

        // Reset results
        document.getElementById('result-per-gpu').textContent = '-- GB';
        document.getElementById('result-total').textContent = '-- GB';
        document.getElementById('result-cpu').textContent = '-- GB';
        document.getElementById('breakdown-params').textContent = '-- GB';
        document.getElementById('breakdown-grads').textContent = '-- GB';
        document.getElementById('breakdown-optimizer').textContent = '-- GB';
        document.getElementById('breakdown-activations').textContent = '-- GB';
        document.getElementById('breakdown-overhead').textContent = '-- GB';
        document.getElementById('feasibility-status').textContent = '--';
        document.getElementById('feasibility-util').textContent = '--%';
    }

    saveConfig() {
        const config = this.collectFormData();
        const jsonStr = JSON.stringify(config, null, 2);
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `gpu-mem-config-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    async copyConfigJSON() {
        const config = this.collectFormData();
        const jsonStr = JSON.stringify(config, null, 2);

        try {
            await navigator.clipboard.writeText(jsonStr);
            this.showError('Config copied to clipboard!', true);
        } catch (error) {
            // Fallback for older browsers
            const textarea = document.createElement('textarea');
            textarea.value = jsonStr;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
            this.showError('Config copied to clipboard!', true);
        }
    }

    showError(message, isSuccess = false) {
        const errorEl = document.getElementById('error-message');
        errorEl.textContent = message;
        errorEl.style.display = 'block';
        errorEl.style.backgroundColor = isSuccess ? 'var(--success-color)' : 'var(--danger-color)';

        setTimeout(() => {
            errorEl.style.display = 'none';
        }, 3000);
    }
}

// Initialize the calculator when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new GPUMemCalculator();
});
