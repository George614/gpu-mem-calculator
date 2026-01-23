// GPU Memory Calculator - Main Application Logic

class GPUMemCalculator {
    constructor() {
        this.apiBase = '/api';
        this.autoCalculateEnabled = true;
        this.debounceTimer = null;
        this.debounceDelay = 1000; // ms - increased from 500 to reduce API calls
        this.isApplyingConfig = false; // Flag to prevent auto-calc during preset loads
        this.lastCalculationTime = 0; // Prevent too frequent calculations
        this.minCalculationInterval = 500; // Minimum time between calculations (ms)
        this.savedConfigs = []; // For comparison mode
        this.initEventListeners();
        this.initAutoCalculate();
        this.initTabListeners();
        this.loadSavedConfigs();
    }

    initEventListeners() {
        // Tab navigation
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab;
                this.switchTab(tabName);
            });
        });

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

        // Show formula details button - use toggle approach
        document.getElementById('show-formula-btn').addEventListener('click', () => {
            this.toggleFormulaExplanation();
        });

        // Initialize engine fields
        this.updateEngineFields('deepspeed');
        this.updateEffectiveGPUs();

        // Store last config for formula explanation
        this.lastConfig = null;
        // Track if formula details are currently visible
        this.formulaDetailsVisible = false;
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

        // Check minimum time between calculations
        const now = Date.now();
        if (now - this.lastCalculationTime < this.minCalculationInterval) {
            return; // Skip this calculation, too soon
        }

        // Clear existing timer
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }

        // Schedule new calculation
        this.debounceTimer = setTimeout(() => {
            this.calculateMemory();
        }, this.debounceDelay);
    }

    /**
     * Client-side validation before making API call
     * Returns {valid: boolean, errors: string[]}
     */
    validateForm() {
        const errors = [];

        // Get form values
        const tensorPP = parseInt(document.getElementById('tensor-pp').value) || 1;
        const pipelinePP = parseInt(document.getElementById('pipeline-pp').value) || 1;
        const dataPP = parseInt(document.getElementById('data-pp').value) || 1;
        const numGPUs = parseInt(document.getElementById('num-gpus').value) || 1;
        const seqParallel = document.getElementById('seq-parallel').checked;
        const engineType = document.getElementById('engine-type').value;
        const zeroStage = parseInt(document.getElementById('zero-stage').value) || 0;
        const moeEnabled = document.getElementById('moe-enabled').checked;
        const numExperts = parseInt(document.getElementById('num-experts').value) || 1;
        const topK = parseInt(document.getElementById('top-k').value) || 1;

        // Validate parallelism consistency
        const effectiveGPUs = tensorPP * pipelinePP * dataPP;
        if (effectiveGPUs !== numGPUs) {
            errors.push(
                `Parallelism mismatch: ${tensorPP}×${pipelinePP}×${dataPP}=${effectiveGPUs} GPUs, ` +
                `but num_gpus=${numGPUs}. These must match.`
            );
        }

        // Validate sequence parallel requires tensor parallel > 1
        if (seqParallel && tensorPP <= 1) {
            errors.push(
                'Sequence parallelism requires tensor_parallel_size > 1, ' +
                `but tensor_pp=${tensorPP}.`
            );
        }

        // Validate ZeRO stages only for DeepSpeed engines
        if (zeroStage > 0 && !['deepspeed', 'megatron_deepspeed'].includes(engineType)) {
            errors.push(
                `ZeRO stages are only supported for DeepSpeed engines, ` +
                `but engine_type='${engineType}' with zero_stage=${zeroStage}.`
            );
        }

        // Validate MoE settings
        if (moeEnabled) {
            if (topK > numExperts) {
                errors.push(
                    `MoE top_k (${topK}) cannot exceed num_experts (${numExperts}).`
                );
            }
            if (numExperts < 1 || numExperts > 256) {
                errors.push(`num_experts must be between 1 and 256, got ${numExperts}.`);
            }
            if (topK < 1 || topK > 8) {
                errors.push(`top_k must be between 1 and 8, got ${topK}.`);
            }
        }

        return {
            valid: errors.length === 0,
            errors: errors
        };
    }

    /**
     * Switch between tabs
     */
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.tab === tabName) {
                btn.classList.add('active');
            }
        });

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
            content.style.display = 'none';
        });

        const activeTab = document.getElementById(`${tabName}-tab`);
        if (activeTab) {
            activeTab.classList.add('active');
            activeTab.style.display = 'block';
        }
    }

    /**
     * Initialize tab-specific event listeners
     */
    initTabListeners() {
        // Inference tab event listeners
        const infCalcBtn = document.getElementById('inference-calculate-btn');
        const infResetBtn = document.getElementById('inference-reset-btn');
        if (infCalcBtn) {
            infCalcBtn.addEventListener('click', () => this.calculateInferenceMemory());
        }
        if (infResetBtn) {
            infResetBtn.addEventListener('click', () => this.resetInferenceForm());
        }

        // GPU memory utilization slider
        const gpuMemUtilSlider = document.getElementById('gpu-memory-util');
        const gpuMemUtilValue = document.getElementById('gpu-memory-util-value');
        if (gpuMemUtilSlider && gpuMemUtilValue) {
            gpuMemUtilSlider.addEventListener('input', (e) => {
                gpuMemUtilValue.textContent = parseFloat(e.target.value).toFixed(2);
            });
        }

        // Multi-node tab event listeners
        const multiCalcBtn = document.getElementById('multinode-calculate-btn');
        const multiResetBtn = document.getElementById('multinode-reset-btn');
        if (multiCalcBtn) {
            multiCalcBtn.addEventListener('click', () => this.calculateMultiNode());
        }
        if (multiResetBtn) {
            multiResetBtn.addEventListener('click', () => this.resetMultiNodeForm());
        }

        // Update total GPUs display
        const numNodesInput = document.getElementById('num-nodes');
        const gpusPerNodeInput = document.getElementById('gpus-per-node');
        const totalGpusSpan = document.getElementById('multinode-total-gpus');

        const updateTotalGpus = () => {
            if (numNodesInput && gpusPerNodeInput && totalGpusSpan) {
                const nodes = parseInt(numNodesInput.value) || 1;
                const gpusPerNode = parseInt(gpusPerNodeInput.value) || 8;
                totalGpusSpan.textContent = nodes * gpusPerNode;
            }
        };

        if (numNodesInput) numNodesInput.addEventListener('input', updateTotalGpus);
        if (gpusPerNodeInput) gpusPerNodeInput.addEventListener('input', updateTotalGpus);

        // Export framework button
        const exportBtn = document.getElementById('export-framework-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.showExportModal());
        }
    }

    /**
     * Load saved configs from localStorage
     */
    loadSavedConfigs() {
        try {
            const saved = localStorage.getItem('gpu-mem-saved-configs');
            if (saved) {
                this.savedConfigs = JSON.parse(saved);
            }
        } catch (e) {
            console.warn('Failed to load saved configs:', e);
            this.savedConfigs = [];
        }
    }

    /**
     * Save current config for comparison
     */
    saveConfigForComparison() {
        const config = this.collectFormData();
        const name = config.model.name || 'unnamed';

        // Add timestamp
        config.savedAt = new Date().toISOString();
        config.id = Date.now();

        this.savedConfigs.push(config);

        // Limit to 10 saved configs
        if (this.savedConfigs.length > 10) {
            this.savedConfigs.shift();
        }

        // Save to localStorage
        try {
            localStorage.setItem('gpu-mem-saved-configs', JSON.stringify(this.savedConfigs));
            this.showError(`Saved config: ${name}`, true);
        } catch (e) {
            this.showError('Failed to save config');
        }
    }

    /**
     * Show comparison modal/panel
     */
    showComparison(configId) {
        const config = this.savedConfigs.find(c => c.id === configId);
        if (!config) return;

        const currentConfig = this.collectFormData();

        // Create comparison HTML
        const comparisonHTML = this.generateComparisonHTML(currentConfig, config);

        // Show in modal (you'll need to add modal HTML to index.html)
        alert('Comparison feature - modal will be added');
    }

    /**
     * Generate HTML for comparison view
     */
    generateComparisonHTML(config1, config2) {
        // Calculate memory for both configs
        // For now, just return placeholder
        return `
            <h3>Configuration Comparison</h3>
            <div class="comparison-container">
                <div class="config-column">
                    <h4>Current Config</h4>
                    <pre>${JSON.stringify(config1, null, 2)}</pre>
                </div>
                <div class="config-column">
                    <h4>Saved Config</h4>
                    <pre>${JSON.stringify(config2, null, 2)}</pre>
                </div>
            </div>
        `;
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
        // Client-side validation first
        const validation = this.validateForm();
        if (!validation.valid) {
            // Show validation errors inline
            this.showError(`Validation error: ${validation.errors[0]}`);
            return;
        }

        const config = this.collectFormData();
        this.lastConfig = config; // Store for formula explanation
        const calculateBtn = document.getElementById('calculate-btn');

        // Update last calculation time
        this.lastCalculationTime = Date.now();

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
                const errorMsg = error.detail?.message || error.detail || 'Calculation failed';
                throw new Error(errorMsg);
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

    async showFormulaExplanation() {
        if (!this.lastConfig) {
            this.showError('Please run a calculation first to see the formula explanation.');
            return;
        }

        try {
            const response = await fetch(`${this.apiBase}/explain-formula`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(this.lastConfig),
            });

            if (!response.ok) {
                throw new Error('Failed to get formula explanation');
            }

            const formulaInfo = await response.json();
            this.displayFormulaExplanation(formulaInfo);
        } catch (error) {
            this.showError(`Failed to load formula explanation: ${error.message}`);
        }
    }

    displayFormulaExplanation(formulaInfo) {
        // Update formula description
        const descEl = document.getElementById('formula-description');
        descEl.innerHTML = `
            <p><strong>Engine:</strong> ${formulaInfo.engine_name}</p>
            <p><strong>Total Memory:</strong> ${formulaInfo.total_memory_gb} GB</p>
            <p>${formulaInfo.formula_description || ''}</p>
        `;

        // Update formula components
        const componentsEl = document.getElementById('formula-components');
        componentsEl.style.display = 'block';

        let componentsHTML = '<h4>Formula Components:</h4><ul class="formula-components-list">';
        formulaInfo.formula_components.forEach(component => {
            componentsHTML += `
                <li>
                    <div class="component-name">${component.name}</div>
                    ${component.formula ? `<div class="component-formula">${component.formula}</div>` : ''}
                    ${component.description ? `<div class="component-calculation">${component.description}</div>` : ''}
                    <div class="component-result">
                        <strong>Result:</strong> ${component.result}
                    </div>
                </li>
            `;
        });
        componentsHTML += '</ul>';
        componentsEl.innerHTML = componentsHTML;

        // Update references
        const refsEl = document.getElementById('references-list');
        const refsContainer = document.querySelector('.formula-references');
        refsContainer.style.display = 'block';

        let refsHTML = '';
        formulaInfo.references.forEach(ref => {
            refsHTML += `<li><a href="${ref.url}" target="_blank" rel="noopener noreferrer">${ref.title}</a></li>`;
        });
        refsEl.innerHTML = refsHTML;

        // Update button text and set visibility flag
        const btn = document.getElementById('show-formula-btn');
        btn.textContent = 'Hide Formula Details';
        this.formulaDetailsVisible = true;
    }

    hideFormulaExplanation() {
        document.getElementById('formula-components').style.display = 'none';
        document.querySelector('.formula-references').style.display = 'none';

        const btn = document.getElementById('show-formula-btn');
        btn.textContent = 'Show Formula Details';
        this.formulaDetailsVisible = false;
    }

    async toggleFormulaExplanation() {
        if (!this.lastConfig) {
            this.showError('Please run a calculation first to see the formula explanation.');
            return;
        }

        if (this.formulaDetailsVisible) {
            // Currently visible, hide it
            this.hideFormulaExplanation();
        } else {
            // Currently hidden, show it
            await this.showFormulaExplanation();
        }
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

    /**
     * Calculate inference memory
     */
    async calculateInferenceMemory() {
        try {
            const config = {
                model: {
                    name: document.getElementById('inference-model-name').value,
                    num_parameters: document.getElementById('inference-num-params').value,
                    num_layers: parseInt(document.getElementById('inference-num-layers').value),
                    hidden_size: parseInt(document.getElementById('inference-hidden-size').value),
                    num_attention_heads: parseInt(document.getElementById('inference-num-heads').value),
                    vocab_size: parseInt(document.getElementById('inference-vocab-size').value),
                    max_seq_len: parseInt(document.getElementById('inference-seq-len').value),
                },
                inference: {
                    engine_type: document.getElementById('inference-engine').value,
                    batch_size: parseInt(document.getElementById('inference-batch-size').value),
                    kv_cache_quantization: document.getElementById('kv-cache-quantization').value,
                    tensor_parallel_size: parseInt(document.getElementById('tensor-parallel-size').value),
                    gpu_memory_utilization: parseFloat(document.getElementById('gpu-memory-util').value),
                },
                hardware: {
                    num_gpus: parseInt(document.getElementById('inference-num-gpus').value),
                    gpu_memory_gb: parseInt(document.getElementById('inference-gpu-model').value),
                },
            };

            const response = await fetch(`${this.apiBase}/inference/calculate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            });

            if (!response.ok) {
                throw new Error('Failed to calculate inference memory');
            }

            const result = await response.json();
            this.displayInferenceResults(result);
        } catch (error) {
            this.showError(`Error: ${error.message}`);
        }
    }

    displayInferenceResults(result) {
        document.getElementById('inference-result-per-gpu').textContent = `${result.total_memory_per_gpu_gb.toFixed(2)} GB`;
        document.getElementById('inference-result-total').textContent = `${result.total_memory_all_gpus_gb.toFixed(2)} GB`;
        document.getElementById('inference-result-params').textContent = `${result.model_params_gb.toFixed(2)} GB`;
        document.getElementById('inference-result-kv-cache').textContent = `${result.kv_cache_gb.toFixed(2)} GB`;
        document.getElementById('inference-result-activations').textContent = `${result.activations_gb.toFixed(2)} GB`;
        document.getElementById('inference-max-batch').textContent = result.max_supported_batch_size || 'N/A';
        document.getElementById('inference-throughput').textContent = `${result.estimated_throughput_tokens_per_sec.toFixed(0)} tokens/sec`;
        document.getElementById('inference-fits').textContent = result.fits_on_gpu ? '✓ Yes' : '✗ No';
        document.getElementById('inference-fits').style.color = result.fits_on_gpu ? 'var(--success-color)' : 'var(--danger-color)';
        document.getElementById('inference-utilization').textContent = `${(result.utilization * 100).toFixed(1)}%`;
    }

    resetInferenceForm() {
        document.getElementById('inference-preset-select').value = 'custom';
        document.getElementById('inference-model-name').value = 'custom-model';
        document.getElementById('inference-num-params').value = '7B';
        document.getElementById('inference-num-layers').value = '32';
        document.getElementById('inference-hidden-size').value = '4096';
        document.getElementById('inference-num-heads').value = '32';
        document.getElementById('inference-vocab-size').value = '32000';
        document.getElementById('inference-seq-len').value = '4096';
        document.getElementById('inference-batch-size').value = '32';
        document.getElementById('kv-cache-quantization').value = 'none';
        document.getElementById('tensor-parallel-size').value = '1';
        document.getElementById('gpu-memory-util').value = '0.9';
        document.getElementById('gpu-memory-util-value').textContent = '0.90';
        document.getElementById('inference-num-gpus').value = '1';
        document.getElementById('inference-gpu-model').value = '80';

        // Clear results
        document.getElementById('inference-result-per-gpu').textContent = '-- GB';
        document.getElementById('inference-result-total').textContent = '-- GB';
        document.getElementById('inference-result-params').textContent = '-- GB';
        document.getElementById('inference-result-kv-cache').textContent = '-- GB';
        document.getElementById('inference-result-activations').textContent = '-- GB';
        document.getElementById('inference-max-batch').textContent = '--';
        document.getElementById('inference-throughput').textContent = '-- tokens/sec';
        document.getElementById('inference-fits').textContent = '--';
        document.getElementById('inference-utilization').textContent = '--%';
    }

    /**
     * Calculate multi-node network overhead
     */
    async calculateMultiNode() {
        try {
            const config = {
                model: {
                    num_parameters: document.getElementById('multinode-num-params').value,
                },
                training: {
                    dtype: document.getElementById('multinode-dtype').value,
                    batch_size: parseInt(document.getElementById('multinode-batch-size').value),
                    seq_length: parseInt(document.getElementById('multinode-seq-len').value),
                },
                parallelism: {
                    tensor_parallel_size: parseInt(document.getElementById('multinode-tensor-pp').value),
                    pipeline_parallel_size: parseInt(document.getElementById('multinode-pipeline-pp').value),
                    sequence_parallel: document.getElementById('multinode-seq-parallel').checked,
                },
                engine: {
                    type: document.getElementById('multinode-engine').value,
                    zero_stage: parseInt(document.getElementById('multinode-zero-stage').value),
                },
                node_config: {
                    num_nodes: parseInt(document.getElementById('num-nodes').value),
                    gpus_per_node: parseInt(document.getElementById('gpus-per-node').value),
                    interconnect_type: document.getElementById('interconnect-type').value,
                },
                optimize_strategy: document.getElementById('multinode-optimize').checked,
            };

            const response = await fetch(`${this.apiBase}/multinode/calculate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            });

            if (!response.ok) {
                throw new Error('Failed to calculate multi-node overhead');
            }

            const result = await response.json();
            this.displayMultiNodeResults(result);
        } catch (error) {
            this.showError(`Error: ${error.message}`);
        }
    }

    displayMultiNodeResults(result) {
        const overhead = result.network_overhead;
        document.getElementById('multinode-overhead-total').textContent = `${overhead.total_overhead_gb.toFixed(2)} GB`;
        document.getElementById('multinode-overhead-allreduce').textContent = `${overhead.allreduce_gb.toFixed(2)} GB`;
        document.getElementById('multinode-overhead-allgather').textContent = `${overhead.allgather_gb.toFixed(2)} GB`;
        document.getElementById('multinode-overhead-reducescatter').textContent = `${overhead.reducescatter_gb?.toFixed(2) || '0.00'} GB`;
        document.getElementById('multinode-overhead-pipeline').textContent = `${overhead.pipeline_gb?.toFixed(2) || '0.00'} GB`;
        document.getElementById('multinode-time-overhead').textContent = `${overhead.estimated_overhead_ms_per_step?.toFixed(2) || 'N/A'} ms/step`;
        document.getElementById('multinode-comm-time').textContent = `${overhead.communication_time_ms_per_step?.toFixed(2) || 'N/A'} ms/step`;
        document.getElementById('multinode-latency').textContent = `${overhead.latency_overhead_ms?.toFixed(2) || 'N/A'} ms`;

        // Display suggestions
        const suggestionsDiv = document.getElementById('multinode-suggestions');
        if (result.suggestions && result.suggestions.length > 0) {
            suggestionsDiv.innerHTML = '<ul>' + result.suggestions.map(s => `<li>${s}</li>`).join('') + '</ul>';
        } else {
            suggestionsDiv.innerHTML = '<p>No optimization suggestions available.</p>';
        }
    }

    resetMultiNodeForm() {
        document.getElementById('multinode-preset-select').value = 'custom';
        document.getElementById('multinode-num-params').value = '7B';
        document.getElementById('multinode-dtype').value = 'bf16';
        document.getElementById('num-nodes').value = '2';
        document.getElementById('gpus-per-node').value = '8';
        document.getElementById('multinode-total-gpus').textContent = '16';
        document.getElementById('interconnect-type').value = 'infiniband';
        document.getElementById('multinode-engine').value = 'deepspeed';
        document.getElementById('multinode-zero-stage').value = '3';
        document.getElementById('multinode-batch-size').value = '4';
        document.getElementById('multinode-seq-len').value = '4096';
        document.getElementById('multinode-tensor-pp').value = '1';
        document.getElementById('multinode-pipeline-pp').value = '1';
        document.getElementById('multinode-seq-parallel').checked = false;
        document.getElementById('multinode-optimize').checked = true;

        // Clear results
        document.getElementById('multinode-overhead-total').textContent = '-- GB';
        document.getElementById('multinode-overhead-allreduce').textContent = '-- GB';
        document.getElementById('multinode-overhead-allgather').textContent = '-- GB';
        document.getElementById('multinode-overhead-reducescatter').textContent = '-- GB';
        document.getElementById('multinode-overhead-pipeline').textContent = '-- GB';
        document.getElementById('multinode-time-overhead').textContent = '-- ms/step';
        document.getElementById('multinode-comm-time').textContent = '-- ms/step';
        document.getElementById('multinode-latency').textContent = '-- ms';
        document.getElementById('multinode-suggestions').innerHTML = '<p>Run calculation to see optimization suggestions.</p>';
    }

    /**
     * Show export framework modal
     */
    showExportModal() {
        const format = prompt('Select export format:\n1 - Accelerate\n2 - Lightning\n3 - Axolotl\n4 - DeepSpeed\n5 - YAML\n6 - JSON\n\nEnter number (1-6):');

        if (!format) return;

        const formatMap = {
            '1': 'accelerate',
            '2': 'lightning',
            '3': 'axolotl',
            '4': 'deepspeed',
            '5': 'yaml',
            '6': 'json',
        };

        const selectedFormat = formatMap[format];
        if (!selectedFormat) {
            this.showError('Invalid format selected');
            return;
        }

        this.exportFrameworkConfig(selectedFormat);
    }

    async exportFrameworkConfig(format) {
        try {
            const config = this.collectFormData();
            const response = await fetch(`${this.apiBase}/export/${format}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            });

            if (!response.ok) {
                throw new Error(`Failed to export ${format} config`);
            }

            const result = await response.json();
            this.downloadConfig(result, format);
        } catch (error) {
            this.showError(`Error: ${error.message}`);
        }
    }
}

// Initialize the calculator when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new GPUMemCalculator();
});
