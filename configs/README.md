# Model Configuration Files

**Note:** These configuration files are maintained for backward compatibility.
For new projects, consider using the built-in model presets instead.

## Using Presets (Recommended)

The GPU Memory Calculator includes pre-configured model presets that can be used
directly from both CLI and web interfaces:

### CLI Usage:
```bash
# List all available presets
gpu-mem-calc presets

# Calculate with a preset
gpu-mem-calc calculate --preset llama2-7b
gpu-mem-calc calculate --preset mixtral-8x7b

# List presets in table format
gpu-mem-calc presets --format table
```

### Web Interface:
Open the web UI and select from the preset dropdown menu.

## Available Presets

- **Dense Models**: LLaMA 2 (7B, 13B, 70B), GPT-3 (175B)
- **MoE Models**: Mixtral 8x7B, GLM-4 (9B), GLM-4.7 (355B), GLM-4.5 Air (106B),
  Qwen1.5-MoE-A2.7B, DeepSeek-MoE (16B)

## Adding Custom Presets

To add a new model preset, edit `web/presets/models.json`:

```json
{
  "my-model": {
    "display_name": "My Model 10B",
    "description": "Custom model configuration",
    "config": {
      "model": { ... },
      "training": { ... },
      "parallelism": { ... },
      "engine": { ... },
      "hardware": { ... }
    }
  }
}
```

Then use it:
```bash
gpu-mem-calc calculate --preset my-model
```

## Custom Config Files

For custom configurations not suitable as presets, you can still create
individual JSON config files in this directory following the format shown
in the examples below.
