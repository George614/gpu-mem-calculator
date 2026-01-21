"""Preset model configurations loader.

This module provides a centralized location for managing model preset
configurations that can be used by both CLI and web interfaces.
"""

import json
from pathlib import Path
from typing import Any

# Base directory for the package
BASE_DIR = Path(__file__).parent.parent.parent.parent


def get_presets_file_path() -> Path:
    """Get the path to the presets JSON file.

    Returns:
        Path to the presets JSON file
    """
    # Check for web/presets/models.json relative to project root
    presets_path = BASE_DIR / "web" / "presets" / "models.json"
    if presets_path.exists():
        return presets_path

    # Fallback to src directory for development installs
    presets_path = BASE_DIR / "src" / "gpu_mem_calculator" / "presets" / "models.json"
    return presets_path


def load_presets() -> dict[str, dict[str, Any]]:
    """Load all preset model configurations.

    Returns:
        Dictionary mapping preset names to their configurations.
        Each preset has: display_name, description, config
    """
    presets_file = get_presets_file_path()

    if not presets_file.exists():
        return {}

    try:
        with presets_file.open("r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def get_preset_config(preset_name: str) -> dict[str, Any] | None:
    """Get a specific preset configuration.

    Args:
        preset_name: Name of the preset to retrieve

    Returns:
        Preset configuration dict, or None if not found
    """
    presets = load_presets()
    preset = presets.get(preset_name)

    if preset is None:
        return None

    # Return just the config part (what the calculator needs)
    return preset.get("config", {})


def list_presets() -> dict[str, dict[str, str]]:
    """List all available presets with metadata.

    Returns:
        Dictionary mapping preset names to their display metadata.
        Each entry has: display_name, description
    """
    presets = load_presets()
    return {
        name: {
            "display_name": preset.get("display_name", name),
            "description": preset.get("description", ""),
        }
        for name, preset in presets.items()
    }
