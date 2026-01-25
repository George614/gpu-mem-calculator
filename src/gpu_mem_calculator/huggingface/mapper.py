"""Map HuggingFace model configs to GPU Memory Calculator ModelConfig."""

from typing import Any


class HuggingFaceConfigMapper:
    """Map HF config.json to ModelConfig."""

    # Mapping of HF config field names to ModelConfig fields
    FIELD_MAPPINGS = {
        # Direct mappings
        "hidden_size": "hidden_size",
        "num_hidden_layers": "num_layers",
        "num_attention_heads": "num_attention_heads",
        "vocab_size": "vocab_size",
        "max_position_embeddings": "max_seq_len",
        # Common alternatives
        "n_layer": "num_layers",
        "n_head": "num_attention_heads",
        "n_embd": "hidden_size",
        "n_positions": "max_seq_len",
    }

    def map_to_model_config(
        self, hf_config: dict[str, Any], model_info: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Map HF config to ModelConfig-compatible dictionary.

        Args:
            hf_config: HuggingFace config.json dict
            model_info: Optional model metadata from HF API

        Returns:
            Dictionary with keys:
                - 'config': ModelConfig-compatible dict
                - 'missing_fields': List of required fields not found
                - 'found_fields': List of fields that were mapped
        """
        model_config = {}
        missing_fields = []
        found_fields = []

        # Extract model name
        if model_info:
            model_config["name"] = model_info.get("modelId", "custom").replace("/", "-")
        else:
            model_config["name"] = "custom-hf-model"

        # Map simple fields
        for hf_field, our_field in self.FIELD_MAPPINGS.items():
            if hf_field in hf_config:
                value = hf_config[hf_field]
                # Ensure type is int
                if isinstance(value, (int, float)):
                    model_config[our_field] = int(value)
                    found_fields.append(our_field)
                elif isinstance(value, str):
                    # Handle special cases like "32B"
                    if our_field == "num_parameters":
                        model_config[our_field] = value
                        found_fields.append(our_field)

        # Extract MoE-specific fields
        moe_config = self._extract_moe_config(hf_config)
        if moe_config:
            model_config.update(moe_config)
            found_fields.extend(moe_config.keys())

        # Handle num_parameters - compute if not provided
        if "num_parameters" not in model_config:
            # Try to compute from architecture
            computed_params = self._estimate_num_parameters(hf_config, model_config)
            if computed_params:
                model_config["num_parameters"] = computed_params
                found_fields.append("num_parameters")

        # Identify missing fields
        required_fields = [
            "num_parameters",
            "num_layers",
            "hidden_size",
            "num_attention_heads",
            "vocab_size",
            "max_seq_len",
        ]

        for field in required_fields:
            if field not in model_config:
                missing_fields.append(field)

        return {
            "config": model_config,
            "missing_fields": missing_fields,
            "found_fields": found_fields,
        }

    def _extract_moe_config(self, hf_config: dict[str, Any]) -> dict[str, Any]:
        """Extract MoE-specific configuration from HF config.

        Args:
            hf_config: HuggingFace config

        Returns:
            Dict with moe_enabled, num_experts, top_k if MoE detected
        """
        moe_config: dict[str, Any] = {}

        # Common HF MoE field names
        num_experts_val = hf_config.get(
            "num_local_experts",
            hf_config.get("num_experts", hf_config.get("n_expert")),
        )
        top_k_val = hf_config.get(
            "expert_capacity",
            hf_config.get("num_experts_per_tok", hf_config.get("top_k")),
        )

        # Type narrowing for MoE fields
        if isinstance(num_experts_val, (int, float)) and num_experts_val > 1:
            moe_config["moe_enabled"] = True
            moe_config["num_experts"] = int(num_experts_val)

            if isinstance(top_k_val, (int, float)):
                moe_config["top_k"] = int(top_k_val)
            else:
                moe_config["top_k"] = 2  # Default

        return moe_config

    def _estimate_num_parameters(
        self, hf_config: dict[str, Any], partial_config: dict[str, Any]
    ) -> int | None:
        """Estimate number of parameters if not provided.

        Args:
            hf_config: Full HF config
            partial_config: Partially built config

        Returns:
            Estimated parameter count or None
        """
        # Check if HF provides the count directly
        if "num_parameters" in hf_config:
            return int(hf_config["num_parameters"])

        # Try to compute from model architecture
        hidden_size = partial_config.get("hidden_size")
        num_layers = partial_config.get("num_layers")
        vocab_size = partial_config.get("vocab_size")

        # Type narrowing for calculation
        if (
            isinstance(hidden_size, int)
            and isinstance(num_layers, int)
            and isinstance(vocab_size, int)
        ):
            # Rough estimate for transformer models
            # Based on: embeddings + transformer layers
            embedding_params = vocab_size * hidden_size
            layer_params = 4 * hidden_size * hidden_size * num_layers  # FFN + attention
            total = embedding_params + layer_params

            # Apply scaling factor for real-world variance
            # (accounting for biases, layernorm, etc.)
            return int(total * 1.2)

        return None
