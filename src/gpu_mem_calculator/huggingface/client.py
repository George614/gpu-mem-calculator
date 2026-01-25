"""HuggingFace Hub client for fetching model metadata."""

from typing import Any, cast

import httpx

from gpu_mem_calculator.huggingface.exceptions import (
    HuggingFaceError,
    InvalidConfigError,
    ModelNotFoundError,
    PrivateModelAccessError,
)


class HuggingFaceClient:
    """Client for interacting with HuggingFace Hub API."""

    def __init__(self, token: str | None = None, timeout: int = 30):
        """Initialize HF Hub client.

        Args:
            token: HF API token for private models (optional)
            timeout: HTTP timeout in seconds
        """
        self.token = token
        self.timeout = timeout
        self.api_base = "https://huggingface.co/api"
        self.raw_base = "https://huggingface.co"

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers with optional authentication."""
        headers = {
            "User-Agent": "GPU-Mem-Calculator/0.1.0",
            "Accept": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get model metadata from HF Hub.

        Args:
            model_id: Model identifier (e.g., "meta-llama/Llama-2-7b-hf")

        Returns:
            Model metadata dict

        Raises:
            ModelNotFoundError: If model doesn't exist
            PrivateModelAccessError: If authentication required
            HuggingFaceError: For network issues
        """
        model_id = model_id.strip()
        if not model_id:
            raise ValueError("Model ID cannot be empty")

        # Sanitize model ID
        model_id = model_id.strip("/")

        url = f"{self.api_base}/models/{model_id}"

        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            response = await client.get(url, headers=self._get_headers())

            if response.status_code == 401:
                raise PrivateModelAccessError(
                    f"Authentication required for model '{model_id}'. "
                    "Please provide a HuggingFace token."
                )
            elif response.status_code == 404:
                raise ModelNotFoundError(f"Model '{model_id}' not found on HuggingFace Hub")
            elif response.status_code != 200:
                raise HuggingFaceError(f"Failed to fetch model info: HTTP {response.status_code}")

            return cast(dict[str, Any], response.json())

    async def get_model_config(self, model_id: str) -> dict[str, Any]:
        """Get model config.json from HF Hub.

        Args:
            model_id: Model identifier

        Returns:
            Model configuration dict

        Raises:
            ModelNotFoundError: If model doesn't exist
            PrivateModelAccessError: If authentication required
            InvalidConfigError: If config.json not found
            HuggingFaceError: For network issues
        """
        model_id = model_id.strip().strip("/")

        # Try to fetch config.json from the repository
        url = f"{self.raw_base}/{model_id}/raw/main/config.json"

        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            response = await client.get(url, headers=self._get_headers())

            if response.status_code == 404:
                # Try alternative branches
                for branch in ["base", "research"]:
                    url = f"{self.raw_base}/{model_id}/raw/{branch}/config.json"
                    response = await client.get(url, headers=self._get_headers())
                    if response.status_code == 200:
                        break

                if response.status_code == 404:
                    raise InvalidConfigError(f"config.json not found for model '{model_id}'")
            elif response.status_code == 401:
                raise PrivateModelAccessError(f"Authentication required for model '{model_id}'")
            elif response.status_code != 200:
                raise HuggingFaceError(f"Failed to fetch model config: HTTP {response.status_code}")

            return cast(dict[str, Any], response.json())

    async def fetch_model_metadata(self, model_id: str) -> dict[str, Any]:
        """Fetch complete model metadata including info and config.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary with 'model_info' and 'config' keys

        Raises:
            ModelNotFoundError: If model doesn't exist
            PrivateModelAccessError: If authentication required
            InvalidConfigError: If config.json not found
            HuggingFaceError: For other errors
        """
        model_info = await self.get_model_info(model_id)
        model_config = await self.get_model_config(model_id)

        return {
            "model_info": model_info,
            "config": model_config,
        }
