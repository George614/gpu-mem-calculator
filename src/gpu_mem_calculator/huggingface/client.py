"""HuggingFace Hub client for fetching model metadata."""

import logging
from typing import Any, cast

import httpx

from gpu_mem_calculator.huggingface.exceptions import (
    HuggingFaceError,
    InvalidConfigError,
    ModelNotFoundError,
    PrivateModelAccessError,
)

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """Client for interacting with HuggingFace Hub API."""

    def __init__(self, token: str | None = None, timeout: int = 30):
        """Initialize HF Hub client.

        Args:
            token: HF API token for private models (optional). Users must provide
                   their own token to access gated models. The app will NOT use
                   the Space's HF_TOKEN environment variable for security reasons.
            timeout: HTTP timeout in seconds
        """
        # Only use explicitly provided token, never auto-detect HF_TOKEN env var
        # This prevents all public users from using the Space owner's token
        self.token = token
        if self.token:
            logger.info("Using user-provided HuggingFace authentication token")
        self.timeout = timeout
        self.api_base = "https://huggingface.co/api"
        self.raw_base = "https://huggingface.co"

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers with optional authentication."""
        headers = {
            "User-Agent": "GPU-Mem-Calculator/0.1.0 (https://github.com/George614/gpu-mem-calculator)",
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
        logger.info(f"Fetching model info from HF API: {model_id}")

        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            response = await client.get(url, headers=self._get_headers())

            logger.info(f"HF API response status: {response.status_code} for {model_id}")

            if response.status_code == 401:
                logger.warning(f"Authentication failed for {model_id}")
                raise PrivateModelAccessError(
                    f"Model '{model_id}' requires authentication. "
                    "This is a gated model - please provide your own HuggingFace token "
                    "in the 'HuggingFace Token' field above. Get your token at: "
                    "https://huggingface.co/settings/tokens"
                )
            elif response.status_code == 404:
                raise ModelNotFoundError(f"Model '{model_id}' not found on HuggingFace Hub")
            elif response.status_code != 200:
                logger.error(f"Failed to fetch {model_id}: HTTP {response.status_code}")
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
        logger.info(f"Fetching config.json from: {url}")

        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            response = await client.get(url, headers=self._get_headers())

            logger.info(f"Config fetch status: {response.status_code} for {model_id}")

            if response.status_code == 404:
                # Try alternative branches
                logger.info(f"Config not found on main branch, trying alternatives for {model_id}")
                for branch in ["base", "research"]:
                    url = f"{self.raw_base}/{model_id}/raw/{branch}/config.json"
                    logger.info(f"Trying branch: {branch}")
                    response = await client.get(url, headers=self._get_headers())
                    if response.status_code == 200:
                        logger.info(f"Found config on branch: {branch}")
                        break

                if response.status_code == 404:
                    logger.error(f"config.json not found for {model_id} on any branch")
                    raise InvalidConfigError(f"config.json not found for model '{model_id}'")
            elif response.status_code == 401:
                logger.warning(f"Authentication failed for config fetch of {model_id}")
                raise PrivateModelAccessError(
                    f"Model '{model_id}' requires authentication. "
                    "Please provide your own HuggingFace token in the form above. "
                    "Get your token at: https://huggingface.co/settings/tokens"
                )
            elif response.status_code != 200:
                logger.error(f"Failed to fetch config for {model_id}: HTTP {response.status_code}")
                raise HuggingFaceError(f"Failed to fetch model config: HTTP {response.status_code}")

            logger.info(f"Successfully fetched config for {model_id}")
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
