"""Custom exceptions for HuggingFace Hub integration."""


class HuggingFaceError(Exception):
    """Base exception for HuggingFace-related errors."""

    pass


class ModelNotFoundError(HuggingFaceError):
    """Raised when a model is not found on HuggingFace Hub."""

    pass


class PrivateModelAccessError(HuggingFaceError):
    """Raised when authentication is required for a private model."""

    pass


class InvalidConfigError(HuggingFaceError):
    """Raised when model config is invalid or missing required fields."""

    pass
