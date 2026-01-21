"""Precision and data type utilities."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Precision:
    """Precision information for a data type."""

    name: str
    bits_per_param: int
    bytes_per_param: float
    is_integer: bool = False


# Standard precision definitions
PRECISION_MAP = {
    "fp32": Precision(name="FP32", bits_per_param=32, bytes_per_param=4.0),
    "fp16": Precision(name="FP16", bits_per_param=16, bytes_per_param=2.0),
    "bf16": Precision(name="BF16", bits_per_param=16, bytes_per_param=2.0),
    "int8": Precision(name="INT8", bits_per_param=8, bytes_per_param=1.0, is_integer=True),
    "int4": Precision(name="INT4", bits_per_param=4, bytes_per_param=0.5, is_integer=True),
}


def get_precision_from_dtype(dtype: str) -> Precision:
    """Get precision info from dtype string.

    Args:
        dtype: Data type string (e.g., "fp32", "fp16", "bf16", "int8", "int4")

    Returns:
        Precision object with bytes per parameter information

    Raises:
        ValueError: If dtype is not supported
    """
    dtype_lower = dtype.lower()
    if dtype_lower not in PRECISION_MAP:
        raise ValueError(
            f"Unsupported dtype: {dtype}. Supported types: {list(PRECISION_MAP.keys())}"
        )
    return PRECISION_MAP[dtype_lower]


def bytes_from_params(num_params: int, dtype: str) -> float:
    """Calculate memory in bytes for a given number of parameters.

    Args:
        num_params: Number of parameters
        dtype: Data type string

    Returns:
        Memory in bytes
    """
    precision = get_precision_from_dtype(dtype)
    return num_params * precision.bytes_per_param


def gb_from_bytes(num_bytes: float) -> float:
    """Convert bytes to gigabytes.

    Args:
        num_bytes: Number of bytes

    Returns:
        Number of gigabytes
    """
    return num_bytes / (1024**3)


def gb_from_params(num_params: int, dtype: str) -> float:
    """Calculate memory in GB for a given number of parameters.

    Args:
        num_params: Number of parameters
        dtype: Data type string

    Returns:
        Memory in GB
    """
    bytes_val = bytes_from_params(num_params, dtype)
    return gb_from_bytes(bytes_val)
