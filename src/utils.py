"""Utility functions for the research project.

This module contains common utility functions used across experiments.
"""

from typing import Any, Dict, List


def compute_mean(values: List[float]) -> float:
    """Compute the mean of a list of values.

    Args:
        values: List of numerical values.

    Returns:
        The arithmetic mean of the input values.

    Raises:
        ValueError: If the input list is empty.

    Example:
        >>> compute_mean([1.0, 2.0, 3.0, 4.0])
        2.5
    """
    if not values:
        raise ValueError("Cannot compute mean of empty list")
    return sum(values) / len(values)


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate experiment configuration.

    Args:
        config: Configuration dictionary to validate.

    Returns:
        True if configuration is valid, False otherwise.

    Example:
        >>> config = {"learning_rate": 0.001, "batch_size": 32}
        >>> validate_config(config)
        True
    """
    required_keys = ["learning_rate", "batch_size"]
    return all(key in config for key in required_keys)
