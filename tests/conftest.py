"""Test configuration and fixtures for pytest."""

import pytest


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing.

    Returns:
        dict: A dictionary with sample configuration values.
    """
    return {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "optimizer": "adam",
    }


@pytest.fixture
def sample_data():
    """Provide sample data for testing.

    Returns:
        list: A list of sample numerical values.
    """
    return [1.0, 2.0, 3.0, 4.0, 5.0]
