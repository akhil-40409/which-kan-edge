"""Tests for utility functions."""

import pytest

from src.utils import compute_mean, validate_config


class TestComputeMean:
    """Test cases for compute_mean function."""

    def test_compute_mean_positive_values(self) -> None:
        """Test mean calculation with positive values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_mean(values)
        assert result == 3.0

    def test_compute_mean_negative_values(self) -> None:
        """Test mean calculation with negative values."""
        values = [-2.0, -4.0, -6.0]
        result = compute_mean(values)
        assert result == -4.0

    def test_compute_mean_mixed_values(self) -> None:
        """Test mean calculation with mixed positive and negative values."""
        values = [-1.0, 0.0, 1.0]
        result = compute_mean(values)
        assert result == 0.0

    def test_compute_mean_single_value(self) -> None:
        """Test mean calculation with a single value."""
        values = [42.0]
        result = compute_mean(values)
        assert result == 42.0

    def test_compute_mean_empty_list(self) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(
            ValueError, match="Cannot compute mean of empty list"
        ):
            compute_mean([])


class TestValidateConfig:
    """Test cases for validate_config function."""

    def test_validate_config_valid(self) -> None:
        """Test validation with valid configuration."""
        config = {"learning_rate": 0.001, "batch_size": 32}
        assert validate_config(config) is True

    def test_validate_config_with_extra_keys(self) -> None:
        """Test validation with extra keys in configuration."""
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "adam",
        }
        assert validate_config(config) is True

    def test_validate_config_missing_learning_rate(self) -> None:
        """Test validation with missing learning_rate."""
        config = {"batch_size": 32}
        assert validate_config(config) is False

    def test_validate_config_missing_batch_size(self) -> None:
        """Test validation with missing batch_size."""
        config = {"learning_rate": 0.001}
        assert validate_config(config) is False

    def test_validate_config_empty(self) -> None:
        """Test validation with empty configuration."""
        config = {}
        assert validate_config(config) is False
