"""Dataset helpers: Feynman + special functions with train/val/test splits."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from eigenflow.datasets.feynman import FEYNMAN_EQUATIONS, FeynmanDatasetGenerator
from eigenflow.datasets.special import SPECIAL_FUNCTIONS, SpecialFunctionGenerator


def train_val_test_split(
    X: jax.Array,
    y: jax.Array,
    key: jax.Array,
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Dict[str, jax.Array]:
    """Shuffle and split into train / val / test dict."""
    n = X.shape[0]
    perm = jax.random.permutation(key, n)
    X, y = X[perm], y[perm]
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "x_train": X[:n_train],
        "y_train": y[:n_train],
        "x_val": X[n_train : n_train + n_val],
        "y_val": y[n_train : n_train + n_val],
        "x_test": X[n_train + n_val :],
        "y_test": y[n_train + n_val :],
    }


def make_dataset(
    task_id: str,
    key: jax.Array,
    *,
    n_samples: int = 2000,
    noise_level: float = 0.0,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    input_scaling: str = "minmax_11",
    target_scaling: str = "standardize",
) -> Dict[str, jax.Array]:
    """Build a split dataset from a Feynman equation id or special-function id."""
    k_data, k_split = jax.random.split(key)
    if task_id in FEYNMAN_EQUATIONS:
        gen = FeynmanDatasetGenerator(task_id)
        noise_type = "gaussian" if noise_level > 0 else "none"
        X, y, _ = gen.generate(
            k_data,
            n_samples,
            noise_level=noise_level,
            noise_type=noise_type,
            input_scaling=input_scaling,
            target_scaling=target_scaling,
        )
    elif task_id in SPECIAL_FUNCTIONS:
        gen = SpecialFunctionGenerator(task_id)
        X, y, _ = gen.generate(
            k_data,
            n_samples,
            input_scaling=input_scaling,
            target_scaling=target_scaling,
            noise_level=noise_level,
        )
    else:
        raise ValueError(
            f"Unknown task {task_id!r}. "
            f"Feynman: {list(FEYNMAN_EQUATIONS)}; "
            f"special: {list(SPECIAL_FUNCTIONS)}"
        )
    data = train_val_test_split(
        X, y, k_split, train_frac=train_frac, val_frac=val_frac
    )
    data["task_id"] = task_id
    data["n_features"] = int(X.shape[1])
    return data


__all__ = [
    "FEYNMAN_EQUATIONS",
    "FeynmanDatasetGenerator",
    "SPECIAL_FUNCTIONS",
    "SpecialFunctionGenerator",
    "train_val_test_split",
    "make_dataset",
]
