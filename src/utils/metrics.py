"""Shared metrics and parameter counting."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def count_params(params: Any) -> int:
    """Count scalar trainable parameters in a pytree."""
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(int(leaf.size) for leaf in leaves))


def rmse(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Root mean squared error."""
    return jnp.sqrt(jnp.mean((pred - target) ** 2))
