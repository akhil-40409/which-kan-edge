"""Shared pytest fixtures."""

import jax
import pytest


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def batch_key():
    return jax.random.PRNGKey(1)
